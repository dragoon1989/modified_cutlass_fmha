// 这个文件也是我们新加的，用于实现不依赖TMA的kernel API
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/arch.h"

#include "../kernel/fmha_tile_scheduler.hpp"
#include "../kernel/fmha_options.hpp"

namespace cutlass::fmha::kernel {

  // 仿照原来的FmhaKernelTma实现这个template
  // 整体来看，kernel API的template设计基本已经与arch解耦了
  // 是一个比较干净的generic template，需要的改动并不多
  // 核心实现都是通过传入的模板参数表示的
  template<
    class CollectiveMainloop,
    class CollectiveEpilogue,
    class... Options
  >
  struct FmhaKernelCpAsync {

    // Options
    static constexpr int kBlocksPerSM = find_option_t<Tag::kBlocksPerSM, Int<2>, Options...>::value;

    using Element = typename CollectiveMainloop::Element;
    using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;

    using TileScheduler = IndividualTileScheduler;

    /*using StagesQ = typename CollectiveMainloop::StagesQ;
    using Stages = typename CollectiveMainloop::Stages;*/

    using TileShape = typename CollectiveMainloop::TileShape;
    using ClusterShape = typename CollectiveMainloop::ClusterShape;

    // 我们实现的collective API里面不带这些类型信息了，去掉
    //using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    //using MainloopPipelineQ = typename CollectiveMainloop::MainloopPipelineQ;

    using SmemLayoutQ = typename CollectiveMainloop::SmemLayoutQ;
    using SmemLayoutK = typename CollectiveMainloop::SmemLayoutK;

    struct SharedStorage {
      union {
        typename CollectiveMainloop::SharedStorage mainloop;
        typename CollectiveEpilogue::TensorStorage epilogue;
      };

      // 这几个类型似乎完全没有用处，更上层的device API也不会来引用
      //using PipelineStorage = typename MainloopPipeline::SharedStorage;
      //using PipelineStorageQ = typename MainloopPipelineQ::SharedStorage;
      //using PipelineStorage = typename CollectiveMainloop::SharedStorage;
      //using PipelineStorageQ = typename CollectiveMainloop::SharedStorage;
      //alignas(16) PipelineStorage pipeline_storage;
      //alignas(16) PipelineStorageQ pipeline_storage_q;

      // 不需要使用Pipeline了
      //using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      //alignas(16) EpiLoadPipelineStorage epi_load;
    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    using ProblemShape = cute::tuple<int, int, int, int, int>;

    // 我们的mainloop和epilogue的Arguments都没做修改，跟SM90原版一致
    // 这样main函数里传递参数的代码也就不需要动了
    struct Arguments {
      ProblemShape problem_size;
      typename CollectiveMainloop::Arguments mainloop;
      typename CollectiveEpilogue::Arguments epilogue;
      KernelHardwareInfo hw_info;
    };

    struct Params {
      ProblemShape problem_size;
      typename CollectiveMainloop::Params mainloop;
      typename CollectiveEpilogue::Params epilogue;
      typename TileScheduler::Params tile_scheduler;
    };

    //using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename CollectiveMainloop::PipelineState;
    //using PipelineParamsQ = typename MainloopPipelineQ::Params;
    using PipelineStateQ = typename CollectiveMainloop::PipelineStateQ;

    static const int MinBlocksPerMultiprocessor = kBlocksPerSM;
    static const int MaxThreadsPerBlock = CollectiveMainloop::MaxThreadsPerBlock;
    // 这里，我们不再使用SM90
    // 不过这个tag是提供给更上层的device API的，它会根据架构的不同选择
    // 不同的kernel launch方式（>=90会使用ClusterLauncher::launch()的方式，
    // 否则使用传统的device_kernel<Kernel><<<...>>>这种封装）
    // 所以这里假设我们换成Ampere架构（SM80）
    //using ArchTag = cutlass::arch::Sm90;
    using ArchTag = cutlass::arch::Sm80;

    static size_t get_workspace_size(Arguments const& args) { return 0; }
    static cutlass::Status initialize_workspace(Arguments const&, void*, cudaStream_t) {
      return cutlass::Status::kSuccess;
    }

    static bool can_implement(Arguments const& args) {
      return CollectiveMainloop::can_implement(args.problem_size, args.mainloop);
    }

    static dim3 get_grid_shape(Params const& params) {
      return TileScheduler::get_grid_shape(params.tile_scheduler);
    }

    static dim3 get_block_shape() {
      dim3 block(MaxThreadsPerBlock, 1, 1);
      return block;
    }

    static Params to_underlying_arguments(Arguments const& args, void* workspace) {
      return Params{
          args.problem_size,
          CollectiveMainloop::to_underlying_arguments(args.problem_size, args.mainloop, workspace),
          CollectiveEpilogue::to_underlying_arguments(args.problem_size, args.epilogue, workspace),
          TileScheduler::to_underlying_arguments(args.problem_size, args.hw_info, ClusterShape{}, TileShape{})
      };
    }

    // 执行入口
    CUTLASS_DEVICE void operator()(const Params& params, char* smem) {

      TileScheduler tile_scheduler{ params.tile_scheduler };

      // Shared memory.
      auto& storage = *reinterpret_cast<SharedStorage*>(smem);

      //int thread_idx = int(threadIdx.x);

      //uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

      //int warp_idx = cutlass::canonical_warp_idx_sync();
      //int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
      //int lane_predicate = cute::elect_one_sync();

      // 这里确实与TMA实现耦合了
      // cpasync不需要descriptor，所以可以直接移除这里的prefetch
      // Issue Tma Descriptor Prefetch from a single thread
      /*if ((warp_idx == 0) && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      }*/

      // 这些跟TMA有关的东西都不需要了
      //PipelineParamsQ pipeline_params_q;
      //pipeline_params_q.transaction_bytes = size(SmemLayoutQ{}(_, _, _0{})) * sizeof(Element); // Q
      //pipeline_params_q.role = MainloopPipelineQ::ThreadCategory::ProducerConsumer;
      //pipeline_params_q.is_leader = warp_group_thread_idx == 0;
      //pipeline_params_q.num_consumers = cutlass::NumThreadsPerWarpGroup;

      //PipelineParams pipeline_params;
      //pipeline_params.transaction_bytes = size(SmemLayoutK{}(_, _, _0{})) * sizeof(Element); // KV
      //pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
      //pipeline_params.is_leader = warp_group_thread_idx == 0;
      //pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup;

      // 现在pipeline机制已经徒有其表了，我们也没必要在这里构造
      //MainloopPipelineQ pipeline_q(storage.pipeline_storage_q, pipeline_params_q, Shape<_1, _1, _1>{});
      //MainloopPipeline pipeline(storage.pipeline_storage, pipeline_params, ClusterShape{});

      //using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
      //typename EpiLoadPipeline::Params epi_load_pipeline_params;
      //epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::ProducerConsumer;
      //epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
      //epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
      //epi_load_pipeline_params.consumer_arv_count = NumThreadsPerWarpGroup;
      //epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
      //EpiLoadPipeline epi_load_pipeline(storage.epi_load, epi_load_pipeline_params);

      // State variables used for iterating the circular buffer
      // smem_pipe_read / release is used by the consumer of SMEM data - i.e MMA
      // smem_pipe_write is used by the producer of SMEM data - i.e TMA
      PipelineState smem_pipe_read; // (index = 0, phase = 0)
      // SM90原版需要构造的是奇偶校验pipelinestate，用于mbarrier的正确使用
      // SM80中的所谓pipelinestate实际只剩下一个迭代规划/计数功能，不再需要phase信息
      // 所以可以直接构造，不需要专门初始化其phase了
      //PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>(); // (index=0, phase=1)
      PipelineState smem_pipe_write;

      PipelineStateQ smem_pipe_read_q;
      //PipelineStateQ smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();
      PipelineStateQ smem_pipe_write_q;

      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer blocks in the Cluster
      // and to finish smem init
      // 这些都用不上了（因为我们既不使用cluster，也不用mbarrier）
      /*if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
      }
      else {
        __syncthreads();
      }*/

      auto blk_coord = tile_scheduler.get_block_coord();

      CollectiveMainloop collective_mainloop;
      auto result = collective_mainloop.compute(
        blk_coord, 
        params.mainloop, 
        params.problem_size,
        smem_pipe_read, smem_pipe_write,
        smem_pipe_read_q, smem_pipe_write_q,
        storage.mainloop
      );

      CollectiveEpilogue epilogue;
      epilogue(typename CollectiveMainloop::TileShapePV{}, 
        blk_coord,
        result, typename CollectiveMainloop::TiledMmaPV{},
        params.problem_size, params.epilogue,
        storage.epilogue);

      // over
      return;
    }
  };

}  // namespace cutlass::fmha::kernel
