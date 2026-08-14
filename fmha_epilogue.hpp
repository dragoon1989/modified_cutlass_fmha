/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "../collective/fmha_common.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/copy_traits_sm80.hpp"
// 引用SIMT版本的epilogue
#include "cutlass/epilogue/collective/collective_epilogue.hpp"

namespace cutlass::fmha::collective {

template<class Element, class ElementAccumulator, class TileShape_WG>
struct FmhaFwdEpilogue {

  static constexpr int Alignment = 16 / sizeof(Element);

  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<Element, ElementAccumulator, void>;
  using CollectiveEpilogueTMA = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape_WG, Shape<_1,_1,_1>, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      void, cute::tuple<int, _1, cute::tuple<int, int>>, Alignment,
      Element, cute::tuple<int, _1, cute::tuple<int, int>>, Alignment,
      cutlass::epilogue::TmaWarpSpecialized,
      DefaultOperation
  >::CollectiveOp;

  struct Arguments {
    Element* ptr_O;
    cute::tuple<int, cute::_1, cute::tuple<int, int>> dO;

    ElementAccumulator* ptr_LSE;
    cute::tuple<cute::_1, cute::tuple<int, int>> dLSE;
  };

  struct Params {
    ElementAccumulator* ptr_LSE;
    cute::tuple<cute::_1, cute::tuple<int, int>> dLSE;
  
    typename CollectiveEpilogueTMA::Params epilogue_TMA;
  };

  using TensorStorage = typename CollectiveEpilogueTMA::TensorStorage;
  using PipelineStorage = typename CollectiveEpilogueTMA::PipelineStorage;
  using LoadPipeline = typename CollectiveEpilogueTMA::LoadPipeline;
  static constexpr int TmaTransactionBytes = CollectiveEpilogueTMA::TmaTransactionBytes;

  template<class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_size, Arguments const& args, void* workspace = nullptr) {
    auto problem_size_o = make_shape(get<2>(problem_size), get<4>(problem_size), 1,
              make_shape(get<0>(problem_size), get<1>(problem_size)));
    typename CollectiveEpilogueTMA::Arguments args_tma{{}, args.ptr_O, args.dO, args.ptr_O, args.dO};
    return Params{
      args.ptr_LSE, args.dLSE,
      CollectiveEpilogueTMA::to_underlying_arguments(problem_size_o, args_tma, workspace)
    };
  }

  template<class TileShape, class BlkCoord, class ResultTuple, class TiledMma, class ProblemShape>
  CUTLASS_DEVICE void operator()(
      TileShape const& tile_shape, BlkCoord const& blk_coord,
      ResultTuple const& result, TiledMma const& tiled_mma,
      ProblemShape const& problem_size, Params const& params,
      LoadPipeline epi_load_pipeline,
      TensorStorage& epi_tensor_storage)
  {
    using X = Underscore;

    auto acc = get<0>(result);
    auto lse = get<1>(result);
  
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  
    int seqlen_q = get<2>(problem_size);
    int num_batch = get<0>(problem_size);
    int num_heads = get<1>(problem_size);
    // Epilogue for lse
    Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE),
        make_shape(seqlen_q, get<1>(tile_shape), make_shape(num_batch, num_heads)),
        make_stride(_1{}, _0{}, get<1>(params.dLSE)));
    Tensor gLSE_full = local_tile(mLSE, tile_shape, make_coord(_, _, _), Step<_1, _1, X>{});
    Tensor gLSE = gLSE_full(_, _, get<0>(blk_coord), get<1>(blk_coord), get<2>(blk_coord));
    Tensor tOgLSE = thr_mma.partition_C(gLSE);
    Tensor cO = make_identity_tensor(take<0,2>(tile_shape));
    Tensor tOcO = thr_mma.partition_C(cO);
    if (get<1>(tOcO(_0{})) == 0) {
      auto tOgLSE_mn = make_tensor(tOgLSE.data(), layout_acc_mn(tiled_mma, tOgLSE.layout()));
      auto tOcO_mn = make_tensor(tOcO.data(), layout_acc_mn(tiled_mma, tOcO.layout()));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<0>(tOgLSE_mn); i++) {
        if (get<0>(tOcO_mn(i)) + get<0>(blk_coord) * get<0>(tile_shape) < get<2>(problem_size)) {
          tOgLSE_mn(i, _0{}) = lse(i);
        }
      }
    }
    auto problem_size_o = make_shape(get<2>(problem_size), get<4>(problem_size), _,
              make_shape(get<0>(problem_size), get<1>(problem_size)));

    CollectiveEpilogueTMA epilogue_tma(params.epilogue_TMA, epi_tensor_storage);

    using EpiStorePipeline = typename CollectiveEpilogueTMA::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename CollectiveEpilogueTMA::LoadPipelineState epi_load_pipe_consumer_state;
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
    epilogue_tma.store(
      epi_load_pipeline, epi_load_pipe_consumer_state,
      epi_store_pipeline, epi_store_pipe_producer_state,
      problem_size_o, tile_shape, make_coord(get<0>(blk_coord), _0{}, _, get<2>(blk_coord)),
      acc, tiled_mma, threadIdx.x % cutlass::NumThreadsPerWarpGroup,
      epi_tensor_storage
    );

    epilogue_tma.store_tail(
      epi_load_pipeline, epi_load_pipe_consumer_state_next,
      epi_store_pipeline, epi_store_pipe_producer_state_next
    );
  }
};

// 我们直接在SM90的原本epilogue实现后面新增一个针对SIMT cp.async的实现
// TileShape是PV阶段的CTA tile shape (bQ, bD, bK)
// 这里有一个问题，我们必须保证epilogue阶段与mainloop阶段的thread数一致
// 在SM90原版代码中，这是通过collective builder来成对选择mainloop和ep
// 来保证协调的，但是在我们这里，必须额外传递一个参数
template<class Element, class ElementAccumulator, class TileShape, 
         int kThreadNum>
struct FmhaFwdEpilogueCpAsync {
  // GEMM的epilogue环节对output tensor的rank有要求，不是跟FMHA一致的
  using StrideOFlat = cute::Stride<int64_t, cute::_1, int64_t>;

  // 既然FMHA中实际上不执行linear combination，这个ep op实际上就只剩一个cast功能了
  // 因为输出O还是Element精度，而累加器acc是ElementAccumulator精度，确实需要这么
  // 一个cast
  using ThreadEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    Element, 1, ElementAccumulator, ElementAccumulator,
    cutlass::epilogue::thread::ScaleType::Nothing>;

  // TODO: 如何选择smem layout
  // 仿照mainloop的做法，我们在smem上先通过R2S获得与输出tile的mn-layout一致的结果
  // 然后再用flat的方式S2G就OK了（但是cp.async不支持S2G，还是得变成S2R再R2G）
  // 重点在于R2S到mn-layout时如何避免bank conflict
  // 这取决于R2S的atom，由于SM80的硬件上不支持stmatrix，所以使用最简单的STS，
  // 也就是universal copy作为atom
  // 每个thread持有的O矩阵的mn坐标需要根据warp level mma的fragment layout查看
  // 在我们已经实现的collective API中，只支持Element = F16, ElementAccumulator = F32
  // 不过看PTX文档，无论是F16还是F32累加，layout倒是一样的
  // 如果把acc表示为(MMA_VAL, ITER_M, ITER_N)，则ITER_M是沿着seqlen_q的迭代次数，在
  // 本例中就是恒定的1
  // ITER_N是沿着head-dim维度（D）迭代的次数
  // 而MMA_VAL则是warp level mma atom的val-layout，对于mma.m16n8k16而言就是(2,2)
  // 而我们的output tensor的mn-layout（即O的layout）应该是(Q,D):(x,1)
  // 也就是N-major/row-major
  // 由于我们的PV阶段mma的Perm-n=8*2，实际上还是个block tile形式的perm，所以每个
  // thread持有的ITER_N个MMA_VAL在mn-layout上是不连续的
  // 只能按照一条STS写回半个MMA_VAL的粒度来操作，由于R2S发生在ep op之前，所以
  // STS一次写回2个F32，vector width = 64bit，嗯，也还好
  // 这样，一个warp的STS.64需要拆成2个half-warp来考虑bank conflict，每个half-warp
  // 写1024bit，对应于O tensor上的一个4行8列的子阵（m4n8），注意O是n-major的
  // 则确定swizzle的M = log2(64bit/32bit) = 1
  // 我们可以仿照mainloop里的做法来设计：
  template <int N>
  CUTE_HOST_DEVICE
    static constexpr auto make_n_major_smem_atom(Int<N>) {
    // 注：这个设计能够确保N=16,64,128,256都是bank conflict free的
    //     但是N=32时，会产生2-way bank conflict
    constexpr int n8 = 8;
    // S2R的位宽是128bit=16B，R2S则是限制在64bit=8B，取二者的最大值
    constexpr int kVec = 16 / sizeof(ElementAccumulator);
    // 由此确定kM
    constexpr int kM = log_2(uint32_t(kVec));   // 也就是2
    // 根据m4n8的子阵大小（32）确定kB
    constexpr int kB = log_2(32u) - kM;
    // 根据SRAM的位宽（1024bit=128B）确定kS
    constexpr int kS = log_2(static_cast<uint32_t>(128 / sizeof(ElementAccumulator))) - kM;

    // over
    return composition(
      Swizzle<kB, kM, kS>{},
      Layout<Shape<_4, Shape<Int<n8>, Int<N/n8>>>, // m4n8 subtile
      Stride< Int<n8>, Stride<    _1, Int<4*n8>>>
      >{}
    );
  }

  // 另外有一点要指出：epilogue的SmemLayout不一定与CTA tile等大
  // 因为ep过程是elementwise的，相互之间没有依赖，如果smem资源不够
  // 也可以分块进行R->S->R->G的过程，这在Epilogue中是自动handle的
  // 注意：TileShape是3维的，我们只要前2维
  using SmemLayoutO = decltype(tile_to_shape(
    make_n_major_smem_atom(get<1>(TileShape{})), take<0,2>(TileShape{})));
  // 如上，R2S我们就用最简单的STS.64即可
  using CopyAtomR2S = Copy_Atom<UniversalCopy<uint64_t>, ElementAccumulator>;

  //////////////////////////////////////////////////////////////////////////////////////
  // 这一阶段可以使用更大位宽的LDS.128了
  using CopyAtomS2R = Copy_Atom<UniversalCopy<uint128_t>, ElementAccumulator>;

  template <class CopyAtom, int kX, int kN>
  CUTE_HOST_DEVICE static constexpr auto
    make_n_major_copy(Int<kX>, Int<kN>) {
    constexpr int kValNum = 8; // 8*F32 for we'd like 8 elem per thread
    constexpr int kThrN = cute::min(kThreadNum, kN / kValNum);
    constexpr int kThrX = kThreadNum / kThrN;

    static_assert(kN % kValNum == 0, "连续维必须被8整除");
    static_assert(kThreadNum % kThrN == 0,
      "CTA 线程数必须能被连续维方向的线程数整除");
    static_assert(kN % (kThrN * kValNum) == 0,
      "连续维必须能被一次 TiledCopy 的宽度整除（仅当线程数不足以铺满一行时才可能触发）");
    static_assert(kX % kThrX == 0,
      "非连续维必须能被该方向的线程数整除，否则 partition 会在 evenly_divides 上失败");

    return make_tiled_copy(
      CopyAtom{},
      Layout<Shape <Int<kThrX>, Int<kThrN>>,  // thread-layout
             Stride<Int<kThrN>, _1>>{},       // N-major
      Layout<Shape<_1, Int<kValNum>>>{}       // val-layout
    );
  }

  using TiledCopyS2R = decltype(make_n_major_copy<CopyAtomS2R>(
    get<0>(TileShape{}), get<1>(TileShape{})));

  // 如前所述，在R2G阶段，由于此前的S2R阶段将每线程的val-layout扩充到两倍atom
  // 所以有更多的元素可以写回global，使得我们仍然可以用更大位宽的STG.128
  using CopyAtomR2G = Copy_Atom<UniversalCopy<uint128_t>, Element>;
  //////////////////////////////////////////////////////////////////////////////////////

  // 执行流程是：R2S -> S2R -> ep op -> R2G
  using EpilogueSimt = cutlass::epilogue::collective::Epilogue<
    StrideOFlat, StrideOFlat, ThreadEpilogueOp,
    SmemLayoutO, CopyAtomR2S, TiledCopyS2R, CopyAtomR2G>;

  // 注意kernel API看到的这个type叫做TensorStorage，所以做个alias
  using TensorStorage = typename EpilogueSimt::SharedStorage;

  struct Arguments {
    Element* ptr_O;
    cute::tuple<int, cute::_1, cute::tuple<int, int>> dO;
    ElementAccumulator* ptr_LSE;
    cute::tuple<cute::_1, cute::tuple<int, int>> dLSE;
  };

  struct Params {
    ElementAccumulator* ptr_LSE;
    cute::tuple<cute::_1, cute::tuple<int, int>> dLSE;
    typename EpilogueSimt::Params epilogue_simt;
  };

  template<class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_size,
    Arguments const& args, void* workspace = nullptr) {

    // L 压平：(B,H) + (H·Q·D, Q·D)  →  B·H + Q·D，索引 l = h + H·b
    auto dO_flat = make_stride(int64_t(get<0>(args.dO)), _1{}, int64_t(get<2, 1>(args.dO)));
    typename EpilogueSimt::Arguments args_simt{
        {},                      // thread{}: alpha/beta 在 ScaleType::Nothing 下无意义
        nullptr, dO_flat,        // ptr_C = nullptr，永不解引用
        args.ptr_O, dO_flat };
    return Params{ args.ptr_LSE, args.dLSE,
                   EpilogueSimt::to_underlying_arguments(problem_size, args_simt, workspace) };
  }

  template<class TileShapeMNK, class BlkCoord, class ResultTuple, class TiledMma, class ProblemShape>
  CUTLASS_DEVICE void operator()(
    TileShapeMNK const& tile_shape, BlkCoord const& blk_coord,
    ResultTuple const& result, TiledMma const& tiled_mma,
    ProblemShape const& problem_size, Params const& params,
    TensorStorage& storage)
  {
    using X = Underscore;

    auto acc = get<0>(result);
    auto lse = get<1>(result);

    // ---- LSE：与 SM90 版本逐字相同（fmha_epilogue.hpp:237-254），纯 STG ----
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    int seqlen_q = get<2>(problem_size);
    int num_batch = get<0>(problem_size);
    int num_heads = get<1>(problem_size);
    // Epilogue for lse
    Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE),
      make_shape(seqlen_q, get<1>(tile_shape), make_shape(num_batch, num_heads)),
      make_stride(_1{}, _0{}, get<1>(params.dLSE)));
    Tensor gLSE_full = local_tile(mLSE, tile_shape, make_coord(_, _, _), Step<_1, _1, X>{});
    Tensor gLSE = gLSE_full(_, _, get<0>(blk_coord), get<1>(blk_coord), get<2>(blk_coord));
    Tensor tOgLSE = thr_mma.partition_C(gLSE);
    Tensor cO = make_identity_tensor(take<0, 2>(tile_shape));
    Tensor tOcO = thr_mma.partition_C(cO);
    if (get<1>(tOcO(_0{})) == 0) {
      auto tOgLSE_mn = make_tensor(tOgLSE.data(), layout_acc_mn(tiled_mma, tOgLSE.layout()));
      auto tOcO_mn = make_tensor(tOcO.data(), layout_acc_mn(tiled_mma, tOcO.layout()));
      CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(tOgLSE_mn); i++) {
          if (get<0>(tOcO_mn(i)) + get<0>(blk_coord) * get<0>(tile_shape) < get<2>(problem_size)) {
            tOgLSE_mn(i, _0{}) = lse(i);
          }
        }
    }

    // ---- ep 与 mainloop 共用smem
    // 所以我们必须确保所有warp上的mainloop已经完成了，才能触碰smem ----
    __syncthreads();

    auto bh = get<2>(blk_coord);                              // (b, h)
    int  l_coord = get<1>(bh) + num_heads * get<0>(bh);       // h + H·b

    // 直接调用epilogue
    EpilogueSimt epilogue{ params.epilogue_simt };
    epilogue(
      make_shape(get<2>(problem_size), get<4>(problem_size), _,
        get<0>(problem_size) * num_heads),                 // (M, N, _, L=B·H)
      tile_shape,                                          // (bQ, bD, bKV)
      make_coord(get<0>(blk_coord), _0{}, _, l_coord),     // 使用压平后的l-coord来索引原来的(B,H)
      acc,            // mainloop的结果寄存器
      tiled_mma,      // mainloop的PV mma
      make_tuple(get<2>(problem_size) - get<0>(blk_coord) * get<0>(tile_shape),
                 get<4>(problem_size), _0{}), // residue-mnk，原则上应该是3维，但是ep
                                              // 实际上不使用K维度，所以只要前两维就够了
                                              // 注意:residue记录当前cta tile到边界的距离
      int(threadIdx.x),
      reinterpret_cast<char*>(&storage));
  }
};
}  // namespace cutlass::fmha::collective
