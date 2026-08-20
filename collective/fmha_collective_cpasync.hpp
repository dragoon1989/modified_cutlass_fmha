// this file is added to implement FMHA mainloop using cp.async instead of TMA
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "../collective/fmha_common.hpp"
#include "../collective/fmha_collective_load.hpp"
#include "../collective/fmha_collective_softmax.hpp"
#include "../kernel/fmha_options.hpp"

// 手动引用SM80的架构特性
#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/copy_traits_sm80.hpp"


template <typename T>
struct F;


namespace cutlass::fmha::collective {

  using namespace cute;
  using cutlass::fmha::kernel::Tag;
  using cutlass::fmha::kernel::find_option_t;

  //----------------------- 下面的部分用于计算不同shape的smem swizzle参数 -----------------------
  // 先设计一个模板，用于将任意列数N分解为一个奇数乘以2的整数次幂的形式，N = odd * p2
  template <unsigned int N>
  struct TrailingZeros {
    // N & (-N) 提取出最低位的 1，然后计算它是 2 的多少次方
    static constexpr unsigned int lowest_bit = N & (-N);
    static constexpr int value = TrailingZeros<lowest_bit / 2>::value + 1;
  };
  // 递归终止条件：当 N 为 1 (即 2^0) 时，尾部 0 的数量为 0
  template <>
  struct TrailingZeros<1> { static constexpr int value = 0; };

  template <unsigned int N> static constexpr unsigned int TrailingZeros_v = TrailingZeros<N>::value;

  // T是矩阵element类型
  // kM是swizzle的M值，反映的是单线程的访存位宽
  // N是矩阵的logic_col
  template <typename T, int SwizzleM, int N>
  struct SwizzleConfig {
    using Element = T;
    static constexpr unsigned int kM = SwizzleM;
    static constexpr unsigned int kSRAMWidth = 1024; // SRAM的位宽
    // 以下都是以2^kM为单位
    static constexpr unsigned int kSpan = (kSRAMWidth / (sizeof(Element) * 8)) >> kM;
    // 这里需要断言：N能被2^kM整除，这样能把它分成kGroup个组，每组包含2^kM个列
    static_assert(N % (1<<kM) == 0, "N must be divisible by 2^kM");
    static constexpr unsigned int kGroup = N >> kM;
    static constexpr unsigned int kLogicCol = 1 << (TrailingZeros_v<(kGroup)>);
    // 确定kB：分为kLogicCol < kSpan和kLogicCol >= kSpan两种情况
    //         kB取决于二者中的最小者
    static constexpr unsigned int kB = cute::log_2(kLogicCol < kSpan ? kLogicCol : kSpan);
    // 确定kS：如果kLogicCol < kSpan，那么让logic_row的比特域跟window_bank紧挨着就行
    //        反之，需要让logic_row的比特域跟logic_col紧挨着才行
    //        所以相当于取决于额kLogicCol和kSpan中的最大者，跟kB刚好相反
    static constexpr unsigned int kS = cute::log_2(kLogicCol < kSpan ? kSpan : kLogicCol);

    // 剩下的部分约定对SRAM的哪两种访存子阵可以避免bank conflict
    // 这里恢复到以element为单位
    // 先考虑flat的子阵
    // 如果kGroup >= kSpan，那么flat子阵应该是m=1,n=kSpan*(2^kM)
    // 反之，flat子阵应该是m=(kSpan/kLogicCol), n=kLogicCol*(2^kM)
    static constexpr unsigned int kFlatM = kGroup >= kSpan ? 1 : (kSpan / kLogicCol);
    static constexpr unsigned int kFlatN = kGroup >= kSpan ? (kSpan << kM) : (kLogicCol << kM);
    // 接下来考虑tiled的子阵
    // 其形状始终是m=kSpan, n=2^kM
    static constexpr unsigned int kTileM = kSpan;
    static constexpr unsigned int kTileN = (1 << kM);
  };
  // --------------------------------------------------------------------------------------------

  // 在SM80上，使用F16精度的warp level mma时，靠这个做c和a的layout变换
  // 这里的基本逻辑是：对于SM80的16bit mma.m16n8k16，从PTX ISA能看到
  // 每个A operand的val-layout恰好等于两个C operand val-layout的肩并肩
  // （前者是(2,2,2)，后者是(2,2)，因为k是n的2倍）
  // 所以为了把C换成A，我们必须要求：
  // i.  C的val-layout是A val-layout的size的整数倍，这是大前提
  // ii. 由于我们在M方向不迭代，所以只能从N方向的迭代里借一些C来拼凑A
  // 也就是说，把C的ITER_N维度拆分成两个子模(factor, ITER_N/factor):(stride, stride*factor)
  // 其中factor个C val肩并肩恰好能拼成一个A val，对于mma.m16n8k16而言，factor = 2
  // 然后把factor这个子模跟C原来的val-layout（即(2,2))重组成A的val-layout，
  // 剩下的ITER_N/factor留下来作为A的ITER_K就行了
  // 
  // 拓展：如果指令是mma.m16n8k8呢？
  // 此时，A和C的val-layout恰好一致，都是(2,2)，所以这种时候不需要特殊转换
  // 相当于factor = 1，可以据此来识别
  // 
  template <class CLayout, class AValueShape>
  CUTE_HOST_DEVICE constexpr auto
  convert_c_layout_to_a_layout_sm80(CLayout const& c, AValueShape const& a) {
    CUTE_STATIC_ASSERT_V(rank(c) == _3{}, "expect C = (MMA, MMA_M, MMA_N)");
    auto factor = size(a) / size<0>(c);
    CUTE_STATIC_ASSERT_V(size(a) == factor * size<0>(c), 
      "val-layout of A must be multiple of val-layout of C");
    CUTE_STATIC_ASSERT_V(size<2>(c) % factor == _0{}, 
      "MMA_N must be divisible by factor, allowing to borrow from MMA_N");
    CUTE_STATIC_ASSERT_V(rank(a) == rank<0>(c) + _1{}, 
      "val-layout of A have one additional rank than C," 
      "allowing to stack multiple C val to get A val");

    // 对于factor > 1的情况，需要从N借位到val上
    if constexpr (factor > _1{}) {
      return make_layout(
        make_shape(a, // 结果是A，所以val-layout肯定还是A的val-layout
          shape<1>(c),  // 只从ITER_N维度借，所以不触动ITER_M
          size<2>(c) / factor), // ITER_N被借走后剩下的作为A的ITER_K
        make_stride(append(stride<0>(c), stride<2>(c)), // 把C被借走的stride塞进val-layout
          stride<1>(c), // ITER_M不动
          stride<2>(c) * factor) // 这些是ITER_N被借走后剩下部分的间距/stride
      );
    }
    else {
      // 反过来，对于factor = 1的情况，A和C的val-layout相同，那么不需要做特殊转换
      return c;
    }
  }
  //template <class CLayout>
  //CUTE_HOST_DEVICE constexpr auto
  //  convert_c_layout_to_a_layout_sm80(CLayout const& c) {
  //  CUTE_STATIC_ASSERT_V(rank(c) == _3{});
  //  CUTE_STATIC_ASSERT_V(rank<2>(c) == _2{}, "must be Tile<_,2*AtomN,_>");
  //  //CUTE_STATIC_ASSERT_V(shape<2, 0>(c) == _2{}, "must be two n8 tile");
  //  return make_layout(
  //    make_layout(append(shape<0>(c), shape<2, 0>(c)),      // ((2,2),2) -> 8
  //      append(stride<0>(c), stride<2, 0>(c))),    // ((1,2),4)
  //    make_layout(shape<1>(c), stride<1>(c)),              // M_iter
  //    make_layout(shape<2, 1>(c), stride<2, 1>(c)));          // K_iter = bKV/16
  //}

  // 不使用TMA，也不使用GMMA
  // 可以认为这个mainloop是专门给SM80和SM89这种只有mma和cp.async的架构使用的
  // 目前只支持16bit输入输出，中间累加器精度是32bit
  template<
    typename Element_,
    typename ElementAccumulator_,
    typename TileShape_, // bQ, BlockKV, Dtile
    class Fusion,
    class... Options
  >
  struct FmhaMainloopCpAsync {

    using Element = Element_;
    using ElementAccumulator = ElementAccumulator_;
    using TileShape = TileShape_;
    using ClusterShape = Shape<_1, _1, _1>; // 3D，这个是上层要求的，不得不定义

    // 限制条件
    CUTE_STATIC_ASSERT(sizeof(Element) == 2, "Element must be 16bit");
    CUTE_STATIC_ASSERT(sizeof(ElementAccumulator) == 4, "ElementAccumulator must be 32bit");

    // Options
    // 很奇怪，我没看到Stages是在哪里设置或者使用默认值的
    static constexpr int StageCount = find_option_t<Tag::kStagesKV, Int<4>, Options...>::value;
    static constexpr int StageCountQ = find_option_t<Tag::kStagesQ, Int<1>, Options...>::value;

    // StagesQ和Stages看来都是typename，那么应该是Int<N>
    using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
    using Stages = cutlass::gemm::collective::StageCount<StageCount>;

    // 16B alignment
    // 不需要使用TMA，但是如果想使用128bit的向量访存，alignment的要求仍然需要保持16B
    static constexpr int Alignment = 16 / sizeof(Element);

    // TileShapeXX都是IntTuple类型（rank=3）
    using TileShapeQK = TileShape;  // (bQ, BlockKV, Dtile)
    using TileShapePV = decltype(select<0, 2, 1>(TileShapeQK{})); // (bQ, Dtile, BlockKV)

    // 这里名为LayoutQKV，实际上只是输入tensor的stride，这个命名很有误导性
    using LayoutQKV = cute::tuple<int, _1, cute::tuple<int, int>>;
    using LayoutQ = LayoutQKV;
    using LayoutK = LayoutQKV;
    using LayoutV = LayoutQKV;

    // 原来的版本直接使用cutlass的collective builder寻找合适的mainloop组件
    /*using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, 
      cutlass::arch::OpClassTensorOp,
      Element, LayoutQ, Alignment,
      Element, LayoutK, Alignment,
      ElementAccumulator,
      TileShapeQK, ClusterShape, Stages,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

    using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder <
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      // the stride for A does not matter since we do not load from smem at all
      Element, LayoutK, Alignment,
      Element, decltype(select<1, 0, 2>(LayoutV{})), Alignment,
      ElementAccumulator,
      TileShapePV, ClusterShape, Stages,
      cutlass::gemm::KernelTmaWarpSpecialized > ::CollectiveOp;*/
    // 而CUTLASS 3.x没有为SM90更早的版本实现collective builder，此路不通
    // 我们只能直接在这一层实现所需的各个零部件，包括：
    // 1.tiled mma
    // 2.smem layout，以及相应的shared storage
    // 3.pipeline and pipeline state
    // 4.load功能
    // 5.gemm功能
    // 
    // TODO: 感觉继承自SM90的流水线设计不是特别理想，考虑后面再优化
    // 流水线深度与tile尺寸的硬性约束（改动任何一条都要重新验算另外两条）：
    //  (1) prologue填满StageCount个slot，peel轮消掉2个 ⇒ 主循环稳态in-flight恒为StageCount-2，
    //      故两处wait常数都是 cp_async_wait<StageCount-3>。StageCount==3时退化成wait_all（无预取），
    //      更小则直接是负数模板实参。
    // （1-1）StageCount必须是偶数，因为目前的mainloop按照K在偶数slot，V在奇数slot的固定逻辑load
    //        如果slot数是奇数，那么每迭代一次，K和V的slot奇偶性就要互换一次，跟mainloop目前的实现冲突
    //  (2) TiledMma的PermutationMNK在N上取了2*AtomN=16（Tile<_,_16,_>），
    //      目的是让B也能用一条ldmatrix.x4（而不是两条.x2）
    //      所以这要求BlockKV是16的倍数。
    // （2-1）而且我们在PV阶段使用的是相同的tiled mma，而PV阶段的N维度是Dtile
    //        这就要求Dtile也需要是16的倍数
    //  (3) K方向（reduction）按AtomK=16迭代，故bD也必须是16的倍数，跟(2-1)的要求一致
    static_assert(StageCount >= 4,
      "稳态wait常数是cp_async_wait<StageCount-3>：StageCount<4会退化或产生负数模板实参");
    static_assert(StageCount % 2 == 0,
      "StageCount必须是偶数，因为mainloop期望K在偶数slot，V在奇数slot");
    static_assert(get<1>(TileShapeQK{}) % 16 == 0,
      "Tile<_,_16,_>要求BlockKV是16的倍数（一个permute块=2个n8 atom-tile）");
    static_assert(get<2>(TileShapeQK{}) % 16 == 0,
      "K方向按AtomK=16迭代，Dtile必须是16的倍数");
    // 
    // A. TiledMma for QK and PV stage
    //    已知CTA tile shape为TileShapeQK和TileShapePV
    //    需要决定的参数包括：
    //    a. threads per CTA，在SM80上，选择128的倍数即可
    //    b. mma inst，对于F16输入，最好选用mma.m16n8k16
    //    原本的选择是由collective builder来做的，现在只能人为来做，泛化性不好，
    //    同时不保证对所有shape和type都合法，但是目前还在学习摸索阶段，只能这样做了
  private:
    // 注意：SM80上的mma本质上都是TN格式的，所以后缀永远是TN
    using MmaOperation_t = SM80_16x8x16_F32F16F16F32_TN;
    using MmaAtom_t = MMA_Atom<MmaOperation_t>;
    // QK阶段结束后，为了执行softmax in the same warp，要求每个warp单独负责
    // 所有N的计算，相当于所有warp沿着M轴依次展开，所以必需确保warp num = M / m16
    // 并且必须是整除，这点是与SM90原版一致的，算是FMHA的经典设计考虑了
    // 同时需要注意，每个Q tile和K tile的reduction维度大小都必须等于D，这样才符合
    // QK阶段只加载一次Q tile的假设（实际上FA2的QK迭代是在不同K tile之间迭代）
    // 不过D不是编译期信息，只能由上层的caller来检验了
    using MmaQKAtomLayout_t = decltype(make_layout(
      make_shape(get<0>(TileShapeQK{}) / get<0>(MmaAtom_t::Shape_MNK{}), //所需warp数
                 _1{}, _1{}),
      LayoutLeft{}));
    // 然后决定perm（可以理解为tiled mma的val-layout）
    // 理论上如果没有什么特殊需求，使用默认的参数<_,_,_>即可，此时的tiled mma
    // 的shape就等于atom * atom-layout
    // 不过这种情况下tiled mma在make fragment A/B/C时也只会分配一个atom的大小
    // 在mma.m16n8k16的场景下，相当于A:m16k16, B:n8k16和C:m16n16
    // 此时A可以通过ldmatrix.x4来load，而B只能靠ldmatrix.x2来load
    // 为了进一步减少ldmatrix指令发射的次数，需要在N方向扩大一些
  public:
    // TODO: 有个疑问，给N维度Perm了之后，mma fragment的N维度ITER_N会怎么变？
    // 注意：_ 是一个object，它的类型是cute::Underscore
    //      这跟Int<N>恰好反过来了，_N是类型Int<N>的alias，而_N{}才是object
    //      所以很容易混淆
    using TiledMmaQK = TiledMMA<MmaAtom_t, MmaQKAtomLayout_t,
                                Tile<Underscore, _16, Underscore>>; // m64n16k16
    
    // 同理，构造PV阶段的tiled mma
    // 这里使用跟QK阶段完全相同的tiled mma即可
    using TiledMmaPV = TiledMmaQK;

    // B. 计算smem的layout
  private:
    // 按照上面的讨论，根据具体的shape来决定smem layout
    // 这里用一个函数把计算逻辑封装起来
    // 这里N是每个smem tensor的宽度
    template <int N>
    CUTE_HOST_DEVICE 
    static constexpr auto make_k_major_smem_atom(Int<N>) {
#if 0
      // 这种方式虽然能解决bank conflict，但会造成L1/TEX <-> LTS的访存非合并，尽量不要采用
      constexpr int n8 = 8;
      CUTE_STATIC_ASSERT(N >= n8 and N % n8 == 0, "N的尺寸不适配ldmatrix.m8n8指令");
      // 由于我们使用ldmatrix.m8n8读SRAM，所以最低限度要保证kVec个连续n坐标
      // 在映射之后也保持连续
      constexpr int kVec = 16 / sizeof(uint16_t);
      // 把N分解为若干个肩并肩的m8n8子阵
      constexpr int kSpan = N / n8;
      // 确定写SRAM阶段的一个wave（1024bit=128B）能够覆盖多少个kSpan
      // 注意：如果N太大，kSpan也会很大，有可能一个wave盖不住一个kSpan
      // 我们不要让kFactor小于1即可（cute::min/max期望输入是uint32的，否则会报warning）
      constexpr int kFactor = 
        cute::max(static_cast<uint32_t>(((128 / sizeof(uint16_t)) / n8) / kSpan), 1u);
      // 根据kFactor的大小来确定kM
      constexpr int kM = log_2(cute::max(static_cast<uint32_t>(kFactor * n8), 
                                         static_cast<uint32_t>(kVec)));
      // kB本质上就是每个m8n8子阵（含8*8=64个元素）里包含多少个kM
      constexpr int kB = log_2(8u * 8u) - kM;
      // kS表示1024bit=128B的SRAM位宽能覆盖多少个kM
      constexpr int kS = log_2(static_cast<uint32_t>(128 / sizeof(uint16_t))) - kM;
      // over
      return composition(
        Swizzle<kB, kM, kS>{},
        Layout<Shape<_8, Shape<Int<n8>, Int<kSpan>>>,
        Stride< Int<n8>, Stride<    _1, Int<8* n8>>>
        >{}
      );
#else
      // 这是最naive又常用的方式，而且对LTS比较友好
      constexpr int kM = cute::log_2(16 / sizeof(Element));
      using config = SwizzleConfig<Element, kM, N>;
      // over
      return composition(Swizzle<config::kB, config::kM, config::kS>{},
        Layout <Shape<Int<config::kTileM>, Int<N>>,
                Stride<            Int<N>,     _1>
        >{}
      );
#endif
    }

  public:
    // 然后用这个函数计算Q、K和V的smem atom
    // CTA的smem shape应该都是rank=3，其中最后一个维度作为stage
    using SmemLayoutQ = decltype(tile_to_shape(
      make_k_major_smem_atom(get<2>(TileShapeQK{})),
      append(select<0, 2>(TileShapeQK{}), Int<StagesQ::value>{})));
    // 这里原本想用insert<1>，但是似乎insert<N>表示插入到rank=N的位置，
    // 而不是插入到rank=N之后，这跟STL中insert的语义不同，容易造成误解
    // 所以干脆换成append

    using SmemLayoutK = decltype(tile_to_shape(
      make_k_major_smem_atom(get<2>(TileShapeQK{})),
      append(select<1, 2>(TileShapeQK{}), Int<Stages::value>{})));
    
    // 注意V是PV阶段mma的operand B
    // 由于V也是D-major存储，但PV阶段的D维度实际上是mma的N维度
    // 所以相当于V是以N-major（row-major）存储的
    // 但是mma要求的operand layout必须是TN，即B必须是K-major
    // 这就需要在G2S之后，利用ldmatrix.trans将SRAM转置
    // 不过，由于ldmatrix仍然是以m8n8的子阵形式读取数据（一个方阵）
    // 所以swizzle设计跟QK阶段没什么区别（嗯，严格来说跟K的layout一致）
    using SmemLayoutV = SmemLayoutK;
    // 这里的SmemLayoutV仍然是跟V-cache的原始layout一样是KV在前，D在后
    // 但是在PV mma中，作为operand B，必须遵守N（即D）在前，K（即KV）在后的convention
    // 所以还需要交换SmemLayoutV的rank-1和rank-2的位置，给mma阶段提供一个符合convention
    // 的tensor view，这里只需要调换两个维度在tuple里的位置，不改变layout映射
    // TODO：我们原本想要用select直接调换SmemLayoutV的rank位置，但是agent提示
    //      SmemLayoutV是一个ComposedLayout而非Layout，不支持select
    //      建议的做法如下，我们测试了一下，确实是可行的，但是目前不太理解其原因
    using SmemLayoutSOSV = decltype(composition(
      SmemLayoutV{}.layout_a(),                       // Swizzle<kB,kM,3>
      SmemLayoutV{}.offset(),                         // Int<0>
      select<1, 0, 2>(SmemLayoutV{}.layout_b())));    // 这里是 plain Layout，select 能用

    // C. 设计pipeline和pipeline state
    //    SM90和SM100的pipeline都是cutlass/pipeline中已经实现的组件
    //    更早的架构原则上不需要使用pipeline，也没有现成的实现
    //    不过，FMHA的主要kernel代码都是在当前的collective API实现的
    //    向上暴露给kernel API的内容中虽然确实包含pipeline相关的部分
    //    但是检查之后发现，这些主要是初始化操作
    //    更上层（builder、device API）不会暴露pipeline的存在
    //    因此，我们可以：
    //    a. 实现collective API时，不再使用SM90风格的pipeline
    //    b. 修改kernel API中所有与pipeline相关的部分
    //    对于SM80，我们不再使用mbarrier来显式管理ring buffer中每个slot的
    //    读和写，因为SM80的cp.async功能非常简单，而且不需要借助mbarrier工作
    //    但是为了维护read和write之间（消费者-生产者之间）的相位差，确保
    //    有序性，我们仍然需要维护原本的pipeline state，确保消费者和生产者
    //    始终按照state给定的slot编号进行消费和生产（唯一的不同是：由于不使用
    //    mbarrier，state里现在只需要维护index，不需要phase）
    //    使用中，只要确保正确更新这两个state即可
    //    同时，由于cp.async是thread粒度，必需用__syncthreads()将其扩展到
    //    CTA-scope的粒度，同时确保任何thread不会提前越过消费者-生产者之间的
    //    相界面，其他应该就没什么特别需要注意的了
    //    

    // 我们可以偷懒，直接挪用SM90的PipelineState类型，因为它符合我们的需求
    // 只是多了一个phase的维护工作，后面如果对性能敏感，可以再实现无phase的
    // 注：cutlass::PipelineState这个模板类就是SM90定义的
    using PipelineState = typename cutlass::PipelineState<Stages::value>;
    using PipelineStateQ = typename cutlass::PipelineState<StagesQ::value>;

    // 这几个类型是给上层用的吗？
    using TileShapeOut = TileShapePV;
    using TiledMmaOut = TiledMmaPV;
    using ElementOut = ElementAccumulator;

    struct SharedStorage {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
      union {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
      };
    };

    // Arguments算是对device API暴露的接口约定：要求host端必须传入这些信息
    // 外面的kernel API可能还会在这个Arguments之外再索取一些其他信息
    // 但那就不关collective API的事了
    struct Arguments {
      const Element* ptr_Q;
      LayoutQ dQ; // 这名字太有误导性了
      const Element* ptr_K;
      LayoutK dK;
      const Element* ptr_V;
      LayoutV dV;
    };

    // D. 接下来，我们需要将load功能从TMA替换成cp.async的TiledMma
    //    SM90版本GMMA的Q、K、V都是以smem作为操作数，因此不需要S2R
    //    SM80版本则必须同时考虑G2S和S2R
    //    这样，SM80的load使用方式会跟SM90有很大不同，我们考虑不再
    //    从形式上模仿后者
    //    从compute()的实现来看，SM90将load行为封装在LoadQ、LoadK
    //    等type中，然后直接执行对应的step来加载数据到指定的
    //    ring-buffer slot中
    //    因此我们只需以自己的方式封装这些操作即可
    //
    // Q/K/V 的 g2s copy 都用它。
    // 借鉴了CUTLASS已有的类似功能：
    // make_simt_gmem_tiled_copy（gemm/collective/builders/sm90_common.inl:127-151）
    //   GmemCopyAtom : 决定每线程一次搬多少（kVec = NumValSrc）
    //   kThreads     : CTA 线程数
    //   (kX, kK)     : 分别是tile 的「非连续维」和「连续维」大小
    template <class GmemCopyAtom, int kThreads, int kX, int kK>
    CUTE_HOST_DEVICE static constexpr auto
      make_k_major_g2s_copy(Int<kX>, Int<kK>) {
      constexpr int kVec = GmemCopyAtom::NumValSrc;         // ← 由 atom 决定
      constexpr int kThrK = cute::min(kThreads, kK / kVec); // 优先沿连续维铺满（合并访存）
      constexpr int kThrX = kThreads / kThrK;

      static_assert(kK % kVec == 0,
        "连续维必须是整数个 copy-atom 向量（F16 + 128-bit 时是 8 的倍数）");
      static_assert(kThreads % kThrK == 0,
        "CTA 线程数必须能被连续维方向的线程数整除");
      static_assert(kK % (kThrK * kVec) == 0,
        "连续维必须能被一次 TiledCopy 的宽度整除（仅当线程数不足以铺满一行时才可能触发）");
      static_assert(kX % kThrX == 0,
        "非连续维必须能被该方向的线程数整除，否则 partition 会在 evenly_divides 上失败");

      return make_tiled_copy(
        GmemCopyAtom{},
        Layout<Shape <Int<kThrX>, Int<kThrK>>,
        Stride<Int<kThrK>, _1>>{},   // K-major 线程序 → 相邻 tid 走相邻 gmem
        Layout<Shape<_1, Int<kVec>>>{});           // 每线程一个 128-bit 向量
    }
    
    // 以下都是QKV共用的
    // 这里cp.async使用ZFILL版本，是为了把OOB的部分清零，这样后续ldmatrix就不需要再考虑OOB
    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
    // using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, Element>;
    static constexpr int MaxThreadsPerBlock = thr_size(TiledMmaQK{});
    static_assert(MaxThreadsPerBlock == thr_size(TiledMmaPV{}),
      "Q/K/V 的 g2s 共用同一个线程数");
    using SmemCopyAtomQK = Copy_Atom<SM75_U32x4_LDSM_N, Element>; // ldmatrix.x4.m8n8
    // 这里说明一下：ldmatrix和ldmatrix.trans的区别在于，前者要求输入矩阵是n-major存储的
    // 而后者则要求输入是m-major存储的，除此之外，二者最后的结果（每个thread上寄存器内的元素
    // 在原矩阵中对应的(m,n)坐标）都是一致的
    using SmemCopyAtomV = Copy_Atom<SM75_U16x8_LDSM_T, Element>; // ldmatrix.x4.trans.m8n8

    // 下面LoadQ/LoadK/LoadV的结构有很大一部分都是重复的，理论上应该可以概括为同一个模板
    struct LoadQ {
      // 先构造tiled copy
      using TiledCopyQ = decltype(make_k_major_g2s_copy<GmemCopyAtom, MaxThreadsPerBlock>(
        get<0>(TileShapeQK{}), get<2>(TileShapeQK{})));

      // 初始化，预构建所有tensor和predicate
      template <typename ProblemShape, typename BlkCoord,
                typename QTensor>
      CUTE_DEVICE
      static auto 
      init(Element const* ptr_Q, 
           LayoutQ const& stride_q, // 注意这个是Q的stride！
           QTensor const& sQ,
           ProblemShape const& problem_size, // [B,H,Q,K,D]
           BlkCoord const& blk_coord) {
        // 运行时的实际Q tensor
        Tensor mQ = make_tensor(make_gmem_ptr(ptr_Q),
          make_shape(get<2>(problem_size), // Q
                     get<4>(problem_size), // D
                     select<0, 1>(problem_size)), // (B,H)
          stride_q);
        // ---- 切出本 CTA 的 (bQ, Dtile) tile ----
        // rank-2 tiler 打在 rank-3 tensor 上，第 3 模透传；
        // coord 第 3 位给具体 (b,h) → 该模被消掉。
        // local_tile 结果是 (bQ, Dtile, RestD)；can_implement 保证 D <= Dtile
        // ⇒ RestD 恒为 1，直接切掉，后面全是 rank-2，干净很多。
        Tensor gQ = local_tile(mQ,
          select<0, 2>(TileShapeQK{}),  // tiler = <bQ, bD>
          make_coord(get<0>(blk_coord), _0{}, get<2>(blk_coord)));// (bQ, bK)，只保留2维

        // ---- 按线程 partition —— 只做一次 ----
        auto thr_copy = TiledCopyQ{}.get_slice(threadIdx.x);
        Tensor tQgQ = thr_copy.partition_S(gQ);  // (VAL, ITER_M, ITER_K)
        Tensor tQsQ = thr_copy.partition_D(sQ(_, _, _0{}));  // (VAL, ITER_M, ITER_K)

        // ---- predicate ----
        // VAL 模是「1 行 × kVec 个连续 K」，且 can_implement 保证 D % kVec == 0，
        // 所以一个 VAL 组要么整组在界内、要么整组在界外 ⇒ 谓词只需 (ITER_M, ITER_K) 粒度。
        // mode-0 给 _1{}：rank 与 src 对齐，且等价于沿 VAL 广播。
        // TODO: 一直很好奇，coord tensor占用什么存储空间？真的会分配对应的存储资源吗？
        //       另外，对于2D的情况，应该可以直接使用两条一维的coord tensor来节约资源吗？
        Tensor cQ = make_identity_tensor(select<0, 2>(TileShapeQK{}));  // (bQ,bD) -> (q,d)
        Tensor tQcQ = thr_copy.partition_S(cQ);                         // (VAL, ITER_M, ITER_K)
        // 一个VAL-layout要么都越界，要么都不越界，因为tensor都是对齐到cp.async的边界
        // 所以没必要给VAL-layout里每个element都分配一个predicate
        Tensor tQpQ = make_tensor<bool>(make_shape(_1{}, size<1>(tQcQ), size<2>(tQcQ)));
        // q_offset是当前CTA tile在Q维度上的起始位置
        int q_offset = get<0>(blk_coord) * get<0>(TileShapeQK{});
        CUTE_UNROLL
          for (int m = 0; m < size<1>(tQpQ); ++m) {
            CUTE_UNROLL
              for (int k = 0; k < size<2>(tQpQ); ++k) {
                // 这个双重循环其实就是在遍历所有VAL
                auto c = tQcQ(_0{}, m, k); // 该VAL组首元素的 (q,d)
                tQpQ(_0{}, m, k) =
                  (get<0>(c) + q_offset < get<2>(problem_size)) and // seqlen_q 边界
                  (get<1>(c) < get<4>(problem_size));          // D 边界（允许D < bD）
              }
          }
        // over，返回一个tuple
        return make_tuple(tQgQ, tQsQ, tQpQ);
      }

      // Q 只载一次：不要 iterator / pipeline state / tile_count
      template <typename State> // State就是init的返回类型
      CUTE_DEVICE static void
      load(State const& state /*直接拆包state获得所需的tensor*/) {
        Tensor dst = get<1>(state);              // 视图拷贝，零成本（指针+静态 layout）
        copy_if(TiledCopyQ{}, get<2>(state), get<0>(state), dst);
        // 不 cp_async_fence()：由 mainloop 显式提交
        return;
      }
    };

    // 同理，实现K的Load操作
    struct LoadK {
      // (BlockKV, Dtile)：注意是 select<1,2>，Q 那边是 select<0,2>
      using TiledCopy = decltype(make_k_major_g2s_copy<GmemCopyAtom, MaxThreadsPerBlock>(
        get<1>(TileShapeQK{}), get<2>(TileShapeQK{})));

      // 返回 (gmem 源, smem 目的, 坐标张量)
      // 谓词不放进 state —— 它每轮都要重算
      template <class ProblemShape, class BlkCoord,
                typename KTensor>
      CUTE_DEVICE static auto
      init(Element const* ptr_K, 
           LayoutK const& stride_k,
           KTensor const& sK,
           ProblemShape const& problem_size, // [B,H,Q,K,D]
           BlkCoord const& blk_coord) {
        // ---- 1. gmem 全局视图：(seqlen_k, D, (B,H)) ----
        //         注意 seqlen_k 是 get<3>，Q 那边是 get<2>
        Tensor mK = make_tensor(make_gmem_ptr(ptr_K),
          make_shape(get<3>(problem_size), // K
            get<4>(problem_size),          // D
            select<0, 1>(problem_size)),   // (B,H)
          stride_k);

        // ---- 2. 切 tile ----
        // KV 方向必须留 `_`（要沿 KV tile 迭代）；D 方向 RestD==1，切掉
        // local_tile → (BlockKV, Dtile, RestK, RestD)  →  (BlockKV, Dtile, RestK)
        Tensor gK = local_tile(mK,
          select<1, 2>(TileShapeQK{}),
          make_coord(_, _0{}, get<2>(blk_coord))); // D方向不做迭代，不保留这个维度

        // ---- 4. partition ----
        auto thr_copy = TiledCopy{}.get_slice(threadIdx.x);
        Tensor tKgK = thr_copy.partition_S(gK);   // (VAL, ITER_N, ITER_K, RestK)
        Tensor tKsK = thr_copy.partition_D(sK);   // (VAL, ITER_N, ITER_K, Stages)

        // ---- 5. 坐标张量：tile *局部* 坐标，对每个 KV tile 都一样，只算一次 ----
        Tensor cK = make_identity_tensor(select<1, 2>(TileShapeQK{}));  // (BlockKV,Dtile) -> (n,d)
        Tensor tKcK = thr_copy.partition_S(cK);                         // (VAL, ITER_N, ITER_K)
        // over
        return make_tuple(tKgK, tKsK, tKcK);
      }

      // 沿用你原来的接口形状：iterator / pipeline state / tile_count 都自动推进
      template <bool kAdvanceIterator = true,
                bool kAdvancePipe = true,
                typename State, typename ProblemShape, typename PipelineState_t>
      CUTE_DEVICE static void
      step(State       const& state,
          ProblemShape const& problem_size,
          int& k_tile,        // 当前 KV tile 序号
          PipelineState_t& smem_write,    // 目的 slot
          int& tile_count)    // 剩余有效 tile 数
      {
        auto const& tKcK = get<2>(state); // (VAL, ITER_N, ITER_K)
        Tensor src = get<0>(state); // (VAL,ITER_N,ITER_K,RestK)
        Tensor dst = get<1>(state); // (VAL,ITER_N,ITER_K,stage)

        // ---- 谓词：局部寄存器数组，shape 全静态，编译期完全展开 ----
        Tensor pred = make_tensor<bool>(make_shape(_1{}, size<1>(tKcK), size<2>(tKcK)));
        // 我们得同时提防K方向和D方向的越界问题
        int n_offset = k_tile * get<1>(TileShapeQK{});  // CTA tile在K方向的起点
        CUTE_UNROLL
          for (int n = 0; n < size<1>(pred); ++n) {
            CUTE_UNROLL
              for (int d = 0; d < size<2>(pred); ++d) {
                // 同样，由于tensor最起码可以确保是与copy VAL对齐，所以按VAL粒度检查
                auto c = tKcK(_0{}, n, d);
                pred(_0{}, n, d) = (tile_count > 0) and
                    (get<0>(c) + n_offset < get<3>(problem_size)) and // K边界
                    (get<1>(c) < get<4>(problem_size));    // D边界
              }
          }

        copy_if(TiledCopy{}, pred, 
                             src(_, _, _, k_tile), 
                             dst(_, _, _, smem_write.index()));
        // 不 cp_async_fence()：由 mainloop 显式提交

        // 只在仍有效时推进 tile 序号 —— 否则源地址会越出 RestK 范围
        if constexpr (kAdvanceIterator) { if (tile_count > 0) ++k_tile; }
        if constexpr (kAdvancePipe) { ++smem_write; }
        --tile_count;
        // over
        return;
      }
    };

    // 最后是V的Load
    struct LoadV {
    public:
      // 构造tiled copy
      // V也是D-major的，但D是PV-mma的N维度，这是不同之处
      // 这会影响S2R的操作，但不影响G2S操作
      // 注意，非连续维是K，连续维是D
      // 
      using TiledCopy = decltype(make_k_major_g2s_copy<GmemCopyAtom, MaxThreadsPerBlock>(
        get<2>(TileShapePV{}), get<1>(TileShapePV{})));

      template <class ProblemShape, class BlkCoord,
                typename VTensor>
      CUTE_DEVICE static auto
      init(Element const* ptr_V,
          LayoutV const& stride_v,
          VTensor const& sV,
          ProblemShape const& problem_size, // [B,H,Q,K,D]
          BlkCoord const& blk_coord) {
        // global tensor view
        Tensor mV = make_tensor(make_gmem_ptr(ptr_V),
          make_shape(get<3>(problem_size),  // K
            get<4>(problem_size),           // D
            select<0, 1>(problem_size)),    // (B,H)
          stride_v);

        // local tile
        // V的blk coord索引规则基本上跟K是一致的
        // 不过PV阶段的CTA tile shape变成了(bQ, bD, bK)，跟QK阶段不同
        // (BlockKV, Dtile, RestK)
        Tensor gV = local_tile(mV,
          select<2, 1>(TileShapePV{}),  // -> (BlockKV, Dtile)
          make_coord(_, _0{}, get<2>(blk_coord))); // D方向不迭代，还是消掉它

        // partition to each thread
        // 这里就能看到：V因为原始shape就是K在前N在后，它的partition的K和N维度位置是反的
        // 这在G2S copy阶段倒是没什么问题，但是一定要注意，在MMA和S2R阶段这会带来很多麻烦
        auto thr_copy = TiledCopy{}.get_slice(threadIdx.x);
        Tensor tVgV = thr_copy.partition_S(gV); // (VAL,ITER_K,ITER_N,RestK)
        Tensor tVsV = thr_copy.partition_D(sV); // (VAL,ITER_K,ITER_N,stage)

        // 构造coordinate tensor
        Tensor cV = make_identity_tensor(select<2, 1>(TileShapePV{})); // (BlockKV,Dtile)
        Tensor tVcV = thr_copy.partition_S(cV);  // (VAL, ITER_K, ITER_N)也是反的...
        // over
        return make_tuple(tVgV, tVsV, tVcV);
      }

      template <bool kAdvanceIterator = true,
                bool kAdvancePipe = true,
                typename State, typename ProblemShape, typename PipelineState_t>
      CUTE_DEVICE static void
      step(State       const& state,
          ProblemShape const& problem_size, // [B,H,Q,K,D]
          int& v_tile,        // 当前 KV tile 序号
          PipelineState_t& smem_write,    // 目的 slot
          int& tile_count)    // 剩余有效 tile 数
      {
        auto const& tVcV = get<2>(state); // (VAL, ITER_K,ITER_N)
        Tensor src = get<0>(state); // (VAL,ITER_K,ITER_N,RestK)
        Tensor dst = get<1>(state); // (VAL,ITER_K,ITER_N,stage)

        // ---- 谓词：局部寄存器数组，shape 全静态，编译期完全展开 ----
        // (1, ITER_K,ITER_N)
        Tensor pred = make_tensor<bool>(make_shape(_1{}, size<1>(tVcV), size<2>(tVcV)));
        // 我们得同时提防K方向和D方向的越界问题
        int k_offset = v_tile * get<2>(TileShapePV{});  // CTA tile在K方向的起点
                                                        // 因为该方向比D方向更重要...
        CUTE_UNROLL
        for (int n = 0; n < size<2>(pred); ++n) { // ITER_N
          CUTE_UNROLL
          for (int k = 0; k < size<1>(pred); ++k) { // ITER_K
            // 还是按VAL粒度检查
            auto c = tVcV(_0{}, k, n); // (k,n)
            pred(_0{}, k, n) = (tile_count > 0) and
              ((get<0>(c) + k_offset) < (get<3>(problem_size))) and // K边界
              ((get<1>(c)) < (get<4>(problem_size)));               // D边界
          }
        }

        copy_if(TiledCopy{}, pred,
                             src(_, _, _, v_tile),
                             dst(_, _, _, smem_write.index()));
        // 不 cp_async_fence()：由 mainloop 显式提交

        // 只在仍有效时推进 tile 序号 —— 否则源地址会越出 RestK 范围
        if constexpr (kAdvanceIterator) { if (tile_count > 0) ++v_tile; }
        if constexpr (kAdvancePipe) { ++smem_write; }
        --tile_count;
        // over
        return;
      }
    };


  public:
    // Params不对device API公开，实际上基本也不对kernel API公开
    // 我们只需要提供转换接口，将公开的Arguments转换成Params就行
    // kernel API会自动替我们做转换，然后把转换结果params传进来
    // 所以我们需要根据自己的需求定义Params
    struct Params {
      /*TMA_Q tma_load_q;
      TMA_K tma_load_k;
      TMA_V tma_load_v;*/
      // 嗯，主要需要的就是初始化G2S LoadX的信息
      // 简单起见，我们直接照抄Arguments就行，后面可以再优化
      const Element* ptr_Q;
      LayoutQ stride_q; 
      const Element* ptr_K;
      LayoutK stride_k;
      const Element* ptr_V;
      LayoutV stride_v;

      // 这些SDPA（及其变种）的scale都是自动从problem shape计算来的
      float scale_softmax;
      float scale_softmax_log2;
      float rp_dropout;
    };

    // 看起来can_implement的逻辑不需要改动
    template<class ProblemShape>
    static bool can_implement(ProblemShape const& problem_size, Arguments const& args) {
      return true
        && (get<4>(problem_size) <= get<2>(TileShape{}))
        && ((get<4>(problem_size) % Alignment) == 0)
        && ((get<2>(problem_size) % Alignment) == 0)
        ;
    }
    // 从Arguments中萃取Params各field的信息
    template<class ProblemShape>
    static Params to_underlying_arguments(ProblemShape const& problem_size, 
                                          Arguments const& args, 
                                          void* workspace) {

      // 这些不需要了
      // auto problem_shape_qk = make_shape(get<2>(problem_size), 
      //                                    get<3>(problem_size), 
      //                                    get<4>(problem_size), 
      //                         make_shape(get<0>(problem_size), get<1>(problem_size)));
      //auto params_qk = CollectiveMmaQK::to_underlying_arguments(problem_shape_qk,
      //  typename CollectiveMmaQK::Arguments{
      //      args.ptr_Q, args.dQ,
      //      args.ptr_K, args.dK,
      //  }, /*workspace=*/ nullptr);
      //auto problem_shape_pv = select<0, 2, 1, 3>(problem_shape_qk);
      //auto params_pv = CollectiveMmaPV::to_underlying_arguments(problem_shape_pv,
      //  typename CollectiveMmaPV::Arguments{
      //      args.ptr_K, args.dK,  // never used, dummy
      //      args.ptr_V, select<1,0,2>(args.dV),
      //  }, /*workspace=*/ nullptr);

      return Params{
        /*params_qk.tma_load_a,
        params_qk.tma_load_b,
        params_pv.tma_load_b,*/
        // 如上所述，直接照抄Arguments的内容
        args.ptr_Q,
        args.dQ,
        args.ptr_K,
        args.dK,
        args.ptr_V,
        args.dV,
        // 下面计算SDPA的scale信息
        1.0f / (float)std::sqrt(get<4>(problem_size)),
        (float)(std::log2(std::exp(1.0)) / std::sqrt(get<4>(problem_size))),
        1.0f };
    }

    template<class BlkCoord, class ProblemShape>
    CUTLASS_DEVICE auto
      compute(
        BlkCoord const& blk_coord,  // 这个由kernel API从scheduler那里取号获得
        Params const& params,       // Params由我们定义，由kernel API初始化传入
        ProblemShape const& problem_size, // 也是由kernel API负责提供
        PipelineState& smem_pipe_read, // PipelineState类型由我们定义，暴露给kernel API，让它初始化传入
        PipelineState& smem_pipe_write,
        [[maybe_unused]] PipelineStateQ& smem_pipe_read_q, 
        [[maybe_unused]] PipelineStateQ& smem_pipe_write_q,
        SharedStorage& storage) // smem storage也是我们自己定义，由device API在launch时分配
    {
      int thread_idx = threadIdx.x;
      int fusion_tile_count = Fusion{}.get_trip_count(blk_coord, TileShape{}, problem_size);

      // 对smem storage进行塑形
      Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{}); // (bM,bK,1)
      Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{}); // (bN,bK,stages)
      Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{}); // (bK,bN,stages)
      // G2S的时候，不需要关心V的rank次序，但是S2R和mma的时候，我们必须遵守以下规则：
      // 对于operand B，rank-0必须是N，rank-1是K
      // 而sV现在rank-0是K，rank-1是N，正好反了
      // 我们之前已经做过rank交换了，得到SmemLayoutSOSV这个layout，就是给PV mma用的
      // 所以只需要照此重新塑形即可
      Tensor sOsV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutSOSV{});

      // 初始化G2S LoadX
      LoadQ load_q{};
      auto load_state_q = load_q.init(params.ptr_Q, params.stride_q,
                                      sQ,
                                      problem_size, 
                                      blk_coord);

      LoadK load_k{};
      auto load_state_k = load_k.init(params.ptr_K, params.stride_k,
                                      sK,
                                      problem_size,
                                      blk_coord);

      LoadV load_v{};
      auto load_state_v = load_v.init(params.ptr_V, params.stride_v,
                                      sV,
                                      problem_size,
                                      blk_coord);


      // Issue cp.async Loads (Prologue fetches)
      {
        // 注意，我们的LoadQ只能load一次，不能step
        load_q.load(load_state_q);

        // commit
        cp_async_fence();
      }

      // Loop over K elems
      // 我们就用简单的int代替iterator吧
      //auto k_tile_iter = cute::make_coord_iterator(fusion_tile_count);
      int k_tile_iter = 0;  // [0, fusion_tile_count)

      // 改个名
      // k_tile_count_simt实际上就是所有需要load的tile数量
      int k_tile_count_simt = 2 * fusion_tile_count;

      // 开始prologue，这里只是预填充了各条LoadX的流水线
      // 很有意思，跟GEMM不同，这里一次性把所有stage都填满了
      {
        CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < StageCount; i++) {
            // 注意，我们的smem_pipe_write是被step()自动更新的
            if (i % 2 == 0) {
              load_k.template step<false>(load_state_k, 
                                          problem_size,
                                          k_tile_iter, 
                                          smem_pipe_write, 
                                          k_tile_count_simt);
            }
            else {
              load_v.template step<true>(load_state_v, 
                                         problem_size,
                                         k_tile_iter,
                                         smem_pipe_write, 
                                         k_tile_count_simt);
            }
            // 把commit group放在条件分支之外
            cp_async_fence();
          }
      } // 到此为止，所有StageCount+1条pipeline全满

      // 创建QK阶段的tiled mma
      TiledMmaQK tiled_mma_qk{};
      auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
      // 跟SM90的GMMA不同，warp-level mma必须用寄存器作为操作数
      // 我们先把Q的fragment分配出去，之后就一直保持不动了
      // 理论上mma的fragment尺寸中val-layout是确定的，由mma atom决定
      // 可以变化的只有atom的迭代次数，这些只需要知道smem tensor的shape就够了
      // 当然，原则上partition_fragment_X会“尽量”尊重输入layout的stride模式，但二者实际并无联系
      Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ(_, _, _0{})); // owning, (VAL_MMA,ITER_M,ITER_K)
      // 假设CTA tile = (bQ=64, bK=32, bD=64)，每CTA 4个warp，则每个warp需要给Q分配
      // 16 * 64个FP16，平均每个thread要32个FP16，即16个32bit寄存器
      // 同理，每个warp给K分配32 * 64个FP16，即64个FP16 per thread，或者说32个32bit寄存器
      
      // 我们还需要额外的ldmatrix环节
      auto smem_tiled_copy_q = make_tiled_copy_A(SmemCopyAtomQK{}, tiled_mma_qk);
      auto smem_tiled_copy_k = make_tiled_copy_B(SmemCopyAtomQK{}, tiled_mma_qk);
      auto thr_smem_copy_q = smem_tiled_copy_q.get_thread_slice(thread_idx);
      auto thr_smem_copy_k = smem_tiled_copy_k.get_thread_slice(thread_idx);
      // 确定thread-view下S2R的起点
      // TODO: 这里有个问题，tSsQ和tSsK的ITER_K是否与tSrQ和tSrK的ITER_K相等？
      //       这关系到后面的ldmatrix + mma循环的写法
      //       由于我们的TiledMma没有对K维度进行permutate，因此预期二者应该是相等的
      Tensor tSsQ = thr_smem_copy_q.partition_S(sQ(_, _, _0{})); // non-owning, (VAL_MMA,ITER_M,ITER_K)
      Tensor tSsK = thr_smem_copy_k.partition_S(sK); // non-owning, (VAL_MMA,ITER_N,ITER_K,stages)
      
      // 确定thread-view下S2R的终点
      // ldmatrix的val-layout跟mma不同，需要retile一下，按照ldmatrix的需求重排寄存器
      Tensor tSrQ_cpview = thr_smem_copy_q.retile_D(tSrQ); // (VAL_CPY,ITER_M_CPY,ITER_K_CPY)

      // Prepare: MMA PV
      // 创建PV阶段的tiled mma
      TiledMmaPV tiled_mma_pv{};
      auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
      // 继续估算，每个warp需要给V分配32*64个FP16，或64个FP16 per thread
      // 也就是tOrV需要32个32bit寄存器

      auto smem_tiled_copy_v = make_tiled_copy_B(SmemCopyAtomV{}, tiled_mma_pv);
      auto thr_smem_copy_v = smem_tiled_copy_v.get_thread_slice(thread_idx);

      // 然后组织S2R
      Tensor tOsV = thr_smem_copy_v.partition_S(sOsV); // (VAL_CPY,ITER_N,ITER_K,stages)

      // 确定当前CTA需要在QK阶段的K上迭代多少次
      // 需要指出：上面计算的k_tile_count_simt才是整个FMHA循环中的迭代次数
      // k_tile_count只是一个方便的中间量，因为本算法将FMHA迭代拆成了两个主要部分：
      // 一个是unmask部分，一个是mask部分，两部分使用不同的循环迭代
      // 用k_tile_count这个量来精确控制两个部分的迭代次数
      int k_tile_count = Fusion{}.get_unmasked_trip_count(blk_coord, TileShape{}, problem_size);

      // 这里有个问题：我们的Q只从global中load一次
      // 所以在prologue之后，第一个stage将会是Q，我们必须先等它完毕
      // 这里又想到一个相关的问题：SM90的GMMA直接用smem做操作数
      // 所以把Q load到smem就可以一直复用
      // 而SM80的mma必须用寄存器操作数，所以必须load到寄存器上再使用
      // 这就有两种情况：
      // 一种是我们给Q分配足够大的寄存器，能够保证每个warp都放下完整的Q tile
      // 中自己要用的片段，那么这样Q只需要从smem load一次到寄存器
      // 之后Q的smem就没用了，相当于浪费了smem，但是需要分配更多寄存器
      // 第二种是我们给Q分配较小的寄存器，比如只够每个warp进行一次mma atom
      // 计算所需的片段，这样QK阶段的mma就需要手动迭代atom，且每次从
      // Q的smem中load所需的寄存器片段，这样更节约寄存器，但需要多次load
      // 目前我们先按第一种情况来设计（基本上就相当于照搬SM90），后面等
      // 我们熟悉之后再优化
      // 这么早就开始wait，是为了给Q的ldmatrix留时间？
      cp_async_wait<StageCount>();
      // 别忘了必须CTA同步才能：
      // 1.将SIMT的load扩展到CTA tile
      // 2.插入compiler fence，防止提前读写Q相关的信息
      __syncthreads();

      // 首次，也是唯一一次S2R for Q，之后就再也不需要了
      copy(smem_tiled_copy_q, tSsQ, tSrQ_cpview);

      // mapping into QK accumulator
      // 这个predicate应该是给online softmax用的
      Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));
      Tensor tPcP = thr_mma_qk.partition_C(cP);
      int m_block = get<0>(blk_coord);
      // E<0>{}可以理解为一个平行于第一维的单位基
      tPcP.data() = tPcP.data() + E<0>{} *m_block* get<0>(TileShapeQK{});

      // Allocate PV acc
      // 原版代码这里为什么要刻意按照CTA tile shape来分配acc？仔细想了一下
      // 因为原版的tiled mma都是从CUTLASS现成的GEMM collective API中借用的
      // 不敢保证其原本的tiled mma mnk恰好等于CTA tile shape
      // 所以这里显式要求按照后者的形状分配
      // 而我们自己的代码，因为是自己根据CTA tile shape构造的tiled mma
      // 所以其mnk已经符合要求了，可以省略partition_fragment_C的第二个参数
      Tensor acc_pv = partition_fragment_C(tiled_mma_pv, take<0, 2>(TileShapePV{}));
      // 每个warp需要分配给acc_pv总计16*64个FP32，或者说每线程32个32bit寄存器
      // 把它清零
      clear(acc_pv);
      /*CUTE_UNROLL
        for (int i = 0; i < size(acc_pv); ++i) {
          acc_pv(i) = ElementAccumulator(0);
        }*/
      
      // 我们还是借用原版的online softmax的实现
      cutlass::fmha::collective::CollectiveSoftmax<ElementAccumulator, Fusion, decltype(params)> softmax{ params };
      auto softmax_state = softmax.init(acc_pv, tiled_mma_pv);

      // 我一直很好奇，为什么很多人喜欢这么写一个if (true) {...} 
      // 进入Peeled block
      if (true)
      {
        // 进行第一轮QK - softmax - PV的计算
        // 注意，虽然k_tile_count最初是用unmasked tile count初始化的
        // 但不意味着peeled block就一定做的是unmasked tile的计算
        // 理由如下：
        // 1. prologue已经提前load了最初的KV tile，这是不区分masked与否的
        // 2. peeled block中排空pipeline跟tile count无关，只看pipeline中
        //    是否真的存在有效的in-flight tile
        // 3. 因此，即便k_tile_count = 0，peeled block获得的其实是第一批
        //    masked KV tile
        // 4. peeled block的计算过程不区分masked和unmasked tile，因此无论获得
        //    的是哪类tile，都能正确计算
        --k_tile_count;
        // Allocate QK acc
        // 这个写法已经解释过了
        // acc_qk占用的寄存器预算是每warp 16*32个FP32，或者说每线程16个32bit寄存器
        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));
        // 后面 acc_qk 会被 convert_c_layout_to_a_layout_sm80 重解释成 PV 的 A 操作数，
        // 这里再次检查ITER_M是否为1，因为这关系到layout是否能convert。
        CUTE_STATIC_ASSERT_V(size<1>(acc_qk) == _1{}, "MMA_M 必须为 1");
        // 先把acc清零
        clear(acc_qk);
        /*CUTE_UNROLL
          for (int i = 0; i < size(acc_qk); ++i) {
            acc_qk(i) = ElementAccumulator(0);
          }*/

        // 我们需要wait cp.async的group
        // 注意，因为已经排空了LoadQ的pipeline，所以现在
        // 只有StageCount条in-flight的pipeline了
        cp_async_wait<StageCount - 1>();

        // MMA QK
        // 这些是GMMA用于布设compiler fence的操作
        // 我们用__syncthreads()替代它们
        //warpgroup_fence_operand(acc_qk);
        //warpgroup_arrive();
        __syncthreads();
        // 到此，in-flight的pipeline减少到StageCount - 1
        // 但Q和第一块K tile尚未被使用，ring-buffer没有空slot
        // 所以无法立即补一条pipeline

        // 这里是很大的不同：GMMA可以直接在smem上计算
        // 但是warp level的mma不行，我们只能先把数据load到寄存器上
        // 然后再计算
        //gemm_zero_acc(tiled_mma_qk, tSrQ(_, _, _, _0{}), tSrK(_, _, _, smem_pipe_read.index()), acc_qk);
        //warpgroup_commit_batch();
        //       分配QK阶段的B operand的fragment
        //       然后从K的smem上ldmatrix填充fragment
        //       最后再调用tiled_mma_qk进行warp level的mma计算
        Tensor tSrK = thr_mma_qk.partition_fragment_B(sK(_, _, _0{}));
        Tensor tSrK_cpview = thr_smem_copy_k.retile_D(tSrK);

        auto ITER_K_QK = size<2>(tSrQ);
        
        // 为了让ldmatrix跟mma交错，我们把ITER_K维度的循环打碎。
        // retile_D 只保证 ITER_K 不变：当 mma 的 FrgV(4) 小于 copy atom 的 val 数(8) 时，
        // 差额是从 ITER_N 里吸收的，所以循环只能按 k_block 切。
        // 同 examples/cute/tutorial/sgemm_sm80.cu:417-418
        // 这一点之前已经多次指出过了
        CUTE_STATIC_ASSERT_V(size<2>(tSrK) == size<2>(tSrK_cpview));
        CUTE_STATIC_ASSERT_V(size<2>(tSsK) == size<2>(tSrK));
        // A 侧（Q）的 K 迭代数必须与 B 侧（K）一致 —— 循环是由 tSrQ 驱动的
        CUTE_STATIC_ASSERT_V(size<2>(tSrQ) == size<2>(tSrK));

        Tensor tSsK_p = tSsK(_, _, _, smem_pipe_read.index());
        for_each(make_int_sequence<ITER_K_QK>{}, [&](auto k_block) {
          copy(smem_tiled_copy_k, tSsK_p(_, _, k_block), tSrK_cpview(_, _, k_block));
          // 我要是写gemm而不是cute::gemm，好像会被当成cutlass::gemm这个namespace的名字...
          cute::gemm(tiled_mma_qk, acc_qk, tSrQ(_, _, k_block), tSrK(_, _, k_block), acc_qk);
        });

        // 确保我们真的使用完这块smem再更新smem_pipe_read
        ++smem_pipe_read;

        // Wait for the pipeline MMAs to drain
        // 这些fence在warp level mma的场合下用不上
        // 因为warp level mma指令是由硬件流水线管理的
        //warpgroup_wait<0>();
        //warpgroup_fence_operand(acc_qk);

        // online softmax的首轮
        softmax.step(acc_qk, tiled_mma_qk, tPcP, softmax_state, problem_size);

        // 接下来给PV阶段准备输入A operand
        // 因为QK阶段的寄存器acc_qk是ElementAccumulator精度的（F32）
        // 不能直接参与mma计算，所以我们只能再分配一个等大的Element精度（F16）的寄存器
        // 这里调整一下val-layout，从C operand的形式变成A operand的形式
        // 似乎不能照抄原版的layout变换？
        // 看convert_c_layout_to_a_layout的实现：
        //convert_c_layout_to_a_layout(CLayout const& c, AValueShape const& a) {
        //  return make_layout(
        //    make_shape(a, shape<1>(c), make_shape(shape<2>(c), size<0>(c) / size(a))),
        //    make_stride(stride<0>(c), stride<1>(c), make_stride(stride<2>(c), size<2>(a) * stride<0, 2>(c))));
        //}
        // 输入的c和a都没问题
        // shape<1>(c)表示ITER_M，没有问题，实际上在本例中就是1
        // shape<2>(c)表示ITER_N，也没问题，本例中就是(BlockKV / 16)
        // size<0>(c)/size(a)这里就有问题了，对于mma.m16n8k16，size<0>(c)=(m16*n8)/32=4
        // 而size(a)=(m16k16)/32=8，所以size<0>(c)/size(a)会向下取整到0
        // 此时构建的layout就不符合我们的要求了
        // 原因在于GMMA的N维度可以很大，而M和K维度固定，所以size<0>(c)可以足够大，能够除得开
        // 这在warp level mma中就不成立，我们需要稍加修改，从ITER_N里“借”一点过来
        // 具体做法请参考convert_c_layout_to_a_layout_sm80()的实现代码
        Tensor acc_qk_fixed = make_fragment_like<Element>(
          convert_c_layout_to_a_layout_sm80(acc_qk.layout(),
             shape<1>(typename decltype(tiled_mma_pv)::LayoutA_TV{})));
        // acc_qk_fixed的寄存器预算由于精度减半，所以只需要acc_qk的一半，即每线程8个32bit寄存器
        // 到此位置，每个线程的寄存器预算（峰值）达到了136个32bit寄存器

        // 然后按照(m,n)的对应关系做cast（从F32 cast到F16）
        Tensor acc_qk_input = make_tensor(acc_qk_fixed.data(), acc_qk.layout());
        CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(acc_qk); i++) {
            acc_qk_input(i) = static_cast<Element>(acc_qk(i));
          }

        // 等待第一块V tile完成load
        // 注意：此时in-flight的pipeline数量进一步减少到了StageCount-1
        cp_async_wait<StageCount - 2>();

        // MMA PV
        //warpgroup_fence_operand(acc_pv);
        //warpgroup_fence_operand(acc_qk_fixed);
        //warpgroup_arrive();
        __syncthreads();
        // 此时，in-flight pipeline减少到StageCount - 2


        // 分配PV阶段的B operand的fragment（A已经有了，就是acc_qk_fixed）
        // 然后从V的smem上ldmatrix填充fragment
        // 最后再调用tiled_mma_pv进行warp level的mma计算
        //gemm_zero_acc(tiled_mma_pv, acc_qk_fixed, tOrV(_, _, _, smem_pipe_read.index()), acc_pv);
        //warpgroup_commit_batch();
        Tensor tOrV = thr_mma_pv.partition_fragment_B(sOsV(_, _, _0{}));
        // retile得到S2R的终点
        Tensor tOrV_cpview = thr_smem_copy_v.retile_D(tOrV);  // (VAL_CPY,ITER_N,ITER_K)

        CUTE_STATIC_ASSERT_V(size<2>(tOrV) == size<2>(tOrV_cpview));
        CUTE_STATIC_ASSERT_V(size<2>(tOsV) == size<2>(tOrV));
        // peel 轮的 A 操作数是整块的 acc_qk_fixed，它的 K 模必须与 tOrV 对齐
        CUTE_STATIC_ASSERT_V(size<2>(acc_qk_fixed) == size<2>(tOrV));

        Tensor tOsV_p = tOsV(_, _, _, smem_pipe_read.index());
        auto ITER_K_PV = size<2>(tOrV);
        for_each(make_int_sequence<ITER_K_PV>{}, [&](auto k_block) {
          copy(smem_tiled_copy_v, tOsV_p(_, _, k_block), tOrV_cpview(_, _, k_block));
          cute::gemm(tiled_mma_pv, acc_pv, acc_qk_fixed(_, _, k_block), tOrV(_, _, k_block), acc_pv);
        });

        //
        // Advance the pipe
        //

        // Advance consumer pipeline
        ++smem_pipe_read;

        // 更新softmax的predicate tensor
        tPcP.data() = tPcP.data() + E<1>{} *get<1>(TileShapeQK{});
      }
      // 第一轮QK-softmax-PV结束，现在in-flight的pipeline只剩下StageCount-2了
      // 由于要求StageCount >= 4，所以此时仍有in-flight的pipeline

      // 接下来为unmasked tiles进行mainloop
      // 所谓unmasked，指的是不需要借助mask就能进行计算的tile
      // 上面peeled block已经对k_tile_count减1了，所以此时可能已经小于0
      // 此时相当于没有其他的unmasked KV tile了，会自动跳过这个mainloop
      CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > 0; --k_tile_count)
        {
          // Allocate QK acc
          Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));
          // 把acc清零（因为acc_qk总是临时分配的）
          clear(acc_qk);
          /*CUTE_UNROLL
            for (int i = 0; i < size(acc_qk); ++i) {
              acc_qk(i) = ElementAccumulator(0);
            }*/

          // 等待本轮的K tile就绪
          cp_async_wait<StageCount - 3>();

          // MMA QK
          /*warpgroup_fence_operand(acc_qk);
          warpgroup_arrive();*/
          __syncthreads();

          /*gemm_zero_acc(tiled_mma_qk, tSrQ(_, _, _, _0{}), tSrK(_, _, _, smem_pipe_read.index()), acc_qk);
          warpgroup_commit_batch();*/
          // 继续执行本轮的QK mma，跟Peel loop一样
          Tensor tSrK = thr_mma_qk.partition_fragment_B(sK(_, _, _0{}));
          Tensor tSrK_cpview = thr_smem_copy_k.retile_D(tSrK);

          CUTE_STATIC_ASSERT_V(size<2>(tSrK) == size<2>(tSrK_cpview));
          CUTE_STATIC_ASSERT_V(size<2>(tSsK) == size<2>(tSrK));
          // A 侧（Q）的 K 迭代数必须与 B 侧（K）一致 —— 循环是由 tSrQ 驱动的
          CUTE_STATIC_ASSERT_V(size<2>(tSrQ) == size<2>(tSrK));

          auto ITER_K_QK = size<2>(tSrQ);
          Tensor tSsK_p = tSsK(_, _, _, smem_pipe_read.index());
          for_each(make_int_sequence<ITER_K_QK>{}, [&](auto k_block) {
            copy(smem_tiled_copy_k, tSsK_p(_, _, k_block), tSrK_cpview(_, _, k_block));
            cute::gemm(tiled_mma_qk, acc_qk, tSrQ(_, _, k_block), tSrK(_, _, k_block), acc_qk);
            });

          // 确保我们真的使用完这块smem再更新smem_pipe_read
          ++smem_pipe_read;

          // 提交load下一轮K tile的cp.async指令
          // 注意，跟prologue一样，load K不能推进iter
          load_k.template step<false>(load_state_k,
            problem_size,
            k_tile_iter,
            smem_pipe_write,
            k_tile_count_simt);
          // 别忘了commit
          cp_async_fence();

          // Wait for the pipeline MMAs to drain
          /*warpgroup_wait<0>();
          warpgroup_fence_operand(acc_qk);
          warpgroup_fence_operand(acc_pv);*/

          // 继续online softmax
          softmax.template step_interleave_begin<false>(acc_qk, tiled_mma_qk, tPcP, softmax_state, acc_pv, tiled_mma_pv, problem_size);

          // 等待本轮的V tile
          cp_async_wait<StageCount - 3>();
          __syncthreads();

          // MMA PV  
          auto layout_qk_input = convert_c_layout_to_a_layout_sm80(acc_qk.layout(), 
            shape<1>(typename decltype(tiled_mma_pv)::LayoutA_TV{}));

          Tensor acc_qk_input = make_tensor(acc_qk.data(), layout_qk_input);
          Tensor tOrV = thr_mma_pv.partition_fragment_B(sOsV(_, _, _0{}));
          Tensor tOsV_p = tOsV(_, _, _, smem_pipe_read.index());
          Tensor tOrV_cpview = thr_smem_copy_v.retile_D(tOrV);  // (VAL_CPY,ITER_N,ITER_K)

          // 同 QK 侧：retile_D 只保证 ITER_K 不变
          CUTE_STATIC_ASSERT_V(size<2>(tOrV) == size<2>(tOrV_cpview));
          CUTE_STATIC_ASSERT_V(size<2>(tOsV) == size<2>(tOrV));
          // 下面的 i 循环由 tOrV 的 K 模驱动，acc_qk_input 被动跟随，两者必须一致
          CUTE_STATIC_ASSERT_V(size<2>(tOrV) == size<2>(layout_qk_input));
          CUTE_STATIC_ASSERT_V(
            rank<0>(layout_qk_input) == rank(stride<1>(typename decltype(tiled_mma_pv)::LayoutA_TV{})),
            "tensor_op_mk_v 要求 val 模与 A 的 val stride 同构（SM80: 必须是扁平 rank-3）");
          // 要求在M方向不迭代
          CUTE_STATIC_ASSERT_V(size<1>(layout_qk_input) == _1{});
          // interleave online softmax和mma
          // 在SM80上，还得加上ldmatrix
          // outer loop在循环PV mma的K方向（reduction方向）
          CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size<2>(tOrV); i++) {
              // 把ldmatrix跟gemm穿插在一起
              copy(smem_tiled_copy_v, tOsV_p(_, _, i), tOrV_cpview(_, _, i));

              Tensor acc_qk_element = make_fragment_like<Element>(layout_qk_input(_, _0{}, _0{}));
              Tensor acc_qk_element_mk = tensor_op_mk_v(tiled_mma_pv, acc_qk_element);
              Tensor acc_qk_input_mk = tensor_op_mk_v(tiled_mma_pv, acc_qk_input(_, _0{}, i));

              // online softmax，针对刚刚完成的一段K-iteration
              softmax.step_interleave_step(acc_qk_input_mk, softmax_state);

              // 从F32 cast到F16获取正确的operand A
              CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < size(acc_qk_element_mk); j++) {
                  acc_qk_element_mk(j) = static_cast<Element>(acc_qk_input_mk(j));
                }
              //warpgroup_arrive();
              
              // inner loop是在循环PV mma的N维度
              // 话说有什么必要把这个inner loop写一遍呢？
              CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < size<1>(tOrV); j++) {
                  cute::gemm(tiled_mma_pv, acc_pv(_, _0{}, j), acc_qk_element, tOrV(_, j, i), acc_pv(_, _0{}, j));
                }
            }
          //warpgroup_commit_batch();

          // Wait for the pipeline MMAs to drain
          /*pipeline.consumer_release(smem_pipe_release);
          ++smem_pipe_release;*/

          ++smem_pipe_read;

          // 启动下一轮迭代的V tile load
          load_v.template step<true>(load_state_v,
            problem_size,
            k_tile_iter,
            smem_pipe_write,
            k_tile_count_simt);
          // 别忘了commit
          cp_async_fence();

          tPcP.data() = tPcP.data() + E<1>{} *get<1>(TileShapeQK{});
        }

      // 接下来准备进入mask tile迭代，先确定迭代次数
      // 注意：这里用的是+=，而不是=，这是为了避免把peeled block已经处理掉的masked tile漏掉
      k_tile_count += Fusion{}.get_masked_trip_count(blk_coord, TileShape{}, problem_size);

      // 开始mask tile迭代
      // 这里就能看到，如果CTA只有1个masked tile，0个unmasked tile
      // 那么peeled block就已经完成计算了，此时k_tile_count在+1之后恰好回到0
      // 又可以自然而然地跳过这个mainloop了
      CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > 0; --k_tile_count)
        {
          // Allocate QK acc
          Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));
          // 把acc清零（因为acc_qk总是临时分配的）
          clear(acc_qk);
          /*CUTE_UNROLL
            for (int i = 0; i < size(acc_qk); ++i) {
              acc_qk(i) = ElementAccumulator(0);
            }*/

          cp_async_wait<StageCount - 3>();

          // MMA QK
          /*warpgroup_fence_operand(acc_qk);
          warpgroup_arrive();*/
          __syncthreads();

          //gemm_zero_acc(tiled_mma_qk, tSrQ(_, _, _, _0{}), tSrK(_, _, _, smem_pipe_read.index()), acc_qk);
          //warpgroup_commit_batch();
          Tensor tSrK = thr_mma_qk.partition_fragment_B(sK(_, _, _0{}));
          Tensor tSrK_cpview = thr_smem_copy_k.retile_D(tSrK);
          CUTE_STATIC_ASSERT_V(size<2>(tSrK) == size<2>(tSrK_cpview));
          CUTE_STATIC_ASSERT_V(size<2>(tSsK) == size<2>(tSrK));
          // A 侧（Q）的 K 迭代数必须与 B 侧（K）一致 —— 循环是由 tSrQ 驱动的
          CUTE_STATIC_ASSERT_V(size<2>(tSrQ) == size<2>(tSrK));

          auto ITER_K_QK = size<2>(tSrQ);
          Tensor tSsK_p = tSsK(_, _, _, smem_pipe_read.index());
          for_each(make_int_sequence<ITER_K_QK>{}, [&](auto k_block) {
            copy(smem_tiled_copy_k, tSsK_p(_, _, k_block), tSrK_cpview(_, _, k_block));
            cute::gemm(tiled_mma_qk, acc_qk, tSrQ(_, _, k_block), tSrK(_, _, k_block), acc_qk);
            });

          // 确保我们真的使用完这块smem再更新smem_pipe_read
          ++smem_pipe_read;

          load_k.template step<false>(load_state_k,
            problem_size,
            k_tile_iter,
            smem_pipe_write,
            k_tile_count_simt);
          // 别忘了commit
          cp_async_fence();

          // Wait for the pipeline MMAs to drain
          /*warpgroup_wait<0>();
          warpgroup_fence_operand(acc_qk);
          warpgroup_fence_operand(acc_pv);*/

          softmax.step_interleave_begin(acc_qk, tiled_mma_qk, tPcP, softmax_state, acc_pv, tiled_mma_pv, problem_size);

          cp_async_wait<StageCount - 3>();
          __syncthreads();

          // MMA PV  
          auto layout_qk_input = convert_c_layout_to_a_layout_sm80(acc_qk.layout(), 
            shape<1>(typename decltype(tiled_mma_pv)::LayoutA_TV{}));

          Tensor acc_qk_input = make_tensor(acc_qk.data(), layout_qk_input);
          Tensor tOrV = thr_mma_pv.partition_fragment_B(sOsV(_, _, _0{}));
          Tensor tOsV_p = tOsV(_, _, _, smem_pipe_read.index());
          Tensor tOrV_cpview = thr_smem_copy_v.retile_D(tOrV);  // (VAL_CPY,ITER_N,ITER_K)

          // 与 unmasked 循环同一组前提，见那边的详细注释
          CUTE_STATIC_ASSERT_V(size<2>(tOrV) == size<2>(tOrV_cpview));
          CUTE_STATIC_ASSERT_V(size<2>(tOsV) == size<2>(tOrV));
          CUTE_STATIC_ASSERT_V(size<2>(tOrV) == size<2>(layout_qk_input));
          CUTE_STATIC_ASSERT_V(
            rank<0>(layout_qk_input) == rank(stride<1>(typename decltype(tiled_mma_pv)::LayoutA_TV{})),
            "tensor_op_mk_v 要求 val 模与 A 的 val stride 同构（SM80: 必须是扁平 rank-3）");
          CUTE_STATIC_ASSERT_V(size<1>(layout_qk_input) == _1{});
          CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size<2>(tOrV); i++) {
              // 把ldmatrix跟gemm穿插在一起
              copy(smem_tiled_copy_v, tOsV_p(_, _, i), tOrV_cpview(_, _, i));

              Tensor acc_qk_element = make_fragment_like<Element>(layout_qk_input(_, _0{}, _0{}));
              Tensor acc_qk_element_mk = tensor_op_mk_v(tiled_mma_pv, acc_qk_element);
              Tensor acc_qk_input_mk = tensor_op_mk_v(tiled_mma_pv, acc_qk_input(_, _0{}, i));

              softmax.step_interleave_step(acc_qk_input_mk, softmax_state);

              CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < size(acc_qk_element_mk); j++) {
                  acc_qk_element_mk(j) = static_cast<Element>(acc_qk_input_mk(j));
                }
              //warpgroup_arrive();

              CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < size<1>(tOrV); j++) {
                  cute::gemm(tiled_mma_pv, acc_pv(_, _0{}, j), acc_qk_element, tOrV(_, j, i), acc_pv(_, _0{}, j));
                }
            }
          //warpgroup_commit_batch();

          // Wait for the pipeline MMAs to drain
          /*pipeline.consumer_release(smem_pipe_release);
          ++smem_pipe_release;*/

          ++smem_pipe_read;

          load_v.template step<true>(load_state_v,
            problem_size,
            k_tile_iter,
            smem_pipe_write,
            k_tile_count_simt);
          // 别忘了commit
          cp_async_fence();

          tPcP.data() = tPcP.data() + E<1>{} *get<1>(TileShapeQK{});
        }

      // Wait for the pipeline MMAs to drain
      /*warpgroup_wait<0>();
      warpgroup_fence_operand(acc_pv);*/

      Tensor lse = softmax.tail(softmax_state, acc_pv, tiled_mma_pv);

      return make_tuple(acc_pv, lse);
    }
  };

}  // namespace cutlass::fmha::collective

