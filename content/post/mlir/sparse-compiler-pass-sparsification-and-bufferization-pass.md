+++
title = 'MLIR Sparse Compiler - SparsificationAndBufferizationPass'
date = '2023-06-05'
tag = ['mlir']
author = 'sh1marin'
+++

如我[上上篇博客](../sparsity/compiler.md)所提，MLIR 实现稀疏算法的方式是靠类似于 TACO 的 Sparse Compiler 来将普通操作转换成稀疏矩阵操作。
为了不引入新的操作和语义，MLIR sparse_tensor 只引入了一些必要的 operation 和 attribute，写 sparse 算法的
基础数据结构和算法操作还是用的 tensor type 和 linalg Dialect。MLIR 用 [Pattern Rewriter](https://mlir.llvm.org/docs/PatternRewriter/)
在 Pass Pipeline 的时候将这些原本对 Dense 的操作改写成 Sparse 的操作，以此来实现对用户完全透明不可见的 Sparsity Transform.

这篇博客从 sparse compiler pipeline 的注册作为入口，自顶向下的看 sparse_tensor Dialect 是怎么实现 sparse compiler 的。

首先从 sparse tensor 的 pipeline 注册开始看

```mlir
// mlir/lib/Dialect/SparseTensor/Pipelines/SparseTensorPipelines.cpp

void mlir::sparse_tensor::registerSparseTensorPipelines() {
  PassPipelineRegistration<SparseCompilerOptions>(
      "sparse-compiler",
      "The standard pipeline for taking sparsity-agnostic IR using the"
      " sparse-tensor type, and lowering it to LLVM IR with concrete"
      " representations and algorithms for sparse tensors.",
      buildSparseCompiler);
}
```

`PassPipelineRegistration` 是个全局的 Pass Pipeline 注册器，这里注册了一系列的 `SparseCompilerOptions`，
并传递了 `buildSparseCompiler()` 函数指针用来构造 Sparse Compiler。`buildSparseCompiler()` 函数本身
只是个 Pass 注册函数，函数接收了一个 `OpPassManager` 并在里面注册从向量化，IR 规范化的 Pass
到 lowering 的 Pass。所有的 Pass 都会通过 `createXXXPass()` 函数将具体的 Pass 类型 implicit 的 upcast
到父类型 `Pass` 上。比如 `SparseGPUCodegenPass` 的注册函数是先创建了一个 `SparseGPUCodegenPass` 的
unique pointer，然后在 return 语句 cast 到父类型 `Pass` 上。

```c++
std::unique_ptr<Pass> mlir::createSparseGPUCodegenPass() {
  return std::make_unique<SparseGPUCodegenPass>();
}
```

所有注册到 sparse_tensor pipeline 的 Pass

```text
* LinalgGeneralization
* SparsificationAndBufferization
* Canoicalizer
* FinalizingBufferize
* (GPU Codegen: 只有在 sparse-compiler pipeline 里指定了 gpu-triple 才会用的一系列 Pass)

    * SparseGPUCodegenPass
    * StripDebugInfoPass
    * ConvertSCFTOCFPass
    * LowerGpuOpsToNVVMOpsPass
    * GpuToLLVMConversionPass

* ConvertLinalgToLoops
* ConvertVectoToSCF
* ConvertSCFToCF
* ExapndStridedMetadata
* LowerAffine
* ConvertVectorToLLVM (With lowerVectorToLLVMOptions)
* FinalizeMemRefToLLVM
* ConvertComplexToStandard
* ArithExpandOps
* ConvertMathToLLVM
* ConvertComplexToLibm
* ConvertVectorToLLVM
* ConvertComplexToLLVM
* ReconcileUnrealizedCasts
```

而 `SparseCompilerOptions` 则是在 `mlir/include/mlir/Dialect/SparseTensor/Pipelines/Passes.h` 定义。
如果需要看详细可以用 `mlir-opt --help` 查看。需要关注 `SparseCompilerOptions` 提供了一个成员函数：

```c++
ConvertVectorToLLVMPassOptions lowerVectorToLLVMOptions() const {
  ConvertVectorToLLVMPassOptions opts{};
  opts.reassociateFPReductions = reassociateFPReductions;
  opts.force32BitVectorIndices = force32BitVectorIndices;
  opts.armNeon = armNeon;
  opts.armSVE = armSVE;
  opts.amx = amx;
  opts.x86Vector = x86Vector;
  return opts;
}
```

# Passes

接下来来分析 Sparsity 以及向量化相关的 Pass。

## Pass: `SparsificationAndBufferization`

> mlir/lib/Dialect/SparseTensor/Transforms/SparsificationAndBufferizationPass.cpp

`SparsificationAndBufferization` pass 负责处理 Tensor 到 Memref 的 Lower。
拥有 Sparsity Attribute 的 Tensor 会被 Sparsification 专门处理并由专门处理
sparse_tensor Dialect 的 Pass 来 lower。

`SparsificationAndBufferization` Pass 非常灵活，支持许多自定义选项。这些选项
都可以通过 `sparse-compiler` pass 传递进去。

`SparsificationAndBufferization` 目前支持的 Options

```c++
SparsificationAndBufferizationPass(
  const bufferization::OneShotBufferizationOptions &bufferizationOptions,
  const SparsificationOptions &sparsificationOptions,
  const SparseTensorConversionOptions &sparseTensorConversionOptions,
  bool createSparseDeallocs,
  bool enableRuntimeLibrary,
  bool enableBufferInitialization,
  unsigned vectorLength,
  bool enableVLAVectorization,
  bool enableSIMDIndex32
)
```

其中

Dense 的 Tensor 是从 `runDenseBufferization()` 函数走的普通 Bufferization 的
路径 lower。 这个函数会把所有带 Sparsity 属性的 Tensor 都过滤掉，只对 dense 的
operation 做 bufferization。

Sparse Tensor 则是通过函数 `runOnOperatoin()` lower，这个函数会跑三次 Pipeline：

### Pipeline 1

第一次跑 pipeline 主要是负责 sparse tensor 的 rewrite。在第一条 pipeline 里会创建一个新的 `OpPassManager`，
其中加入 `PreSparsificationRewritePass` 和 `EmptyTensorToAllocTensorPass`。

#### PreSparsificationRewritePass

文件位置：mlir/lib/Dialect/SparseTensor/Transform/SparseTensorPasses.cpp

`PreSparsificationRewritePass` 负责处理重写 sparse tensor，像一些转换 dense tensor 到 sparse tensor，
重塑 sparse tensor 等。其主要往 `RewritePattenSet` 里加入
`FoldInvariantYield, FuseSparseMultiplyOverAdd, FuseTensorCast` 三个 `OpRewritePattern`。
其中 `FoldInvariantYield` 负责优化 sparse tensor 里的零值，
`FuseSparseMultiplyOverAdd` 负责合并乘积和加的操作，比如：

```text
T(i,j) = SUM(k, A(i,j,k) * B(i,j,k) * ... )
X(i,j) = S(i,j) * T(i,j)

// After FuseSparseMultiplyOverAdd

X(i,j) = SUM(k, S(i,j) * A(i,j,k) * B(i,j,k) * ... )
```

而 `FuseTensorCast` 负责将 tensor 类型转换操作优化成直接的类型覆写。
其负责三种 rewrite：

1. 消除无意义的 Type cast

如果在使用 tensor.cast 的时候，cast 操作两边的类型完全相同，那么 FuseTensorCast 就会直接把这些 cast
全部优化掉。

以下的代码会被完全优化掉

```mlir
%0 = tensor.cast %a : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
%1 = tensor.cast %0 : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
```

2. 消除多次 tensor cast

如果忽视 sparse 的属性之后，`tensor.cast` 的源类型和目标类型是完全相同的，则这个 tensor.cast 操作会被消除掉，
然后前一个操作产生的 tensor 类型属性会被修改成目标类型的属性。

```mlir
// Before
%extracted_slice = tensor.extract_slice %a[1, 0] [1, 3] [1, 1] : tensor<2x3xi64, #SortedCOO> to tensor<1x3xi64>
%cast = tensor.cast %extracted_slice : tensor<1x3xi64> to tensor<1x3xi64, #Slice>

// After
%extracted_slice = tensor.extract_slice %a[1, 0] [1, 3] [1, 1] : tensor<2x3xi64, #SortedCOO> to tensor<1x3xi64, #Slice>
```

3. 修复错误的 tensor.cast 使用

如果用 tensor.cast 的任意一个操作数有 Sparse 的属性，`FuseTensorCast` 会把这个 operation 换回
`sparse_tensor.convert`

```mlir
// Before
%0 = tensor.cast %a : tensor<?xf32> to tensor<?xf32, #SparseVector>

// After
%0 = sparse_tensor.convert %a : tensor<?xf32> to tensor<?xf32, #SparseVector>
```

### Pipeline 2

第二条 Pipeline 负责对 Tensor 做 Sparse Bufferization 操作。
在这条 Pipeline 里，默认会加入 `SparsificationPass` 和 `PostSparsificationRewritePass`。
根据 `vl` 是否在 `sparse-compiler` pass 里设置了大于零的值，还会可选的加入 `createLoopInvariantCodeMotionPass`
和 `SparseVectorizationPass`。

除此之外，根据 `RuntimeLibrary` 选项是否启用，PassManager 还会可选的加入一部分 Pass。
如果启用了 RuntimeLibrary，则只有 `SparseTensorConversionPass` 会被加入 `OpPassManager`，
如果没启用，那么会加入 codegen pass `SparseTensorCodegenPass`，还有 `SparseBufferRewritePass`
和 `StorageSpecifierToLLVMPass`。

### Pipeline 3

第三条 Pipe line 则是对剩余的 Dense tensor 做 bufferization 操作。
