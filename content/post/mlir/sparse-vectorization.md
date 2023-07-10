+++
title = 'MLIR - Sparse Vectorization'
date = '2023-07-10'
tag = ['mlir']
author = 'sh1marin'
+++

MLIR Sparse Compiler 里对循环做向量化的方式是靠 Transform OP 的代码重写。
向量化的初步实现实现位于 *lib/Dialect/SparseTensor/Transforms/SparseVectorization.cpp* 里。

## Transform 选项

Sparse Vectorization transform 的操作支持三个自定义选项，
通过 `sparse_tensor::VL` 结构体传递。

```cpp
struct VL {
  unsigned vectorLength;
  bool enableVLAVectorization;
  bool enableSIMDIndex32;
};
```

* `vectorLength`：用来指定向量化时，一个向量的元素数量。
  比如 `vector<16xf32>` 有 16 个元素。
* `enableVLAVectorization`：用来指定是否启用可变长向量，
  启用则提供对 ARMSVE 这类的架构的支持。
* `enableSIMDIndex32`：用来指定是否对 `gather/scatter` 操作使用 32 bit 的索引。

`VL` 结构体会被用在很多地方，比如用来构建 `VectorType`:

```cpp
/// Constructs vector type for element type.
static VectorType vectorType(VL vl, Type etp) {
  return VectorType::get(vl.vectorLength, etp, vl.enableVLAVectorization);
}
```

在 `VectorType::get` 里，`vectorLength` 用于指定 `vector` 的 shape，
因为传递进的是一个 `unsigned int`，所以这里会构造一个 Rank-1 的 `vector`。
`etp` 是 Element Type 的缩写，顾名思义用来指定 `vector` 元素的类型。
而 `enableVLAVectorization` 值会传递进第三个参数 `numScalableDims` 里。
`numScalaDims` 是个 `unsigned int` 的参数，默认值是 0。

## Sparse vectorization rewrite

在 `mlir::populateSparseVectorizationPatterns` 会为 `RewritePatternSet` 加入两个 `Rewriter`，
分别是 `ForOpRewriter` 和 `ReducChainRewriter`，由这两个 `Rewriter` 来负责初步的向量化实现。

### ForOpRewriter

首先先看 `ForOpRewriter`， 其支持上述的三个自定义选项，
主要负责用于向量化对 tensor 的 `for` 操作。
在 `ForOpRewriter` 里，`matchAndRewrite` 成员函数负责代码重写和生成。
其只负责对仅有一个 block 的，步长为 1 的，且由 Sparse Compiler 生成的 for 循环实现向量化。

```cpp
// ForOpRewriter::matchAndRewrite(...)
if (!op.getRegion().hasOneBlock() || !isConstantIntValue(op.getStep(), 1) ||
    !op->hasAttr(LoopEmitter::getLoopEmitterLoopAttrName()))
  return failure();
```

> * `isConstantIntValue` 只有在第一个参数是常量，且值等于第二个参数的时候返回 `true`。
>   在此处表示，只有 for 循环的 step 是常量 1 时返回 `true`。
>
> * `LoopEmitter` 是一个用来管理 sparse tensors 并帮助生成循环操作的一个类。
>   此处用来帮助判断该循环是由 Sparse Compiler 生成的。

区域条件判定结束之后，`matchAndRewrite` 会调用两次 `vectorizeStmt` 函数，
一次传递 `false` 给 `vectorizeStmt` 函数的 `codegen` 参数，用来对原代码做 Analyze；
只有第一次 IR analyze 成功了，才会第二次调用并传递 `true` 以开始对 IR 做向量化。

```cpp
if (vectorizeStmt(rewriter, op, vl, /*codegen=*/false) &&
    vectorizeStmt(rewriter, op, vl, /*codegen=*/true))
  return success();
```

### `vectorizeStmt`

`vectorizeStmt` 是对传入的 `for` 操作实际进行 IR 重写（向量化) 的函数。
因为在调用函数前已经检查过了 for 操作里的 block 数量，
所以此处只拿出 for 循环的第一块 block 进行操作。

```cpp
Block &block = forOp.getRegion().front();
```

如果这块 block 里只有不超过一个的操作，那么只有可能是最后的 `yield` 操作。
这类 for 操作不会被向量化。

```cpp
if (block.getOperations().size() <= 1)
  return false;
```

之后会生成一些之后要用到的变量：

```cpp
// 当前 For 操作的位置
Location loc = forOp.getLoc();
// 将 for 操作 block 内最后一个操作 cast 成 scf.yield
scf::YieldOp yield = cast<scf::YieldOp>(block.getTerminator());
// 拿到 yield 操作的前一个操作（倒数第二个操作）
auto &last = *++block.rbegin();
```

接下来是生成 `vmask`。这部分代码只有第二次调用，也就是设置 `codegen` 为 `true` 时调用。
首先会用上述的 `vectorLength` 选项，先创建一个 `step` 常量的 IR。
接下来，如果启用了对 ARMSVE 平台的编译支持，就会启用上述 `enableVLAVectorization` 变量，
然后在当前操作位置创建 `VectorScaleOp` 操作，并且把 `step` 加上 `muli` 操作。

```cpp
Value vmask;
if (codegen) {
  Value step = constantIndex(rewriter, loc, vl.vectorLength);
  if (vl.enableVLAVectorization) {
    Value vscale =
        rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
    step = rewriter.create<arith::MulIOp>(loc, vscale, step);
```

上面的 cpp 代码在传递 `vl=4` 和 `enable-vla-vectorization=true` 给 `sparse-vectorization` pipeline 后
的重写 MLIR 里长这样：

```mlir
%c4 = arith.constant 4 : index
%vscale = vector.vscale
%step = arith.muli %vscale, %c4 : index
```

接下来会根据 for 循环的两个可能的操作目的进行两种不同的 for 操作重写。
在 `vectorizeStmt` 里分成了两类，一类是 reduction 另一类是 store。
判断是 `reduction` 类型还是 `store` 类型的方式很简单，只是判断 `scf.yield` 是否有操作数。
如果有操作数，那么就是 `reduction` 类型。

结构上长这样：

```cpp
// if codegen block 上面准备的变量
scf::YieldOp yield = cast<scf::YieldOp>(block.getTerminator());
// if (codegen) ...
if (!yield.getResults().empty()) {
  // IR rewrite for reduction
} else {
  // IR rewrite for store
}
```

* 对于 reduction 类型的重写，`vectorizeStmt` 会为其生成一个新的循环

```cpp
scf::ForOp forOpNew;
// if (codegen)
// ...
Value init = forOp.getInitArgs()[0];
VectorType vtp = vectorType(vl, init.getType());
Value vinit = genVectorReducInit(rewriter, loc, yield->getOperand(0),
                                 forOp.getRegionIterArg(0), init, vtp);
forOpNew = rewriter.create<scf::ForOp>(
    loc, forOp.getLowerBound(), forOp.getUpperBound(), step, vinit);
forOpNew->setAttr(
    LoopEmitter::getLoopEmitterLoopAttrName(),
    forOp->getAttr(LoopEmitter::getLoopEmitterLoopAttrName()));
rewriter.setInsertionPointToStart(forOpNew.getBody());
```

其中前三个函数调用和 `forOpNew->setAttr(...)` 将原来的 for 操作的内容设置到新的 for 操作上，
而第四个操作 `rewriter.create` 则在创建新函数是，把步进的值改为新的 `step` 值。
最后将后续操作的插入点设置到了新的 for 操作的 body 位置。

* 对于 store 类型的重写，`vectorizeStmt` 只会调整其步进的值 (stride) 和 insert 的位置。

```cpp
rewriter.updateRootInPlace(forOp, [&]() { forOp.setStep(step); });
rewriter.setInsertionPoint(yield);
```

第一个函数调用将 for 循环的 `step` 值设置为了 `vectorLength`，
如果是开了 VLA 则设置为 `vscale + vectorLength`。
第二个函数调用将后续 IR 的插入点设置在原来 for 循环的 `yield` 操作之前。

TODO: 继续后面的源码阅读
