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

`VectorType` 是 MLIR 里用来表达多维的 SIMD 向量的高抽象层级的类型。
在 `VectorType::get` 里，`vectorLength` 用于指定 `vector` 的 shape，
因为传递进的是一个 `unsigned int`，所以这里会构造一个 rank-1，size-1，值为 `vectorLength` 的 `llvm::ArrayRef`，
用来指明 `VectorType` 的 shape。
`etp` 是 Element Type 的缩写，顾名思义用来指定 `vector` 元素的类型。
而 `enableVLAVectorization` 值会传递进第三个参数 `numScalableDims` 里。
`numScalableDims` 是个 `unsigned int` 的参数，默认值是 0。
传递 `enableVLAVectorization` 这个 boolean 值进入 `numScalableDims` 会出现一个 implicit cast，
把这个 field 的值设置为 1，指明可拓展的维度有 1 个。

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
为了避免有任何歧义，下面用 `ForOp` 来指代被向量化的 for 循环操作。

> #### C++ 里的 `ForOp` class 和 MLIR 是什么关系？
>
> 一个典型的 `ForOp` 通常用来描述 MLIR 里一整个 `scf.for` 的操作 block。
>
> ```mlir
> %c0  = arith.constant 0 : index
> %c1  = arith.constant 1 : index
> %c8  = arith.constant 8 : index
>
> scf.for %i = %c0 to %c8 step %c1 {
>   // ...
> }
>
> %0 = arith.constant 0.0 : f64
> %1 = scf.for %i = %c0 to %c8 step %c1 iter_args(%vin = %0) -> f64 {
>   // ...
>   %vout = // operation to %vin
>   scf.yield %vout : f64
> }
> ```
>
> 其可能在循环里创建新值，于是需要 `scf.yield` 返回每次迭代的结果。

因为在调用函数前已经检查过了 `ForOp` 里的 block 数量，
所以此处只拿出 `ForOp` 的第一块 block 进行操作。

```cpp
Block &block = forOp.getRegion().front();
```

如果这块 block 里只有不超过一个的操作，那么只有可能是最后的 `yield` 操作。
这类 `ForOp` 不会被向量化。

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
然后在当前操作位置创建 `VectorScaleOp` 操作，并用 `arith.muli` 操作将 `step` 和 `vscale` 的值相加。

> 总结一下，不开 vla：`step = vlen`，开 vla：`step = vlen + vscale`

```cpp
Value vmask;
if (codegen) {
  Value step = constantIndex(rewriter, loc, vl.vectorLength);
  if (vl.enableVLAVectorization) {
    Value vscale =
        rewriter.create<vector::VectorScaleOp>(loc, rewriter.getIndexType());
    step = rewriter.create<arith::MulIOp>(loc, vscale, step);
// ...
```

上面的 cpp 代码在传递 `vl=4` 和 `enable-vla-vectorization=true` 给 `sparse-vectorization` pipeline 后
的重写 MLIR 里长这样：

```mlir
%c4 = arith.constant 4 : index
%vscale = vector.vscale
%step = arith.muli %vscale, %c4 : index
```

接下来会根据 `ForOp` 的两个可能的操作目的进行两种不同的 for 操作重写。
在 `vectorizeStmt` 里分成了两类，一类是 reduction 另一类是 store。
判断是 `reduction` 类型还是 `store` 类型的方式很简单，只是判断 `scf.yield` 是否有操作数 (operand)。
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

* 对于 reduction 类型的重写，`vectorizeStmt` 会为其生成一个新的循环操作 (`forOpNew`)

```cpp
scf::ForOp forOpNew;
```

然后先把旧 `ForOp` 的一些状态和设定都复制到新 `ForOp` 上，然后修改其迭代的步进值为上述设置的 `step`。

```cpp
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

其中前三个函数调用和 `forOpNew->setAttr(...)` 将原来的 `ForOp` 的内容设置到新的 `ForOp` 上，
而第四个操作 `rewriter.create` 则在创建新函数是，把步进的值改为新的 `step` 值。
最后将后续操作的插入点设置到了新 `ForOp` 的 body 位置。

* 对于 store 类型的重写，`vectorizeStmt` 只会调整其步进的值 (stride) 和 insert 的位置。

```cpp
rewriter.updateRootInPlace(forOp, [&]() { forOp.setStep(step); });
rewriter.setInsertionPoint(yield);
```

第一个函数调用修改了 `ForOp` 的步进值。
第二个函数调用将后续 IR 的插入点设置在原来 for 循环的 `yield` 操作之前。

#### `genVectorMask`

初步调整 `ForOp` 之后，就需要使用 `ForOp` 的参数生成 `vmask` 了。

`vmask` 的值由 `genVectorMask` 函数生成，
这个函数用到了 `ForOp` 的 induction variable，
循环的上下限和步进值 `step`。

> #### [induction variable](https://en.wikipedia.org/wiki/Induction_variable)
>
> ```mlir
> scf.for %i = %lo to %hi
> ```
> 在 MLIR 的 `scf.for` 操作里，induction variable 通常指 `%i`。

在 `genVectorMask` 函数里，
首先会创建一个 1 bit 的 [conditional mask vector](https://en.wikipedia.org/wiki/Predication_(computer_architecture)) *类型*。

```cpp
VectorType mtp = vectorType(vl, rewriter.getI1Type());
// Equals to `vector<4xi1>` when vlen = 4 in MLIR
```

然后判断传入的 `ForOp` 的上下限和步进值的关系。

* 关系满足

如果上下限的差值可以被步进值整除（`((hi - lo) % step) == 0`），
则上面创建的 mask vector 会被全设置为 `true`。
其中 `matchPattern` 还会再一次检查上下限和步进值 `step` 是不是都是常量。

```cpp
IntegerAttr loInt, hiInt, stepInt;
if (matchPattern(lo, m_Constant(&loInt)) &&
    matchPattern(hi, m_Constant(&hiInt)) &&
    matchPattern(step, m_Constant(&stepInt))) {
  if (((hiInt.getInt() - loInt.getInt()) % stepInt.getInt()) == 0) {
    Value trueVal = constantI1(rewriter, loc, true);
    return rewriter.create<vector::BroadcastOp>(loc, mtp, trueVal);
  }
}
```

结合上一层的 `vmask = genVectorMask(..)` 调用，这一段 `ForOp` 改写后生成的 MLIR 类似于：

```mlir
// vlen = 8

// Value trueVal = constantI1(...);
%true = arith.constant 1 : i1
// VectorType mtp = vectorType(vl, I1Type) => vector<8xi1>
// vmask = rewriter.create<vector::BroadcastOp>(loc, mtp, trueVal);
%vmask = vector.broadcast %true : i1 to vector<8xi1>

// %vmask is has value ( 1, 1, 1, 1, 1, 1, 1, 1 )
vector.print %vmask : vector<8xi1>
```

* 关系不满足

如果迭代的长度不能被步进的量整除，比如遍历下限 0 上限 128，但是每次步进量为 3，
那么就要生成另一种 mask 对余数做处理，防止下一次步进会越过 for 循环的上限。

在 `genVectorMask` 里，首先生成了一个新的 AffineMap：

```cpp
auto min = AffineMap::get(
    /*dimCount=*/2, /*symbolCount=*/1,
    {rewriter.getAffineSymbolExpr(0),
     rewriter.getAffineDimExpr(0) - rewriter.getAffineDimExpr(1)},
    rewriter.getContext());
```

这一段生成出来的 MLIR 代码等价于：

```mlir
#min = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
```

接下来会把 `ForOp` 的上限，induction variable 和步进值做成 `ValueRange`，
把它和上面的 `AffineMap` 作为参数传入 `affine.min` 操作中。

```cpp
Value end = rewriter.createOrFold<affine::AffineMinOp>(
    loc, min, ValueRange{hi, iv, step});
```

这里只是看 C++ 代码可能稍微有点难懂，
在这里，rewriter 会创建（或者直接 inline 成值）`affine.min` 操作，
生成类似于以下的代码：

```mlir
scf.for %iv = %lo to %hi step %step {
  %end = affine.min #map (%hi, %iv)[%step]
```

根据上述 `#map` 的定义，
这一个 `affine.min` 操作会变成，
取 `ForOp` 的上限 `%hi` 减去 induction variable `%iv` 的差值，和步进值 `%step` 相比，
返回这两个值中间的最小值。

```python
# Pseudo Python code
min( step, (hi - iv) )
```

最后传入 `VectorType`, `end` 到 `vector.create_mask` 里生成 vmask 值。

```cpp
// VectorType mtp = vectorType(vl, I1Type) => vector<vl x i1>
return rewriter.create<vector::CreateMaskOp>(loc, mtp, end);
```

这个操作最后会返回一个 \\([0, end]\\) 都设置为 1 的 vector mask。

```mlir
// 假设 vl = 8，affine.min 返回 6

%vmask = vector.create_mask %end : vector<8xi1>
vector.print %vmask : vector<8xi1>

// [ 1, 1, 1, 1, 1, 1, 1, 0 ]
```
