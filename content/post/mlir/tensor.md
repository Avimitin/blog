+++
title = 'Tensor Concept'
date = '2023-06-12'
tag = ['tensor', 'mlir']
author = 'sh1marin'
+++


## Tensor 的一些概念

不论是 Pytorch 还是 TensorFlow，Tensor 的表现都和
[Numpy 的 Array](https://numpy.org/doc/stable/user/basics.indexing.html) 类似。

在 PyTorch 里一个 2 维的 Tensor 可以用这样的方式来一探究竟：

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

Output:

```text
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

torch.ones 创建了一个全是浮点数 1 的 4x4 的矩阵。
在 Python 里 `Array[i, j]` 只是 `Array[(i, j)]` 的语法糖，
所以这里可以很清晰的理解 `First column`，即第一列是怎么被索引的。

而在 TensorFlow 里的 tensor 都是 Immutable 的，
一切看起来像是更新的操作，实际上都会创建新的 tensor。

MLIR 的 tensor 概念上和 TensorFlow 的 tensor 比较相近。
在 TensorFlow 里，有常量 tensor：

```python
rank_0_tensor = tf.constant(4)
```

这是一个没有任何轴（没有行列）的，纯常量数值。
而一个 rank 1，只有一个轴的 tensor，表现上则和向量相近：

```python
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
```

一个带有两个轴，具有行列的矩阵，则是一个嵌套的数组：

```python
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
```

这里同时还设置了 dtype 数据类型为 float16。

tensor 不仅限于只有 0-2 个轴，一个向量可以有更多的轴，比如三个轴来表现一个立方体：

```python
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])
```

上述例子是个 3x2x5 的，三个轴（或者称三个维度）的 tensor。
可以观测到，在描述维度的时候，是从外向内的。

一般普通的 tensor 要求所有的边都能成直角，
即在某一个维度或者某一个轴上的数值的数量都应该相等。
但也有特例，比如稀疏矩阵 Sparse Tensor。

在 TensorFlow 里有这么几个词汇用来描述 tensor。
***Shape*** 通常是个数组，用来描述所有轴上元素的数量，比如上述 `rank_3_tensor` 的 shape 是 `[3,2,5]`。
***Rank*** 用来描述 tensor 有几个轴，一个常量 rank 为 0，一个向量的 rank 是 1...
***Axis*** 或者 ***Dimension*** 用来指定 tensor 的某一特定的维度。
***dtype*** 通常指 tensor 内数据的类型。
最后 ***Size*** 用来描述 tensor 里所有元素的数量。

有一个很重要的概念点是，二维 tensor 并不意味 rank 2 tensor，一个 rank 2 的 tensor
很有可能并不用来描述二维空间。

```python
rank_4_tensor = tf.zeros([3, 2, 4, 5])
```

在上述的例子里，rank_4_tensor 是一个 `rank` 为 `4` 的 tensor，其内部所有的元素的值都是 0。
`shape` 是 `[3, 2, 4, 5]`，有 4 个 `axis`，其中 `axis 1` 的长度是 2，
size 则是 `3*2*4*5 = 120`。

TODO: 什么是 batch & feature axis

### Broadcasting

broadcasting 指在某些条件下，当对较小的 tensor 进行组合操作时，
较小的 tensor 会被自动 "拉长 "以适应较大的 tensor。
比如将一个常量与向量做乘积，常量会先变成同等长度的向量，然后再做乘积。
而一个 shape 为 `[3, 1]` 的 tensor 和 shape 为 `[1, 4]` 的向量做乘积运算
会最后生成一个 `[3, 4]` 的 tensor。

## MLIR

在 MLIR 里，tensor 是个原生的类型，存在于 builtin Dialect 里，而对 tensor 的操作则放在 tensor Dialect 里。
这些都是独立的 tensor 创建和修改操作，不会与其他 Dialect 存在耦合。
一些对 tensor 的特殊计算操作会放到其他的 Dialect 里。

MLIR 里的 tensor 支持各类元素类型，可以用来表达 MLIR 里的各种概念，包括但不限于：

* 用来表达适用于高性能计算的大型且密集的元素集合
* 使用 sparse_tensor.encoding 来表达适用于高性能计算的大型稀疏数据集合
* 用小型一维的 tensor 存储 index 类型，来表达 shape Dialect 中的 shapes
* 用于表达字符串或者可变的元素集合

在 MLIR 里，有两种 tensor 类型，一种是 RankedTensorType 另一种是 UnrankedTensorType。
RankedTensorType 的 Rank 是固定的，但 Axis(或者说 Dimension) 是可以动态的。
表达一个 Rank-2 且固定 Shape 的 Tensor 可以用 `tensor<3x2xi32>`，
一个 shape 是 `[3,2]` 的，dtype 是 i32 的 tensor。
也可以用 `tensor<?x?x?xi32>` 来表达 Rank-3 但动态的 tensor。
除此之外，MLIR 的 Tensor 也可以用来表达常量，
比如 `tensor<f32>` 表达一个 Rank 0，元素类型为 f32 的常量。
维度的长度也可以是 0，用类似于 `tensor<0x4xf32>` 的类型来表达。
同时这种特殊的用 `i x j` 来表达 shape 的方式，也限制住不能使用十六进制来定义 axis 的值。

UnrankedTensorType 则用来表达动态 rank 的 tensor 类型，用 `tensor<*xdtype>` 来表示，
比如 `tensor<*xi32>`。

如此灵活的抽象的 tensor 类型是 MLIR 设计的一部分，
使用者不应该操心一个 tensor 的机器表示应该是怎么样的，
这些抽象操作到机器层级的映射最后会被 lower 到 memref Dialect 上，
在那一个层级才有相对底层的缓存访问实现。

有关 tensor 相关操作的文档，可以在 [tensor Dialect](https://mlir.llvm.org/docs/Dialects/TensorOps/)
里查看。

### Bufferization

在 MLIR 里，将 tensor 的操作转换到 memref 操作的这么一个过程被称作为 ***Bufferization***。
最早的时候 MLIR 是在 tensor 到 memref Dialect 转换的时候做 bufferization，但为了
减少内存 alloc 和复制，后来改成了用单个 pass (***One-Shot Bufferization***) 来一次性 bufferize
整个程序。

***Bufferization*** 进程有两个目标：1. 尽量少的申请内存，2. 尽量少的复制内存。
为了实现这两个目标，bufferize 可能会为了复用内存而产生很多很复杂的算法。
给定一个 tensor 的操作结果，***Bufferization*** 需要选择一个 memref buffer 来存储。
最安全的做法是给所有的操作都生成一个新的 buffer，但这肯定是不符合预期的。
而为了安全复用缓存，使得操作不会覆写一些仍然需要的数据，就使得决策变得相当困难复杂。
除此之外，也有一些复杂的情况会影响 ***Bufferization*** 进程的决策，
比如有时候复制内存的开销可能小于重新计算的开销，或者有些平台不支持申请新内存。

为了简化这个问题，***One-Shot Bufferization*** 只对 *destination-passing style* 的操作做 *Bufferization*。
什么是 *destination-passing style* 呢，考虑以下这个例子：

```mlir
%0 = tensor.insert %cst into %t[%idx] : tensor<?xf32>
```

`tensor.insert` 复制一份给定的 tensor，将给定常量插入 index，并返回这个复制的 tensor。
在上述例子中， `%0` 是返回值，`%csr` 和 `%t` 是 `tensor.insert` 的操作数，
因为 `%csr` 是常数，而 `%t` 是个 tensor，因此此处 `%t` 就是 destination，
在考虑如何存放 `%0` 时就只有两个选择：

1. 创建一个新的 buffer
2. 复用操作数 `%t` 的 buffer

在程序执行的过程中可能会存在更多的无用垃圾 buffer 可以拿来复用，
但复用那些内存会引入更加巨量的问题。

如果是不符合 *destination-passing style* 的操作，*Bufferization* 会为这些 tensor 开辟新的内存。

```mlir
%0 = tensor.generate %sz {
^bb0(%i : index):
  %cst = arith.constant 0.0 : f32
  tensor.yield %cst : f32
} : tensor<?xf32>
```

比如此处 `tensor.generate` 只接受一个 `block`，并没有任何的 "*destination*"，
因此 *Bufferization* 会为返回的 tensor 开辟新的内存。
也可以用 `linalg.generic` 改写成 *destination-passing style*：

```mlir
#map = affine_map<(i) -> (i)>
%0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]}
                    outs(%t : tensor<?xf32>) {
  ^bb0(%arg0 : f32):
    %cst = arith.constant 0.0 : f32
    linalg.yield %cst : f32
} -> tensor<?xf32>
```

这里很明显的能看到 `%t` 就是 destination，但同时也能看出一点奇怪的地方，
`outs` 里的参数似乎就是用来被覆写的，为什么还要专门传入一个参数呢？
可以看下下面这个例子:

```mlir
%t = tensor.extract_slice %s [%idx] [%sz] [1] : tensor<?xf32> to tensor<?xf32>
%0 = linalg.generic ... outs(%t) { ... } -> tensor<?xf32>
%1 = tensor.insert_slice %0 into %s [%idx] [%sz] [1]
    : tensor<?xf32> into tensor<?xf32>
```

`tensor.extract_slice` 取出一小段切片，之后会被 *bufferize* 到 `memref.subview`。
然后把这个切片 `%t` 传入到 `linalg.generic` 的 outs 里，
最后 `tensor.insert_slice` 则会将 `%0` 插入被取出 slice 的原 tensor 里。
由这个例子可以看出一个设计传入特定 `out` 的原因：可以用来指定覆写哪一段的内存。

除此之外，值得一提的是，`tensor.insert_slice` 最后会被优化掉。
有 SSA 的加持，MLIR 能发现源操作数来源于目标操作数，于是这些操作能被消除掉。
