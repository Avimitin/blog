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

