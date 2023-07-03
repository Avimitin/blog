+++
title = 'Sparse storage scheme used in MLIR'
date = '2023-07-03'
tag = ['mlir', 'sparsity']
author = 'sh1marin'
+++

在 MLIR 里，sparse compiler 会更根据 sparse tensor encoding 的属性，调整 tensor 的存储方式。
SparseTensor Dialect 里提供了 `coordinates`/`values`/`positions` *operation*s，
这三个 *operation*s 有点类似于 `bufferization.to_memref`，将高抽象的 tensor 降至具体的缓存操作，
让我们得以一窥 sparse tensor 的存储结构。

利用这三个操作，我们可以很清晰的看到 MLIR 的 Sparse Compiler 具体对 tensor 使用的存储方式。
在开始介绍之前，我先提前写一个等会可以用来辅助的伪代码， MLIR 操作 `dump`。

```mlir
func.func @dump(%arg: tensor<*xf32, #Attr>) {
  // Iterate all the dimension of the tensor
  affine.for %lvl = 0 to L {
    // Print pointers
    %p = sparse_tensor.positions %arg { level = %lvl : index }
          : tensor<*xf32, #Attr> to memref<?xindex>
    // Convert operation...
    vector.print %p : ...

    // Print indices (coordinate)
    %c = sparse_tensor.coordinates %arg { level = %lvl : index }
          : tensor<*xf32, #Attr> to memref<?xindex>
    // ...
    vector.print %c : ...
  }

  // Print values (non-zero entries)
  %v = sparse_tensor.values %arg : tensor<*xf32, #Attr> to memref<?xf32>
  // ...
  vector.print %v1 : ...

  return
}
```

上面的这个 `@dump` 操作中，我省略了很多类型转换的操作，
以及使用了 `affine.for` 来遍历 tensor 的每一个 dimension。
但实际上 `level` 只接受静态常数，在编译期会用来检查访问是否超越 tensor rank 。

在一个 tensor 内有 0 到指定级别的 dimensions，在指定 tensor 的 sparse attribute 时，
需要用 `lvlType` （旧版本是 `dimLevelType` ）对不同的 dimension 注释稀疏程度。
对于给定 `compressed` 级别的，Sparse Compiler 会对这一层的 tensor 使用如下的存储方式：

```mlir
#SparseVector = #sparse_tensor.encoding<{
  lvlType = [ "compressed" ]
}>

// Create a sparse vector with value 1.1,2.2,3.3,4.4 at v0[3], v0[6], v0[9], v0[12]
%v0 = arith.constant sparse<
  [ [3], [6], [9], [12] ],
  [ 1.1, 2.2, 3.3, 4.4 ]
> : tensor<16xf32>

%sv0 = sparse_tensor.convert %v0 : tensor<16xf32> to tensor<16xf32, #SparseVector>
call @dump(%sv0)
```


```json
pointers[0]: [ 0, 4 ]
indices[0]: [ 3, 6, 9, 12 ]
values: [ 1.1, 2.2, 3.3, 4.4 ]
```

在上面的例子里，因为给定了一个 rank 只有 1 的 tensor，所以 `pointers` 和 `indices`
都只是一个仅有一个元素的数组。values 则永远是一个一维的数组。
`pointers` 数组用来存放访问的 “范围”。
比如上述的例子，`[0, 4]` 用来 `indices` 和 `values` 在访问 tensor dimension 1 时的取值。
使用 sparse vector 做例子会有些令人迷惑：“看起来我不需要多这么一个 pointers 存储我就可以知道我的访问便捷呀？”
所以下面这里再给一个 8 维度，使用 Compressed Sparse Row 存储的例子：

```mlir
// "dense" for first level, "compressed" for the second level
#CSR = #sparse_tensor.encoding<{
  lvlType = [ "dense", "compressed" ]
}>

// Create a sparse matrix %m
// +-----+-----+-----+-----+-----+-----+-----+-----+
// + 0.0 | 1.1 | 0.0 | 0.0 | 2.2 | 0.0 | 0.0 | 0.0 |
// +-----+-----+-----+-----+-----+-----+-----+-----+
// + 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
// +-----+-----+-----+-----+-----+-----+-----+-----+
// + 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
// +-----+-----+-----+-----+-----+-----+-----+-----+
// + 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
// +-----+-----+-----+-----+-----+-----+-----+-----+
// + 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
// +-----+-----+-----+-----+-----+-----+-----+-----+
// + 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
// +-----+-----+-----+-----+-----+-----+-----+-----+
// + 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
// +-----+-----+-----+-----+-----+-----+-----+-----+
// + 0.0 | 0.0 | 3.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
// +-----+-----+-----+-----+-----+-----+-----+-----+
//
// with value 1.1 in m[0, 1], 2.2 in m[0, 4], 3.3 in m[7, 2]
// with rank 2, shape 8x8, dtype f32
%m = arith.constant sparse<
  [ [0, 1], [0, 4], [7, 2] ],
  [  1.1,    2.2,    3.3  ]
> : tensor<8x8xf32>

%sm = sparse_tensor.convert %m : tensor<8x8xf32> to tensor<8x8xf32, #CSR>
call @dump(%sm) : (tensor<8x8xf32, #CSR>) -> ()
```

```json
d1: dense
d2:
    pointers[1]: [ 0, 2, 2, 2, 2, 2, 2, 3, 3 ] # Length = 9
    indices[1]: [ 1, 4, 2 ]
    values: [ 1.1, 2.2, 3.3 ]
```

使用 CSR 结构的存储，第一层会用 dense 的缓存来存储，第二层才使用特殊的数据结构。
其中 pointers 的含义依旧不变，仍旧是用来指引 indices 和 values 的 range。
抽象总结的来说，对于 \\(row i\\)，设 `pointers[1][i-1]` 为 \\(start\\)，设 `pointers[1][i]` 为 \\(end\\)，
则范围 \\([start, end)\\) 指引了 \\(row i\\) 上的 indices 和 values 的取值范围。

举例来讲，假设我目前想访问 \\(row 1\\) 上的值，则 `pointers[1][0]`, `pointers[1][1]` 设置
访问的 range 为 \\([0, 2)\\)，意味着仅有 `indices[1][0..2]`，`values[0..2]` 的值是属于 \\(row 1\\) 的，
即 `[1, 4]` 和 `[ 1.1, 2.2 ]`。而我想访问 \\(row 8\\) 上的值，则 `pointers[1][6], pointers[1][7]` 设置
访问的 range 为 \\([2, 3)\\)，即对应 indices 的值 `2` 和 values 的值 `3.3`。
对于一个零值而言，访问其 row 则只能得到非法 range \\([n, n)\\)，取不到值，则返回零。

现在再和上面的矩阵对比，我想 `pointers` 的意义就一目了然了。
