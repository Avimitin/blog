---
title: 'HashMap::Entry 的 or_insert 返回了什么'
author: avimitin
date: 2021/08/21 19:52
tag: [rust, std, reference]
categories: rust
---

# HashMap::Entry 的 or_insert 返回了什么

Tags: Language Specific

这几天学 Rust 的时候看到的一块代码给我带来了一点疑惑：

```rust
use std::collections::HashMap;

fn main() {
    let mut hashmap = HashMap::new();

    let foo = hashmap.entry("foo").or_insert(1);
    println!("foo has value {}", foo);

    *foo += 1;
    println!("foo has value {}", foo);

		println!("hashmap now store key-value pair: {:?}", hashmap);
}
```

在这里我定义了一个新的哈希表变量 `hashmap`，并使用 `HashMap.entry().or_insert()` 方法在检测到没有 `"foo"` 这个键的时候，插入键值对 `"foo" = 1` ，并返回一个值，这个值可以进行解引用操作，并自增一位。

编译之后程序会输出：

```bash
$ rustc main.rs && ./main
foo has value 1
foo has value 2
hashmap now store key-value pair: {"foo": 2}
```

可以看到，不止 foo 的值自增了，`hashmap` 里的值也自增了一位。只看代码，貌似 `foo` 是一个指向哈希表里的值的引用。但是没有声明 `mut`，为什么我可以解引用并修改他的值呢，如果要修改我难道不应该用 `let mut foo = ...` 来声明 `foo` 是可写的吗？

这里其实我理解错了，其实 `foo` 本身已经是一个 mutable reference，查看[文档](https://doc.rust-lang.org/std/collections/hash_map/enum.Entry.html#method.or_insert)，可以看到 `or_insert()` 返回的是一个可变引用而不是值。

![lib](/images/rust/What_Does_HashMap_Entry_or_insert_Return/lib.png)

所以在此处解引用指向的是哈希表里的值而不是 `foo` 本身，`or_insert()` 方法已经传回来了一个可写的引用，`foo` 的类型就是一个可写的引用。而我们修改的是哈希表里的值不是 `foo` 自己的值，所以就不需要声明 mutable 啦。

更加直观一点，`foo` 其实就是下面的这个 `ref_integer` 。

```rust
fn main() {
    let mut origin_integer = 1;
    let ref_integer = &mut origin_integer;

    *ref_integer+=1;

    println!("origin_integer: {}", origin_integer);
}
```

`ref_integer` 在此处是一个可写引用，解引用之后就是在修改 `origin_integer` 的值，而我并不是在修改 `ref_integer` 存的引用，所以不需要声明 `ref_integer` 是可写的。
