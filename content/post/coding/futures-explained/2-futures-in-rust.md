+++
title = 'Rust 的 futures 实现（WIP）'
date = '2022-05-03'
tag = ['rust', 'async', 'futures']
author = 'sh1marin'
+++

> 翻译自 <https://cfsamson.github.io/books-futures-explained/1_futures_in_rust.html>

# 2. Rust 的 Futures 实现

> 在这一章中，你将会学到：
>
> - Rust 的高层级并发概念
> - 当你写 async 代码时，Rust 为你提供的东西和缺少的东西
> - 为什么我们在写 async 时需要一个 runtime 库
> - leaf-future 和 non-leaf-future 之间的区别
> - 如何处理 CPU 密集任务

## Future

所以 future 是什么？

future 用来代表一系列在现在不能立即完成，要过一段时间才能完成的操作。

Async Future 的完成在 Rust 里主要使用轮询(`Poll`)来实现，其中，一个异步任务将需要三个阶段来完成。

1. **轮询阶段(The Poll phase)**。当一个 `Future` 被轮询时，与其对应的任务继续将执行，
直至这个任务被执行到了某个阻塞操作，无法立刻完成而结束。
我们通常把 **运行时(Runtime)** 中对 `Future` 进行轮询的功能称作为 **Executor**。
2. **等待阶段（The Wait phase)**。一个通常被称作 `reactor` 的事件资源，能够把 Future
的状态改变为等待状态，标注其正在等待某个事件的发生。这个 reactor 事件源可以确保当
Future 需要的事件发生时唤醒 Future 并继续执行任务。
3. **唤醒阶段(The Wake phase)**。事件发生，`reactor` 唤醒 `Future`。然后轮到 `Executor`
继续轮询 Future，让这个 Future 从刚才的断点继续执行下去，直到遇到下一个阻塞操作或者
该 Future 已经完全完成才结束这一整个循环。

接下来我们将要讨论一下 leaf-future 和 non-leaf-future。把这两个概念区分清楚是很重要的事情，
因为在生产实践中，这两种 future 差别是很大的。

## Leaf futures

*leaf future* 由运行时 (Runtime) 创建，用来代表像套接字 (socket) 这样的资源。

```rust
let mut stream = tokio::net::TcpStream::connect("127.0.0.1:3000")
```

> 译者注：诸如 [tokio](https://tokio.rs)/[async-std](https://async.rs/) 这样的
> crates 提供了运行时库 (Runtime Library)。
> 而 async await 则是 rustc 内置的语法糖。这部分的概念容易搞混，请一定要区分开。

对这些资源的操作，比如对 socket 进行读写操作，都是非阻塞操作。
这些操作会返回一个 future。这个返回的 future 就是 leaf 类型的 future。
称其为 leaf-future 的原因是，这种 future 是真的在被等待完成的 future。

除非你在自己写运行时(Runtime)，一般来说你都不需要自己实现 leaf-future。
不过这个教程将会把 leaf-future 的实现给过一遍。除此之外，你也不太可能
会需要将一个 leaf-future 交给运行时来执行。在下一章你就会知道原因。

## non-leaf-future

non-leaf-future 则是运行时的使用者，也就是我们这些用户来创造的。
我们使用 `async` 关键字来创造一个异步的 *task*，然后交给运行时来执行。

一个异步的程序主要由 non-leaf-future 来组成，你可以把这种 future 看作一系列可以暂停的计算。
把概念弄清楚非常重要，因为这种 future 只代表一系列的操作。通常，这种 future
会用 `await` 来等待 leaf future 执行完成。

```rust
// Non-leaf-future
let non_leaf = async {
    let mut stream = TcpStream::connect("127.0.0.1:3000").await.unwrap();// <- yield
    println!("connected!");
    let result = stream.write(b"hello world\n").await; // <- yield
    println!("message sent!");
    ...
};

non_leaf.await
```

> 这里 `async {  }` 把内部的各个操作包起来，然后把这一系列的操作包成 future
> 并传递给 `non_leaf` 这个变量。最后你可以等待新生成的 `non_leaf`。

在上面的程序中，最关键的点在于那几个 `await` 关键字。每次调用 `await` 关键字
都会把当前的控制权交给运行时(Runtime)的任务管理器。当任务可以继续完成时，控制权
会重新交回给这个 future，而这个 future 不会从头开始执行，而是从刚刚 `await` 的地方继续执行。

相对于 leaf-future 而言，non-leaf-future 不被用来等待 I/O 资源。轮询 non-leaf-future
会让 future 运行到 leaf-future 的执行点，然后当 leaf-future 返回 pending 时，控制权
就被交还给 Runtime 管理了。

## Runtimes

像 C#, JavaScript, Java, Golang 以及其他众多语言通常都会自带运行时来处理并发。
在这些语言中，你只需要使用默认提供的 Runtime 库来管理并发就可以了。
和这些语言相比，Rust 并不自带处理并发用的运行时。你需要自己在众多的库中做出选择并导入到
自己的项目中。
所以如果你带着其他语言的并发模型来学 Rust 的 async await，会有很多纳闷的点。

因 Futures 而产生的大量复杂代码其实来源于运行时的复杂性。写一个高效的运行时是一件很难的事情。
而学习如何正确使用运行时也是一件需要费功夫的事情。不过你会发现在不同的运行时之间都有一些相似性。
学会使用一个 Runtime 能让你更轻松的阅读后面的篇章。

### async runtime 的概念模型

我发现制造一个高层级的思维模型能更好的推断出 Futures 是怎么工作的。
要做到这一点，必须要介绍这个能执行完我们 Futures 的 runtime 的概念

**Rust 中一个完整的 async 系统将可以分成以下三部分**：

1. Reactor
2. Executor
3. Future

这三个部分是如何协同工作的呢？他们通过一个叫做 `Waker` 的对象来实现。
Reactor 使用 `Waker` 来通知 Executor 某个 Future 已经准备好继续被执行了。即把
future 从睡眠状态中唤醒了。

一旦你理解了 `Waker` 的生命周期和所有权，你就能从用户的视角来明白 Future 的机制了。
下面是生命周期的过程：

- Executor 创建一个新的 `Waker`。一个常见，但不必要的做法是为每个在 Executor 注册
过的Future 都创建一个新的 Waker。
- 当 future 在 executor 注册时，它会在 executor 那获得一个 `Waker` 的拷贝。因为
`Waker` 用智能指针传输（就像 `Arc<T>` 那样），所有的拷贝都指向同个地址。因此任何
调用 `Waker` 拷贝的行为都能指向原初的 `Waker`，进而唤醒对应的 Future。
- Future 拷贝 `Waker` 并把它传到 Reactor 来存储，等待稍后使用。

在
