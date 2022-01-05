---
title: 一个测试终端 IO 速度的脚本
date: 2021-11-14
tags:
- terminal
- linux
- st
thumbnail: /images/linux/Write-A-Script-To-Test-Terminal-IO/screenshot.png
---
## 前言

今天学到了一个新的知识：终端的绘制性能会影响程序的性能。当一个程序以单线程
运行时，终端输出和文件 IO 都在同一个线程，终端如果绘制速度慢，因为 IO 是阻塞的
，程序内部的 stdout 执行速度也会下降。

## 测试

那么有什么办法可以测试一下哪些终端绘制速度比较快呢？我在 Rust CN 群看到的这
么个办法：同时跑文件 IO 和终端 IO 来测试。有一个方式是用 find 来遍历系统内
所有文件，因为 find 毎找到一个文件都会输出一次，我们从根目录开始 find，就可
以实现同时高频率的文件 IO 和终端 IO 了。所以这里我们只需要简单的 `find /` 即
可。

而运行时间的测试则用 shell 提供的 time 函数来测试。

为了比对各个终端，我做了一些额外工作来获取终端的名字。具体脚本如下：

```bash
#!/bin/bash

# 这里获取运行中的终端名字
get_terminal() {
  PARENT_SHELL=$(ps -p $$ -o ppid=)
  PARENT_TERMINAL=$(ps -p $PARENT_SHELL -o ppid=)
  TERMINAL=$(ps -p $PARENT_TERMINAL o args=)
  printf $TERMINAL
}

# 测试
time find /

# 方便分清各个不同的终端
echo -e "\n\nTesting in terminal: $(get_terminal)"
```

## 结果

![result](/images/linux/Write-A-Script-To-Test-Terminal-IO/result.png)

其中 Alacritty 和 simple terminal 打得不相上下，而 kitty 稍逊一筹，konsole
（右下） 是最慢的。后续我再运行了几次结果都相近。

### 数据

| alacritty      | st             | kitty          | konsole |
| -------------- | -------------- | -------------- | ------- |
| 4.209s         | 4.031s         | 5.402s         | 8.539s  |

## 链接

- Alacritty: https://alacritty.org/
- Simple Terminal: https://st.suckless.org/
