---
title: neovim 的简易教程 [WIP]
date: 2021-10-31
tags:
- neovim
---
## 前言

最近我打算开一个 neovim 系列，这个博客系列会分成几块来讲，首先是 neovim 的一些键位和
设置，然后再讲如何配置 packer 这个插件管理器，之后就是零零碎碎的写一写插件介绍。

我的配置文件教程会基于 Lua 和 neovim 0.5+ 来配置。所以如果你是打算用 vim 的，因为 vim
仅支持 vimscript，这篇文章可能会不太适合你。

本来是想写一点类似于卖安利的文字的。不过最近实在疲于给别人推荐他们并不感兴趣的东西，
所以我就不在这里多费口舌介绍 neovim 的优势了。这篇文章主要还是写给那些对 neovim 感兴
趣但是不知道怎么入手的人。

## neovim 的一些基础操作

安装好 neovim 之后，在命令行输入 `nvim` 打开 neovim。在打开的界面的左上角，有一个长方
体，那个就是你的光标。中间有许多提示文字，提示让你输入 `:help`, `:q`, `:checkhealth`。

如果你是第一次用 neovim，通常会对这个界面感到困惑，这很正常，不必担心。我会在这一节教
授你一些基础的 neovim 用法，足够你进行基本的文字编辑。

首先我想要讲的是如何退出 neovim。对于每一个第一次使用 vim 的新手，如何退出 vim 就如同
灰烬审判者古达一样，把他们拦在熟练使用 vim 的大门之外。退出 vim 是如此困难，以至于所
有人都在玩退出 vim 的梗。虽然 neovim 默认给 <kbd>Ctrl-c</kbd> 按键绑定上了退出提示，但是
搞明白 neovim 的提示也让新手们费了不少功夫。

![vim-meme](https://i.redd.it/hei6djw6jop71.png)

所有新手最不明白的一个问题可能就是 `:q` 到底是什么意思。在哪里输入 `:q`，

## 配置文件

neovim 的配置通常从一个 init.lua 文件开始。在 Linux 和 Mac OS 系统，你需要把 init.lua
文件放在 `~/.config/nvim` 目录下。在 Windows 系统，你需要把 init.lua 文件放在
`~/AppData/Local/nvim`。具体详尽的目录你可以打开 neovim，输入 `:echo stdpath('config')`。


