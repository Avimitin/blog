---
title: Simple Terminal - simple and suckless terminal
date: 2021-02-10 16:42
categories:
- [system, linux]
- [system, terminal]
tags:
- simple terminal
- terminal
- linux
thumbnail: https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20210210164630.png
tnalt: "simple terminal screenshot"
tldr: "Don't you have a simple and easy-to-use terminal yet? I will help you build it."
---

# Simple Terminal 的搭建和配置

## 前言

Simple Terminal 是一个基于 X 的终端，拥有非常棒的 Unicode 和 Emoji 的支持，同时也支持 256 色，拥有绝大部分的终端特性，但是却极其微小，就算在我打了许多补丁之后，他仍然只占用 108K 的存储空间，快且轻量，是重度终端用户的一个很不错的选择。

![预编译的 st 仅占用 108k](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20210210155347.png)

本篇文章目的在教你打造一个自己的 st，如果没有需求也可以前往我的[仓库](https://github.com/Avimitin/st)克隆我的源码，直接编译安装就可以了。

## 下载源码

Simple Terminal 的官网：<https://st.suckless.org/> ，不需要下载 Download 的那个 st ，直接 clone 源码仓库就好：

```bash
git clone https://git.suckless.org/st
```

## 安装依赖

Simple Terminal (以下简称 st) 需要 `libx11-dev` 和 `libxft-dev` 两个包，对于 Debian 和 Ubuntu 用户来说直接使用 apt 安装即可，Arch 系的大部分发行版都已经包含。

克隆官方的仓库之后，编辑 `config.mk` 文件，编译时 st 会基于这个文件进行配置，一般来说只需要改两行即可：

```makefile
X11INC = /usr/local/X11
X11LIB = /usr/local/X11
```

然后用 root 权限执行编译安装：

```bash
sudo make clean install
```

文件会复制到 `/usr/local/bin/` 目录下，一般直接执行就能启动。

## 配置

一般直接安装就能用了，但是没有人会喜欢一个黑黝黝字体那么丑的终端，所以需要进行一些基础配置。 

st 没有配置文件，所有的配置都是直接编译进二进制文件里的，在第一次执行完 `make install` 之后 st 目录下应该有个 `config.def.h` 和 `config.h` 文件，直接删除 `config.def.h` ，然后修改 `config.h`：

- 修改字体

在第一行就能看到一行 `...font...`  字样，你可以修改里面的字体名和文字大小

```c
static char *font = "Liberation Mono:pixelsize=12:antialias=true:autohint=true";
```

建议你使用 Jetbrains Mono 或者 Fira Mono 这类有 nerd font 补丁的等宽字体，nerd fonts可以在 github 下载：<https://github.com/ryanoasis/nerd-fonts/releases> 。下载完之后解压到 `~/.config/fonts` 文件夹并执行命令 `fc-cache -fv` 刷新字体缓存，然后回到 `config.h` 文件修改字体：

```diff
-static char *font = "Liberation Mono:pixelsize=12:antialias=true:autohint=true";
+static char *font = "JetbrainsMono Nerd Font:pixelsize=24:antialias=true:autohint=true";
```

然后重新执行 `sudo make clean install` 安装。

- 修改主题

你可以在 <http://terminal.sexy/> 上点击 Scheme Browser 选择喜欢的主题，然后点击 Export，Format 选择 `Simple Terminal` 然后复制即可。

打开 `config.h` 文件，找到 `/* Terminal Color...` 这行，把复制的文字替换即可。比如我这里选择了 Tomorrow Night 主题：

```diff
-       "black",
-       "red3",
-       "green3",
-       "yellow3",
-       "blue2",
-       "magenta3",
-       "cyan3",
-       "gray90",
-
-       /* 8 bright colors */
-       "gray50",
-       "red",
-       "green",
-       "yellow",
-       "#5c5cff",
-       "magenta",
-       "cyan",
-       "white",
-
-       [255] = 0,
-
-       /* more colors can be added after 255 to use with DefaultXX */
-       "#cccccc",
-       "#555555",
+       [0] = "#1d1f21", /* black   */
+       [1] = "#cc6666", /* red     */
+       [2] = "#b5bd68", /* green   */
+       [3] = "#f0c674", /* yellow  */
+       [4] = "#81a2be", /* blue    */
+       [5] = "#b294bb", /* magenta */
+       [6] = "#8abeb7", /* cyan    */
+       [7] = "#c5c8c6", /* white   */
+
+/* 8 bright colors */
+       [8]  = "#969896", /* black   */
+       [9]  = "#cc6666", /* red     */
+       [10] = "#b5bd68", /* green   */
+       [11] = "#f0c674", /* yellow  */
+       [12] = "#81a2be", /* blue    */
+       [13] = "#b294bb", /* magenta */
+       [14] = "#8abeb7", /* cyan    */
+       [15] = "#ffffff", /* white   */
+
+       /* special colors */
+       [256] = "#1d1f21", /* background */
+       [257] = "#c5c8c6", /* foreground */
+
 };
```

然后编译安装。

- 打补丁

你也可以给源码打上 diff 补丁来增加 st 的功能。补丁可以在官网 patches 页面下寻找：<https://st.suckless.org/patches/> ，这里我推荐几个比较实用的补丁，别的可以根据自己需求安装。

1. 背景透明 

首先是背景透明：<https://st.suckless.org/patches/alpha/> ，通过给背景加上 alpha 通道实现透明，打开页面下载最新的 0.8.2 版本，放到 st 仓库目录下后，执行命令打上补丁：

```bash
patch < st-alpha-0.8.2.diff
```

它会找不到 config.def.h 然后询问你打到哪，你输入 config.h 即可。

你可能想修改透明度，找到下面这行修改值即可，1 即是不透明

```diff
- float alpha = 0.8;
+ float alpha = 0.2;
```

2. 然后是 [anysize](https://st.suckless.org/patches/anysize/)，可以帮助你的 simple terminal 适应各种大小的拉伸。
3. 如果想在桌面启动 st，可以打上 [DesktopEntry](https://st.suckless.org/patches/desktopentry/) 补丁，他会帮你自动生成 .desktop 文件。
4. 如果你的 st 显示 emoji 很奇怪你可能还会需要 [fontfix 补丁](https://github.com/Avimitin/st/blob/master/patches/st-fontfix.diff)
5. 除此之外你还需要滚动终端显示内容，所以需要打上 [scrollback 补丁](https://st.suckless.org/patches/scrollback/)，默认下摁 Alt + PageUp 或 PageDown 翻页。

如果你先修改上翻下翻的键位，可以参考下面：

```diff
-	{ ShiftMask,            XK_Page_Up,     kscrollup,      {.i = -1} },
-	{ ShiftMask,            XK_Page_Down,   kscrolldown,    {.i = -1} },
+	{ MODKEY,               XK_u,           kscrollup,      {.i = 1} },
+	{ MODKEY,               XK_j,           kscrolldown,    {.i = 1} },
+	{ MODKEY|ControlMask,   XK_u,           kscrollup,      {.i = -1} },
+	{ MODKEY|ControlMask,   XK_j,           kscrolldown,    {.i = -1} },
```

这里我把上翻一行改为了 Mod + u （Mod 键即使 alt 键），下翻一行改为了 Mod + j，上翻一页改为 Mod + Ctrl + u。

6. [font2](https://st.suckless.org/patches/font2/)，可以帮助你把 emoji 显示不正确的问题通过增加 fallback 的形式修复，你就不需要改变喜欢的字体了。

> 如果在打补丁的时候遇到打不上错误的问题，他会在本地生成 config.h.rej 文件你把里面带 + 号的复制到 config.h 的指定位置，把带 - 的在 config.h 里找到删除即可。

## 最后

你也可以免去所有麻烦直接克隆我配置好的文件：

```bash
git clone git@github.com:Avimitin/st.git
cd st
sudo make clean install
```

或者前往 [release](https://github.com/Avimitin/st/releases)（可能有的修改不会及时上传）下载我预编译的 `linux-amd64` 版本。
