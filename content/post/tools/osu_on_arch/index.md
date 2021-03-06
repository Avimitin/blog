+++
title = '在 Arch Linux 上玩 osu stable'
date = '2022-04-27'
tag = ['osu', 'Arch Linux']
author = 'sh1marin'
+++
![screenshot](./images/screenshot.png)

# 前言

在 Arch Linux 上玩 osu 是我许久以来的愿望了。
我现在日常在 Linux 上工作，但有时候手痒想玩游戏的时候却总要重启，
切双系统到 Windows 上。因为历史原因，osu stable 没有全平台适配。
而做了全平台适配的 osu lazer 目前仍在开发阶段，在 lazer 上打出的
成绩不提交服务器。且 lazer 的键位机制和原版也相去甚远，需要重新适应。

但好在我们还有 Wine。大约在 2009 年的时候 osu 论坛就已经有了用 Wine
运行 osu 的教程：<https://osu.ppy.sh/community/forums/topics/14614?n=1>。
2015 年 osu 修复了 Wine 里 OpenGL 引擎不能正常运行的问题，在 Linux
上运行 osu 已经非常简单了。

# 安装 osu

在 AUR 上已经有人打好了 osu 的 PKGBUILD 脚本。
安装好 AUR Helper，然后直接安装即可。

```bash
paru -S osu
```

这个 PKGBUILD 本质上就是帮你把 wine 装好，把 osu 客户端放到
`$XDG_DATA_HOME` 下，然后把变量写入启动脚本。
你可以执行 `cat /usr/bin/osu-stable` 这个脚本看看他具体做了什么。

安装好之后不要直接启动，你需要先安装好 `.Net Framework 4.0`，
不要用默认提供的 `mono`。

```bash
winetricks dotnet40
```

他在启动的时候会问你要不要装 Gecko，也不要安装，直接取消即可。

> 为什么 osu stable 和 osu lazer 不能加入到官方源呢？
>
> 这里有个商标问题。osu 的软件本体是用 MIT License 开源的，但 osu 这个标志则是有商标保护的，
> 而 Arch Linux 不被授权分发。
>
> Peppy 的回复： https://github.com/ppy/osu/issues/3723#issuecomment-439010666

# 安装驱动

想要 wine 正常运行游戏还需要安装两个驱动。
在安装驱动前你需要先启用 `multilib` 这个源：
<https://wiki.archlinux.org/title/Official_repositories#multilib>

## 声卡

如果你使用的是
PulseAudio，安装 `lib32-libpulse` 即可。

如果你用的是 Pipewire，安装 `lib32-libpulse` 和 `pipewire-pulse`。

## 显卡

如果你用的 Nvidia 显卡，你需要安装 `lib32-nvidia-utils`。
如果用的是 Intel/AMD 显卡，则需要安装 `lib32-mesa`。

# 内核更换

更换内核不是必要的，但我非常推荐你换成 `linux-zen` 内核来玩游戏。
我自己在对比体验后，能明显感觉 `linux-zen` 内核玩游戏相对于 Arch Linux
的原版内核来说要更加顺滑。相对于 Windows 来说就是“更加顺滑++++”。

```bash
sudo pacman -S linux-zen linux-zen-headers
```

如果你是 grub 用户，你需要重新生成一下 grub 的 config。
其他引导程序的用户需要自己查询一下 wiki。

```bash
sudo grub-mkconfig -o /boot/grub/grub.cfg
```

然后重启，选择 Arch Linux (More option)，然后选择 linux-zen 内核的 Arch Linux 启动即可。

重启后打开一个终端，输入 `uname -r` 查询内核 release 信息，是 zen 就对了。

## 显卡模块加载

linux-zen 内核默认用 `nouveau` 来驱动显卡，如果你用的官方驱动就会发现显卡输出无了。

对于官方驱动，你需要编译一份 Nvidia 的内核模块。
dkms 可以帮我们自动化这个工作，以后每一次 nvidia 驱动更新，或者系统内核更新，dkms
都会帮我们自动编译加载。

```bash
sudo pacman -S nvidia-dkms
```

# 数位板驱动

你现在应该已经能正常玩了，但如果你是个老 osu 板子玩家，你可能还需要调整映射。
在 AUR 里已经有人打包好了 `OpenTabletDriver` 的驱动，用 AUR Helper 下载即可。

```bash
paru -S opentabletdriver
```

然后你需要把内核里的数位板驱动关掉。因为 OpenTabletDriver 会尝试用你调整过的
小映射来映射准星位置，而内核则一直用全板映射来映射你的准星，于是就会导致准星
位置不断修正。说人话就是这两个驱动会让你准星疯狂左右横跳。

```bash
echo "blacklist wacom" | sudo tee -a /etc/modprobe.d/blacklist.conf
sudo rmmod wacom
```

第一条命令把 wacom 驱动加入黑名单，以后内核再也不会加载这个模块。
第二条命令把当前加载的驱动模块删除。

# 音频延迟

至少我自己是没听出来，但你也可以参考 The Poon 的博客，把音频延迟降低到极致。

<https://blog.thepoon.fr/osuLinuxAudioLatency/>

The Poon 里面说的他打好 patch 的 wine 你不需要自己编译，也不需要用他的源。
在 Arch Linux CN 已经有编译好的版本，加载 Arch Linux CN 源然后下载安装即可。

## CN 源配置

参考文档： <https://www.archlinuxcn.org/archlinux-cn-repo-and-mirror/>

```bash
sudo echo -e "[archlinuxcn]\nServer = https://repo.archlinuxcn.org/$arch" >> /etc/pacman.conf
sudo pacman -Syu
sudo pacman -S archlinuxcn-keyring
```

Arch Linux CN 也有很多国内镜像源，如果你有网络业障可以在这个 Repo 里找
对你网络友好的镜像： <https://github.com/archlinuxcn/mirrorlist-repo>

## 下载 The Poon 的 wine

下载 wine-osu 这个包：

```bash
sudo pacman -S wine-osu
```

然后编辑一下 `/usr/bin/osu-stable` 这个文件：

```bash
sudo vim /usr/bin/osu-stable
```

首先我们要更换 wine 的可执行位置，这里我们把 `wine-osu` 的可执行文件存放目录
放在 `$PATH` 变量的开头：

```bash
# file: /usr/bin/osu-stable

export PATH=/opt/wine-osu/bin:$PATH
```

然后就是加入一些 The Poon 在文中提到的变量：

```bash
# file: /usr/bin/osu-stable

export STAGING_AUDIO_DURATION=5000
```

至于加什么变量，值怎么调就自己参照 The Poon 的博客来就好了。

还有一件事，`/usr/bin/osu-stable` 这个文件会在下次更新 osu 的时候
被覆写，建议调好变量之后做好备份，在下次更新之后 cp 覆盖回去就行了。

# 文件访问

OSU 的游戏目录映射到了 `$XDG_DATA_HOME/osu-stable` 下，一般如果没有
特意设置这个变量，都会存放在 `~/.local/share/osu-stable` 下。
下载好的皮肤和歌直接移动进去就行了。

理论上来讲安装 osu 的时候，他会复制一份 `osu.desktop` 文件到
`/usr/share/applications/` 目录下，里面写好了各类 osu 文件的 mime type。
对于常见的 DE 直接双击文件就好了，终端用户可以用 `xdg-open` 这个命令。

但如果你的系统不能正确识别 mime type ，皮肤解压就会有点麻烦。
你可以把 `.osk` 文件复制到 `~/.local/share/osu-stable/Skins` 目录下
之后，进入游戏按 `Ctrl-o` 打开设置，在设置最顶上有个打开游戏目录的选项，
点击之后就能使用一个 Windows 的文件管理器，在里面双击你的 `.osk` 文件，
就能正确识别 mime type 并调用 osu 来解压了。

歌曲的话不用解压，每次导入新歌在游戏内摁 `F5` 就行了。

你可以参考 [wiki](https://wiki.archlinux.org/title/XDG_MIME_Applications)
来试着修复自己的 mime type。

> See you next time~
