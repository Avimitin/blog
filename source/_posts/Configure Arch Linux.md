---
title: 如何配置 Arch Linux
date: 2021-02-21 17:28
categories:
- [system, linux]
tags:
- arch
- linux
thumbnail: https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20210221172745.png
tnalt: "my desktop screenshot"
description: "I will show you how to configure the raw arch linux to a beatiful and easy for work desktop environment"
---

<!-- vim-markdown-toc GFM -->

* [Network](#network)
* [Update](#update)
* [安装必要的软件](#安装必要的软件)
* [添加用户](#添加用户)
* [在 VMWare 上装 Arch 的设置（可选）](#在-vmware-上装-arch-的设置可选)
  * [添加驱动](#添加驱动)
  * [安装](#安装)
* [安装桌面环境](#安装桌面环境)
  * [Install Display manager](#install-display-manager)
  * [自启动](#自启动)
  * [HiDPI](#hidpi)
* [分辨率](#分辨率)
* [装点别的](#装点别的)
  * [输入法](#输入法)
  * [Network manager](#network-manager)
  * [字体](#字体)
  * [终端](#终端)
  * [壁纸](#壁纸)
  * [通知系统](#通知系统)
* [主题](#主题)
* [常见错误](#常见错误)

<!-- vim-markdown-toc -->

## Network

如果你有跟着我上一篇文章，安装了 `NetworkManager`，此时你
可以输入命令 `nmtui` 来打开一个终端网络管理器来管理网络。

```console
#启动网络管理器的 daemon
systemctl start NetworkManager.service
#设置为开机自启动
systemctl enable NetworkManager.service

#启动终端管理器
nmtui
```

如果没有安装，那你可能需要把 U盘插上，mount 好分区之后
重新 chroot，并安装需要的网络管理器。

## Update

然后执行命令更新：

```bash
pacman -Syu
```

## 安装必要的软件

```bash
pacman -S man base-devel
```

## 添加用户

创建一个叫做 TOM 的用户（把 TOM 换成你的名字）：

```bash
useradd -m -G wheel TOM
```

修改 TOM 的密码：

```bash
passwd TOM
```

修改 `sudofile`， 没有 vi 就装一个：

```bash
visudo
```

找到 `Uncomment to allow members of group wheel...` 这一行，把井号去掉：

```bash
%wheel ALL=(ALL) ALL
```

保存文件退出之后，就可以输入 `exit` 退出当前用户，登录刚加入的用户了。

## 在 VMWare 上装 Arch 的设置（可选）

如果不是用 VMware 装的虚拟机 arch 可以跳过这段。

首先建议你看一遍 wiki：
[VMware/Install Arch Linux as guest](https://wiki.archlinux.org/index.php/VMware/Install_Arch_Linux_as_a_guest) 
和 
[VMware Tools for Linux Guests](https://www.vmware.com/support/ws5/doc/ws_newguest_tools_linux.html#wp1118025)

### 添加驱动

你需要打开 `/etc/mkinitcpio.conf` 文件，然后找到 `MODULES` 这一段，往里添加模块名：

```text
/etc/mkinitcpio.conf
--------------------
...
MODULES=(vmw_balloon vmw_pvscsi vmw_vmci vmwgfx vmxnet3 vmblock vsock vmnet vmmon)
...
```

然后点击上方的 `虚拟机 -> 安装 VMware Tools` 来插入 CD 盘。

### 安装

首先先给启动脚本创建文件夹：

```bash
for x in {0..6}; do mkdir -p /etc/init.d/rc${x}.d; done
```

然后挂载光盘：

```bash
mkdir /mnt/cdrom
mount /dev/cdrom /mnt/cdrom
```

把安装文件复制出来：

```bash
cd tmp
cp /mnt/cdrom/VMwareTools.tar.gz ./
tar -zxf ./VMWareTools.tar.gz
```

然后进入解压出来的 `vmware-tools-distrib` 文件夹启动 perl 脚本：

```bash
cd vmware-tools-distrib
perl ./vmware-install.pl
```

重启，登录并启动 VMware tools

```bash
/etc/init.d/rc6.d/K99vmware-tools start
```

然后把 VMware tools 链接到 systemd 自启动文件中：

```bash
vim /etc/systemd/system/vmwaretools.service
#-----------------edit-----------------------
[Unit]
Description=VMWare Tools daemon

[Service]
ExecStart=/etc/init.d/vmware-tools start
ExecStop=/etc/init.d/vmware-tools stop
PIDFile=/var/lock/subsys/vmware
TimeoutSec=0
RemainAfterExit=yes
 
[Install]
WantedBy=multi-user.target
#------------------end-----------------------
systemctl enable vmwaretools.service
```

## 安装桌面环境

我目前先用 X 不用 Wayland：

```bash
sudo pacman -S xorg
```

然后寻找适合你显卡的驱动：

```bash
sudo pacman -S xf86-video
```

然后选择什么桌面就自己挑了，KDE，GNOME，i3WM，或者 Deepin，选自己喜欢的装。

### Install Display manager

Display manager，也可以称作登录管理器，是一个图形用户界面，帮助你登录进系统并启动桌面。

```bash
sudo pacman -S light-dm
# lightdm-gtk-greeter  是默认的皮肤，你也可以选择 deepin
sudo pacman -S lightdm-gtk-greeter
```

配置一下

```bash
sudo vim /etc/lightdm/lightdm.conf
```

执行退出。然后启用开机自启

```bash
sudo systemctl enable lightdm
```

装好窗口管理的  session 之后就可以直接运行启动了：

```bash
sudo systemctl start lightdm
```

启动之后选择自己需要的桌面，然后登录就能进入桌面了。

### 自启动

lightdm 启动的时候会执行 `~/.xprofile` , `~/.xsession`, `~/.Xresources`, 可以把脚本添加进去，
启动时执行。启动脚本如何写我会在后面的软件章写。

### HiDPI

如果需要设置 dpi，在 `~/.Xresources` 里写入：

```text
Xft.dpi = 192
```

150% 放大填 144, 200% 放大填 192。

## 分辨率

一般来说应该能正常识别分辨率的，如果分辨率不对劲你可以输入 `xrandr` 来检查一下输出配置：

```bash
xrandr
###############
#   output    #
###############
Screen 0: minimum 1 x 1, current 1920 x 1080, maximum 16384 x 16384
Virtual1 connected primary 1920x1080+0+0 (normal left inverted right x axis y axis) 0mm x 0mm
   1920x1080     60.00*+
   2560x1600     59.99
   1920x1440     60.00
   1856x1392     60.00
   1792x1344     60.00
   1920x1200     59.88
   1600x1200     60.00
   1680x1050     59.95
   1400x1050     59.98
   1280x1024     60.02
   1440x900      59.89
   1280x960      60.00
   1360x768      60.02
   1280x800      59.81
   1152x864      75.00
   1280x768      59.87
   1024x768      60.00
   800x600       60.32
   640x480       59.94
Virtual2 disconnected (normal left inverted right x axis y axis)
```

如果分辨率不对，你可以输入下面的命令来切换：

```bash
xrandr --output Virtual1 --mode 1920x1080
```

如果列出来的分辨率也不对劲，你可以参考下面的方法来创建脚本开机修改分辨率：

```bash
#!/bin/bash
xrandr --newmode "1920x1080_60.00"  173.00  1920 2048 2248 2576  1080 1083 1088 1120 -hsync +vsync
xrandr --addmode Virtual1 1920x1080_60.00
xrandr --output Virtual1 --mode 1920x1080_60.00
```

其中，首段创建配置文件的 newmode 后的参数是通过下面这个命令得到的，把他换成你需要的分辨率，得到的配置参数可以防止错误配置导致的频闪：

```bash
cvt -r 1920 1080
```

可以把上面的脚本写入 `~/.xprofile` 启动时自动执行。

> 更多配置方法参阅：https://wiki.archlinux.org/index.php/Xrandr#Configuration

## 装点别的

**如果你不认识下面的软件，我很希望你能先去 Arch Wiki 和 GitHub 查询过之后再来安装，毕竟 Arch 就是一个完全让你自定义的系统，不要照搬**

### 输入法

安装： `yay -S fcitx fcitx-rime fcitx-configtool`

在 `~/.xprofile` 中写入:

```bash
#!/bin/sh
export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XMODIFIERS="@im=fcitx"
```

同时也可以再加一句 `fcitx &` 来开机自启动 fcitx：

```bash
#!/bin/sh

# fcitx settings
export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XMODIFIERS="@im=fcitx"

fcitx &
```

用 `fcitx-configtool` 来启动 fcitx 管理界面。然后把 rime 勾选上, 第一个放 `Keyboard-English`， 
rime 放在第二位。

Fcitx 还需要一些额外的 lib 来辅助工作，但是是可选的：

```bash
yay -S nuspell hunspell hspell aspell libvoiko
```

### Network manager

Network Manager 可以帮你更轻松的管理网络设置, 以及附带的 GUI 也更加友善。

执行 `yay -S networkmanager networkmanager-applet` 安装。

然后在 `~/.xprofile` 中写入 `nm-applet &` 来开机自启动

### 字体

推荐安装：

- [nerd font](https://github.com/ryanoasis/nerd-fonts/releases)

我自己用的是 `JetbrainsMono Nerd Font` DWM 则使用的 `Ubuntu Mono Nerd Font`

- 英文字体设置：

```text
/etc/locale.conf
----------------
LANG=en_US.UTF-8
LC_ADDRESS=en_US.UTF-8
LC_IDENTIFICATION=en_US.UTF-8
LC_MEASUREMENT=en_US.UTF-8
LC_MONETARY=en_US.UTF-8
LC_NAME=en_US.UTF-8
LC_NUMERIC=en_US.UTF-8
LC_PAPER=en_US.UTF-8
LC_TELEPHONE=en_US.UTF-8
LC_TIME=en_US.UTF-8
```

- Emoji 字体：

可以在下述字体中选择一个安装。

```bash
yay -Ss ttf-linux-libertine ttf-inconsolata ttf-joypixels ttf-twemoji-color noto-fonts-emoji ttf-liberation ttf-droid
```

- 中文字体

你可以查看 [这个页面](https://wiki.archlinux.org/title/Localization/Chinese)
来选择一些中文字体。个人推荐 Adobe 的思源黑体。

### 终端

我用的 simple terminal: [Avimitin/st](https://github.com/Avimitin/st)，
参阅：[Simple Terminal 的配置](https://avimitin.com/system/simpleterminal.html)。

### 壁纸

```bash
yay -S feh
```

Some tips and tricks:

```bash
# move all your image to a library like: ~/Pictures/WallPapers
feh --recursive --randomize --bg-fill ~/Pictures/WallPapers/
# this command will show all the picture as background randomly
```

### 通知系统

```bash
yay -S notification-daemon
```

## 主题

可以在 gnome-look.org 上下载喜欢的 GTK 主题和图标，然后在本地
创建文件夹：

```console
mkdir -p ~/.local/share/themes
mkdir -p ~/.local/share/icons
```

把下载好的主题放进 themes 文件夹里，图标放进 icons 文件夹里。
然后安装 `lxapperance` 更换。

## 常见错误

- `Gtk-Message: Failed to load module "colorreload-gtk-module"`

是正常的，但是假如你看着不舒服，可以安装 `kde-gtk-config` 包，
里面包含必要的 Modules. 
