---
title: Arch Linux Configuration Guide
date: 2021-02-21 17:28
categories:
- [system, linux]
tags:
- arch
- linux
thumbnail: https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20210221172745.png
tnalt: "my desktop screenshot"
tldr: "I will show you how to configure the raw arch linux to a beatiful and easy for work desktop environment"
---

#  Configure Arch Linux

## Network

登录 Root 账户之后，先 ping 一下随便一个网页，看看有网没，有网就不管了，我反正没网要设置一下。

先看一下网络设备：

```bash
ip link
```

应该会出现类似这样的文字：

![](http://xahlee.info/linux/i/linux_ip_link_show_output_2017.png)

如果按照图片，你就有一个叫做 `eth0` 的网络设备，编辑配置文件：

```bash
sudo vim /etc/network/20-wired.network

#-----edit-----#
[Match]
Name=eth0

[Network]
DHCP=yes
#-end of edit-#
```

然后执行下面这个命令启动并启用网络配置自启：

```bash
sudo systemctl start systemd-networkd
sudo systemctl enable systemd-networkd
```

> further reading: [systemd-networkd](https://wiki.archlinux.org/index.php/Systemd-networkd)

## Update

然后执行命令更新：

```bash
pacman -Syu
```

## Install Needed Binaries

```bash
pacman -S man base-devel
```

## Add a new user for yourself

```bash
useradd -m -G wheel TOM
```

修改 TOM 的密码

```bash
passwd TOM
```

修改 `sudofile`， 没有 vi 就装一个

```bash
visudo
```

找到 `Uncomment to allow members of group wheel...` 这一行，把井号去掉：

```bash
%wheel ALL=(ALL) ALL
```

保存文件退出之后，就可以输入 `exit` 退出当前用户，登录刚加入的用户了。

## 在 VMWare 上装 Arch

首先建议你看一遍 wiki：[VMware/Install Arch Linux as guest](https://wiki.archlinux.org/index.php/VMware/Install_Arch_Linux_as_a_guest) 和 [VMware Tools for Linux Guests](https://www.vmware.com/support/ws5/doc/ws_newguest_tools_linux.html#wp1118025)

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

## Install Desktop Environment

我目前先用 X 不用 Wayland：

```bash
sudo pacman -S xorg xorg-server
```

然后选择什么桌面就自己挑了，KDE，GNOME，i3WM，或者 Deepin，选自己喜欢的装。

### Install Display manager

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

### 安装 DWM 窗口管理

[dwm](https://dwm.suckless.org/) 是我正在尝试使用的一个窗户管理器，因为之前有用过很久的 i3wm，如果没有相应的使用经验我建议还是去用 GNOME 这种经典的用鼠标的窗口管理，对标签式管理有一定了解之后再来尝试。

### 下载

```bash
git clone https://git.suckless.org/dwm
```

### 安装

```bash
sudo make clean install
```

### 添加进 Session

```bash
# 你可能需要先创建文件夹
vim /usr/share/lightdm/sessions/dwm.desktop

# +-----edit-----+
[Desktop Entry]
Name=dwm
Comment=Log in using the dwm window manager
Exec=/usr/local/bin/dwm
TryExec=/usr/local/bin/dwm
Type=Application
# +---end of edit---+
```

可以到 dwm 官网的 patches 页面下载安装插件，也可以用已经打好补丁的版本：[Theniceboy/dwm](https://github.com/theniceboy/dwm)，进入文件夹输入命令 `sudo make clean install` 重启即可。自定义的部分在 `config.h` 里，懂一丢丢 C 语言应该就能看的懂配置文件的意思的了，（不会编程的感觉也不会看这个？）

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

> 更多配置方法参阅：https://wiki.archlinux.org/index.php/Xrandr#Configuration

## 装点别的

**如果你不认识下面的软件，我很希望你能先去 Arch Wiki 和 GitHub 查询过之后再来安装，毕竟 Arch 就是一个完全让你自定义的系统，不要照搬**

### 字体

推荐安装：

- [nerd font](https://github.com/ryanoasis/nerd-fonts/releases)
- [powerline/fonts](https://github.com/powerline/fonts)

我自己用的是 `JetbrainsMono Nerd Font`  和 `Inconsolata for Powerline` 作为 fallback。DWM 则使用的 `Ubuntu Mono derivative Powerline`

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

```bash
yay -S ttf-linux-libertine ttf-inconsolata ttf-joypixels ttf-twemoji-color noto-fonts-emoji ttf-liberation ttf-droid
```

- 中文字体

```bash
yay -S wqy-bitmapfont wqy-microhei wqy-microhei-lite wqy-zenhei adobe-source-han-mono-cn-fonts adobe-source-han-sans-cn-fonts adobe-source-han-serif-cn-fonts
```

### 终端

我用的 simple terminal: [Avimitin/st](https://github.com/Avimitin/st)，参阅：[Simple Terminal 的配置](https://avimitin.com/system/simpleterminal.html)。

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

### 半透明

```bash
yay -S picom-git
picom -b
```

### Git

```bash
sudo pacman -S git
```

### Firefox

```bash
sudo pacman -S firefox
```

### AUR

follow [Jguer/yay](https://github.com/Jguer/yay) installation

### TLP

用来管理笔记本电池

follow [linrunner/TLP](https://github.com/linrunner/TLP) | [ArchWiki/TLP](https://wiki.archlinux.org/index.php/TLP)

### SSH

```bash
#!/bin/bash
yay -S openssh
ssh-keygen -t ed25519 -C "your_email@example.com"
eval `ssh-agent -s`
ssh-add ~/.ssh/id_ed25519
```

### Lazygit

```bash
yay -S lazygit
```

### FZF

fuzzy file finder

![](https://raw.githubusercontent.com/PatrickF1/fzf.fish/main/images/directory.gif)

- CLI

```bash
# install
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install
```

- Fish

```bash
# install fd(another find)
yay -S fd
# install bat(another cat)
yay -S bat
fisher install PatrickF1/fzf.fish
```

- vim

```vim
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
```

### Fish Shell

```bash
yay -S fish

# commands below are optional
# plugin manager
curl -sL https://git.io/fisher | source && fisher install jorgebucaran/fisher
fisher install matchai/spacefish
fisher install acomagu/fish-async-prompt
```

### neovim

![image-20210221171417583](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20210221171519.png)

```bash
yay -S neovim-nightly-bin
git clone https://github.com/Avimitin/nvim.git ~/.config/nvim
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.37.2/install.sh | bash
nvm install node
node install -g neovim
pip3 install pynvim
```

我的 vim 配置文件在： [Avimitin/nvim](https://github.com/Avimitin/nvim)

### Python

```bash
yay -S python3 python-pip
python -m pip install --upgrade pip setuptools wheel
```

