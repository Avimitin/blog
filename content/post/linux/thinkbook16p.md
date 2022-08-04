+++
title = '记我在 ThinkBook 16P 遇到的一些坑'
date = '2022-08-04'
tag = ['lenovo', 'thinkbook']
author = 'sh1marin'
+++

入职的时候公司给配了一台 ThinkBook 16p，i5-12500H 无显卡版本的。
然后装好 Arch Linux 后遇到了大量的坑。

首先是显示。联想的机器因为一些莫名其妙的原因，在 Linux 上滑动鼠标会有
画面撕裂的 BUG。

Arch Linux 在 5.18.15 内核里引入了一个修复：
<https://github.com/archlinux/linux/commit/b7c36a998b8027386b5ea43ecff46057c83655cb>，
直接 Syu 就可。

关于这个 BUG 的报告： <https://gitlab.freedesktop.org/drm/intel/-/issues/5440>

---

然后是 WiFi。联想贪便宜整了个便宜的螃蟹网卡，结果内核里没有这个驱动。
用命令 `lspci -nn` 看到型号是: `Realtek Semiconductor Co., Ltd. Device [10ec:b852]`，
查了一下查到 Ubuntu 论坛里提到 <https://github.com/HRex39/rtl8852be> 这个仓库里的
开源驱动可以用。

跟着 README 的引导下载好依赖 `sudo pacman -S bc linux-headers`，然后
把仓库拷贝下来跑 `make -j8 && sudo make install && sudo modprobe 8852be`，
重启就可以啦！

---

最后是虚拟机，装了个 VirtualBox 来跑微信和腾讯会议，结果安装 Windows10
的时候安装进度卡在 20% 不动了。也没有任何日志报错。

查看了一下内核日志 `sudo dmesg`，发现内核日志里提到 `traps: Missing ENDBR`
以及 `kernel BUG at arch/x86/kernel/traps.c`，问了一下才知道原来是 Intel
在 12 代 CPU 里激进的引入了还不稳定的特性。在启动内核时加上选项 `ibt=off` 即可。

我用 GRUB 引导，在 /etc/default/grub 文件里的 `GRUB_CMDLINE_LINUX_DEFAULT` 的
值里增加一个 `"ibt=off"` 即可。
