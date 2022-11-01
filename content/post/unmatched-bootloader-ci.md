+++
title = '给 Unmatched Bootloader 写个自动化！'
date = '2022-11-01'
tag = ['unmatched', 'github action', 'binfmt']
author = 'sh1marin'
+++

Unmatched 是 SiFive 发行的一款 RISC-V 的板子，出厂自带 Ubuntu 系统。
办公室的同事写了几个脚本，给这块板子弄了个非常清真，自主可控的
Arch Linux 固件。但是那几个脚本有些问题，并不能无脑构建出固件。
同时也必须需要一个 riscv64 qemu system 或者硬件来构建。
我对这个脚本稍微做了一些修改，并加上了交叉编译支持，于是这几个脚本
可以一键无脑在 x86_64 Arch Linux 环境里出 Unmatched 固件了。

那么问题来了，既然有脚本了，理论上我们是不是可以用 GitHub Action 来自动化
这个过程？每次 push，甚至每周，或者每过几天就自动 build 一次，每次我们要用
的时候只需要下载固件就可以了，这多方便啊。于是我接下了这个任务（大坑），
试着给我们的仓库加上 CI 脚本。

## #1 Dubious ownership in repository

年幼无知的我以为，在 Action 里跑个 Arch Linux 的 docker image，加几句脚本调用，
剩下的就交给时间就可以了。于是先出了一版简单的 ci workflow：

```yaml
name: bootloader-builder
on:
  push:
    branches:
      - master
      - ci

jobs:
  bootloader-build-job:
    runs-on: ubuntu-latest
    container: archlinux:latest
    steps:
      - run: pacman -Syu --noconfirm git
      - uses: actions/checkout@v3
      - name: Install deps
        run: |
          git submodule update --init
          ./install-deps.sh
      - name: Build-opensbi.sh
        run: ./build-opensbi.sh cross-compile
      - name: Build U-Boot
        run: ./build-u-boot.sh cross-compile
      - name: Make image
        run: ./mk-image.sh cross-compile
```

这个 action 很简单，设置了仅在 `master`, `ci` 两个分支做变动的时候触发脚本，
接着在 ubuntu 的宿主机起一个 archlinux 的 docker container，并依次执行固件
的构建脚本。
因为我们的仓库 vendor 了 u-boot 和 opensbi 这两个仓库，所以安装好
依赖之后还需要记得 `git submodule init & update` 来拉取上游仓库。

因为我们是在 Arch Linux 镜像里，而不是在 Ubuntu 这个 Host 主机里
执行 checkout action，而默认 Arch Linux 镜像是不带 git 的，要记得在 checkout 之前装好 git，
不然 checkout action 就会用 curl 来下载源码的 tar 文件，这样是不会带上 .git 目录的，
然后就会不得不重新 init 仓库了。

好，push 一下看看 GitHub Action 怎么说。

```text
...
fatal: detected dubious ownership in repository at ......
...
```

嗯？这是什么问题？稍微查了一下发现原来 git 在 init 的时候会探测执行 git 指令的
用户和这个仓库的用户是否一致，这样可以减少一些在共享环境里的风险。
而 CI 的用户环境和本地的用户环境不一样，自然会有这个报错。

> 关于这个探测的详细可以看这条 commit:
> <https://github.com/git/git/commit/8959555cee7ec045958f9b6dd62e541affb7e7d9>

当然这个行为可以手动取消，在 git config 里设置一下 `safe.directory` 这个选项即可：

```
git config --global --add safe.directory /path/to/srcdir
```

## #2 cannot find unused loop device

接下来依旧进展不顺，CI 执行到了打包镜像的环节，但设置 loop device 的时候 losetup 报错
`losetup: cannot find an unused loop device`。思索了一下，感觉可能是 docker container
屏蔽了宿主机的设备读取。查了一下 `docker-run` 的 man page 发现了这么一条参数：

```text
--privileged [true|false]
  Give extended privileges to this container. A "privileged" container is given access to all devices.

When the operator executes docker run --privileged, Docker will enable access to all devices on the host as well
as set some configuration in AppArmor to allow the container nearly all the same access to the host as processes
running outside of a container on the host.
```

好，启用！修改一下 CI 配置文件的 container 部分，加上这个选项：

```diff
-    container: archlinux:latest
+    container:
+      image: archlinux:latest
+      options: "--privileged"
```

## #3 snapd is shit

现在能在 docker 内读写设备了，push 一版看看还有啥问题。

```text
mke2fs
The file /dev/loop0p3 does not exist and no size was specified.
mount: special device /dev/loop0p3 does not exist
```

嗯？CI 的 host 居然不止一个 loop 设备吗。在脚本里加上 `losetup -a`，发现原来
给 Ubuntu 的 Snapd 吃了三个 loop device，而手写的 `/dev/loop0p3` 指定错设备了。

> 吐槽：怎么会有 snapd 这种东西在 CI 环境里

```text
/dev/loop0: [2049]:71364 (/var/lib/snapd/snaps/core20_1623.snap)
/dev/loop1: [2049]:71365 (/var/lib/snapd/snaps/snapd_17029.snap)
/dev/loop2: [2049]:71363 (/var/lib/snapd/snaps/lxd_22753.snap)
/dev/loop3: [2049]:1586605 (...)
```

问题不大，给 `losetup` 加上 `--show` 参数之后，他会往 stdout 输出当前使用的
loop device 的名字，用一个变量来存这个名字就行：

```diff
-losetup -f -P $IMAGE_FILE
+LODEV=$(losetup -f --show -P $IMAGE_FILE)
-mkfs.ext4 /dev/loop0p3
+mkfs.ext4 "${LODEV}p3"
 mkdir rootfs
-mount /dev/loop0p3 rootfs/
+mount "${LODEV}p3" rootfs/
```

再推一版到 CI 上，格式化的参数终于是对了，但还是遇到了这一小节开头的问题。

```text
The file /dev/loop3p3 does not exist and no size was specified.
```

## #4 partition not found

既然 `losetup -a` 可以看到 `/dev/loop3` 这个设备，那么问题就应该出在分区上，
为什么 `/dev/loop3p3`，`/dev/loop3` 的第三个分区挂不上呢？

在研究这个问题之前先提前说明一下为什么会有三个分区。这个固件脚本会在本地
truncate 一个新的文件，分好三个区，前两个区一个写 spl 另一个写 uboot，最后
一个分区写系统镜像。这就是为什么上面在做 mkfs 和 mount 的操作。

`losetup -f -P $IMAGE_FILE` 将这个新 truncate 的分区连接到 loop device 上，
连接好之后我们就应该可以操作 `/dev/loopNp3` 这个分区，如果访问不到的话，我想
只有可能是分区有问题。往上翻了翻日志，发现分区的时候 `sgdisk` 发了一条 warning：

```text
Warning: The kernel is still using the old partition table.
The new table will be used at the next reboot or after you
run partprobe(8) or kpartx(8)
```

难道是因为命令执行得太快了，系统没有分区信息？装上 parted 这个包，
尝试用命令 `partprobe` 通知系统关于分区的变动，然后再执行了一下
`partprobe -s` 命令查看分区，居然在 `/dev/loop3` 这个设备上看不到任何
分区。

这下头皮发麻了，为啥呢，为啥没有分区信息呢？百思不得其解的我请来了我伟大的鴨鴨走搜索引擎神，
发现了这么一条 issue: <https://github.com/moby/moby/issues/27886>。终于理解了 docker 的申必奥义。

简单的来讲，当你加上 `--privileged` 参数的时候，docker 会将 `/dev` 下所有的文件复制进容器里，
但容器起了之后，任何对 `/dev` 做的修改都不会被更新。

在我们的例子里，起容器的时候 docker 将 `/dev/loop3` 这个文件复制进了容器。在容器内我们执行了
`losetup -D` 并连接上我们的分区文件，虽然宿主机（Ubuntu）里设备文件更新了，但 docker 并不会
把这个新文件复制到容器里，docker 只复制一次，然后它就摆了。

那咋办呢？往下翻 issue，tonyfahrion 给我们提供了[答案](https://github.com/moby/moby/issues/27886#issuecomment-417074845):

```bash
LOOPDEV=$(losetup --find --show --partscan ${IMAGE_FILE})

PARTITIONS=$(lsblk --raw --output "MAJ:MIN" --noheadings ${LOOPDEV} | tail -n +2)
COUNTER=1
for i in $PARTITIONS; do
    MAJ=$(echo $i | cut -d: -f1)
    MIN=$(echo $i | cut -d: -f2)
    if [ ! -e "${LOOPDEV}p${COUNTER}" ]; then mknod ${LOOPDEV}p${COUNTER} b $MAJ $MIN; fi
    COUNTER=$((COUNTER + 1))
done
```

他从 lsblk 读取指定的 loop 设备文件的信息，然后用 mknod 手动创建这些设备文件。
非常感谢他，我马上抄了一份放进镜像打包文件里，加上一个 docker 环境的判断：

```bash
if [[ -f /.dockerenv ]]; then
    ...
fi
```

docker 必会在容器的根目录下放一个 .dockerenv 文件，只要这个文件存在，我们就可以
假设这个脚本在 docker 里运行。

## exec format error

修 loop device 找不到的问题已经把我累得半死不活了，抱着 “这回应该可以了吧” 的 kimoji，
我又 push 了一版到 Action 上。很开心，mount 能挂上了，pacstrap 也跑起来了，这下完事大吉了，哈哈。

然而 binfmt 又给了我一重锤：

```text
call to execv failed (Exec format error)
error: command failed to execute correctly
```

关于 binfmt 的详细介绍可以看[维基](https://en.wikipedia.org/wiki/Binfmt_misc)，简单来说
它是内核用来判断如何执行一个可执行文件的一串格式符号，识别架构是 binfmt 的工作
。出现 `(Exec format error)` 说明 kernel 不能正确识别可执行文件的二进制代码。

那么为什么不能正确识别呢？首先先看日志，这段报错出在 pacstrap 的安装包和 post-transaction hook
两个阶段，而这两个阶段都是在 RISC-V 环境里运行的，大致可以推断用了 x86_64 环境跑 riscv64 的软件。
于是和同事商量了一下之后，猜测可能是 binfmt 没有挂上。

因为官方仓库的 qemu-user-static-binfmt 没有开 P flag 不能用，需要用同事特别打包过的
devtools-riscv64 包来提供可用的 binfmt，这个包官方库里没有，archlinuxcn 有但已经过时，
我们只好手动从 AUR 上拉取下来自己手动 build。

手动打 AUR 的包很简单，但有几个小问题需要注意

1. 关于打包用户

首先 Arch Linux 的打包工具 makepkg 不能用 root 执行，我也不想创建新用户,
因此这里用默认自带的 nobody 账户来打包，而 `makepkg -si` 包含了 sync 和 install 两个需要
root 权限的操作，会默认调用 sudo，GitHub Action 是没有用来交互的 shell 的，
那么这里就要记住给 `/etc/sudoers` 配置免密码。

2. 关于文件权限

跑 ci 的时候当前目录下所有文件都是归属于 root 用户的，要记得给 nobody 用户当前目录的写权限，
不然会有 permission denied 的问题。

3. 去掉确认步骤

还是因为 Action 没有交互 Shell，makepkg -si 包括了 install 这个需要确认的步骤，所以记得
加上 `--noconfirm` 参数。

完整的打包流程如下：

```bash
# 让任何人都能免密码使用 sudo
echo "ALL ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers
# 克隆打包脚本
git clone https://aur.archlinux.org/devtools-riscv64.git
cd devtools-riscv64
# 配置目录权限
chown -R nobody $PWD
# 切换到 nobody 账户来执行 makepkg -si
sudo -u nobody makepkg -si --noconfirm
```

## docker containers use host OS kernel

有 binfmt 了，有 qemu-user-static 了，这下能正常调用 exec 了吧？
把修好的 ci 脚本 push 上去，发现问题依旧。抓耳挠腮也想不明白还有哪里有问题。

又和同事请教了一番，发现我思路对了，但忘记了关键的一环：我在 docker 内。

binfmt 是内核在做的事情，而 docker 镜像是共用宿主机内核的。我在镜像内
配置 binfmt 并不会影响 Ubuntu 内核。所以得换个解决方案：先在宿主机配好 binfmt，
然后再起 docker。

给宿主机配 binfmt 已经有 actions 可以复用了：<https://github.com/docker/setup-qemu-action>。
但是因为我给所有的 step 配置了: `container: archlinux:latest`，
直接往 workflow 里直接加一个新的 step 只是在 arch 的镜像里再跑一个 docker 罢了，
遍历 GitHub Action 的文档，也没有什么 `pre-container` 之类的 hook。

那咋办呢，思索了一下，干脆不破不立，直接把整个 container 的配置删了，我不需要 GitHub
帮我传递命令了，我自己手动来。最后的 CI 配置变成了这样：

```yaml
...
    runs-on: ubuntu-latest
    steps:
      - name: Setup rv64 binfmt
        uses: docker/setup-qemu-action@v2
      - uses: actions/checkout@v3
      - name: Build Image
        run: ./mkin-docker
      - name: Upload Image
        uses: actions/upload-artifact@v3
        with:
          name: unmatched-bootloader-image
          path: /artifact/image-*.raw
```

新的配置去掉了之前所有的构建步骤，去掉了 container 配置，设置上 rv64 的 binfmt
并下载源码，然后执行源码里的 mkin-docker 脚本。

这个脚本也很简单，就是把删掉的构建步骤移动到 docker run 里，把本地源码挂载进去，
挂载一个构建好的镜像目录，然后用 archlinux 镜像调用 bash 执行单引号里的命令。
单引号里的命令就是正常写脚本那样写就行了，-c 选项会把所有文本吃进去当作命令执行：

```bash
docker run \
  -v $PWD:/srcdir \
  -v $PWD/images:/artifact \
  --privileged \
  archlinux:latest \
  bash -lc '
  ...
'
```

And finally，我终于看到了无报错的完整日志和可用的固件了，可喜可贺可喜可贺。

---

- 一些小吐槽

本来以为很轻松的 Action 构建，结果花了两天来从坑里爬出来，真心很累。
配置 GitHub Action 也很蛋疼，Debug 只能 push 上去，等构建，然后看日志，
改，再推，再等......同时你的邮箱会被塞满各种 build fail 的邮件。

这期间我的心态发生了一点变化：不觉得 yaml 工程师很酷吗，
非常符合我对程序员的想象，哈哈。
