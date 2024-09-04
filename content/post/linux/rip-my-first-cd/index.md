+++
title = '在 Arch Linux 上用 Exact Audio Copy 扒碟'
date = '2024-09-04'
tag = ['linux', 'rip']
author = 'sh1marin'
+++
刺团的几张碟在 nyaa.si 上只有 「天使动漫」的 TSDM 资源。
虽然有资源，但是是用的 RAR 打包的，看着就很没有下载欲望。
正好其实这几张碟我都有买，我决定学一下扒碟技术，重新提交一份干净一点的资源。

![唉，少女乐队](./images/cd.jpg)

主要参考的两篇文章：

* Linux setup 看的这篇： <https://sharzy.in/posts/2024-02-23-eac-on-wine/>
* 抓碟看的这篇： <https://zexwoo.blog/posts/tutorials/eac-ripping/>

这篇主要补充一些细节：

1. 想要检查 /etc/fstab 的内容正确性，可以用 `findmnt --verify`
2. Arch 上跑 `pacman -S wine` 即可，虽然说 `wine-mono` 装上可以解决 .Net 依赖问题，但是我在跑安装器的时候总是不认。
3. 在用 wine 执行安装器之前，用 winetricks 装上 .Net 2.0，这是 EAC 必要依赖。
4. 因为我扒的日语碟，还是不出所料遇到豆腐字了，用 `winetricks corefonts`, `winetricks allfonts` `winetricks fakechinese` 把所有字体装上。然后在跑 EAC 的时候加上 `LC_ALL` 环境变量。比如 `LC_ALL=zh_CN.UTF-8 wine EAC.exe`。
5. wine 的 C 盘放在 `~/.wine/drive_c` 下面，但软件内弹出菜单没有显示带 `.` 的文件的选项，可能会在选择 flac 程序的时候卡住。
这个时候不该继续在家目录找，而是要继续往上翻到根目录，使用那里映射的 C 盘。
6. 如果跟着教程说的换 flac 版本了，下载的 flac-win.zip 文件夹里有 64 和 32 位两个版本，实际测试 64 位是完全可用的。
7. Player.exe 安装器可能有问题，装好之后不在路径里，需要指明完整路径 `wine ~/.wine/drive_c/Program Files (x86)/Player/Player.exe`
（不要忘记 `LC_ALL`）
8. wayland 下面的粘贴板还是有问题，我还没找到解决方案，暂时还是先手输入。
9. 抓碟教程里的配置文件在第一次 import 就好，不需要每次重复导入（会覆盖）。
10. Amazon 的专辑图片清晰度足够高，可以直接去 amazon.co.jp 搜。
11. 想要把专辑封面放进 flac 文件里可以用这个命令：

```bash
ffmpeg -i example.flac -i cover.jpg -map 0:a -map 1 -codec copy -metadata:s:v title="Album cover" -metadata:s:v comment="Cover (front)" -disposition:v attached_pic output.flac
```

还需要往多个 flac 添加同个封面图的话，我是这么做的：

```bash
# 我扒碟的输出路径
RIP_OUTPUT="$HOME/Rip/CD/トゲナシトゲアリ - 視界の隅 朽ちる音 (2024) {UNIVERSAL MUSIC LLC, UMCK-5754, CD} [FLAC]"
STORAGE="/mnt/hdd/Music/トゲナシトゲアリ - 視界の隅 朽ちる音 (2024) {UNIVERSAL MUSIC LLC, UMCK-5754, CD} [FLAC]"

pushd "$RIP_OUTPUT"
for file in $(find . -name '*.flac'); do
    ffmpeg -i "$file" -i cover.jpg -map 0:a -map 1 -codec copy -metadata:s:v title="Album cover" -metadata:s:v comment="Cover (front)" -disposition:v attached_pic "$STORAGE/$file"
done

cp *.m3u8 "$STORAGE/$file"
cp *.Cue "$STORAGE/$file"
```
