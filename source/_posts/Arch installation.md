---
title: Arch Linux Installation Guide
date: 2021-02-20 18:48
categories:
- [system, linux]
tags:
- arch
- linux
thumbnail: https://archlinux.org/static/logos/archlinux-logo-dark-1200dpi.b42bd35d5916.png
tnalt: "FEAR LESS ARCH MORE"
tldr: "Notes about manually install arch linux"
---

# Arch installation

## Get arch 

Download Arch Linux from https://archlinux.org/download/
and follow guide from https://wiki.archlinux.org/index.php/Installation_guide

## Before boot (OPTIONAL)

You can press `e` to edit install configuration like:

```bash
# If you think the screen is too small.
nomodset video=800x450
```

## Setting font (OPTIONAL)

If you think the font is too small and hard to look, you can change font for 
a more comfortable look.

```console
setfont /usr/share/kbd/consolefonts/LatGrkCyr-12x12.gz
```

## Configure key (OPTIONAL)

```sh
touch keys.conf
#-------edit--------#
keycode 1 = Caps_Lock
keycode 58 = Escape
#----end of edit-----#
loadkeys keys.conf
```

## Setting WiFi

Detail: https://wiki.archlinux.org/title/Iwd#iwctl.

First of all, get your device id:

```sh
# input iwctl to get promt
$ iwctl

[iwd]# device list
```

You may found a name like `wlan0`, that's the name we will use next.

Then, to scan for networks:

```console
[iwd]# station your-device-name scan
```

You can then list all available networks:

```console
[iwd]# station your-device-name get-networks
```

Finally, to connect to a network:

```console
[iwd]# station your-device-name connect SSID
```

> If you have Chinese SSID name which you can't input you can use TAB to auto
complete.

## Set correct time

[timedatectl(1)](https://man.archlinux.org/man/timedatectl.1)

```sh
timedatectl set-ntp true
# check status
timedatectl status
# check available timezone
timedatectl list-timezones
# change timezone
timedatectl set-timezone Asia/Shanghai
```

## Partition disk

```sh
# check your disk
fdisk -l
# start fdisk
# example:
# fdisk /dev/sda
fdisk {DISK_NAME}
```

### Example Layout

#### BIOS with MBR

Detail: https://en.wikipedia.org/wiki/Partition_type

| Mount point | Partition               | Partition type | Suggested size          |
| ----------- | ----------------------- | -------------- | ----------------------- |
| `[SWAP]`    | `/dev/*swap_partition*` | Linux swap     | More than 512 MiB       |
| `/mnt`      | `/dev/*root_partition*` | Linux          | Remainder of the device |

#### UEFI with GPT

GUID: https://en.wikipedia.org/wiki/GUID_Partition_Table#Partition_type_GUIDs
EFI: https://wiki.archlinux.org/index.php/EFI_system_partition

| Mount point               | Partition                     | [Partition type]()    | Suggested size          |
| ------------------------- | ----------------------------- | --------------------- | ----------------------- |
| `/mnt/boot` or `/mnt/efi` | `/dev/*efi_system_partition*` | EFI system partition] | At least 260 MiB        |
| `[SWAP]`                  | `/dev/*swap_partition*`       | Linux swap            | More than 512 MiB       |
| `/mnt`                    | `/dev/*root_partition*`       | Linux x86-64 root (/) | Remainder of the device |

### Fdisk

Detail: https://wiki.archlinux.org/title/Fdisk

### Example for GPT

you can follow step by step

```sh
# get help message
m

# create GPT partition table
g

# ---- /mnt/boot ---- #
# create a new partition for /mnt/boot
n
# Partition number
1
# First Sector
# use default

# Last Sector
+2G
# signature can be remove, if the disk had been partition before

# --- [SWAP] --- #
# create a new partition for [SWAP]
n
# partition number
3
# First Sector
# use default

# Last Sector
+2G

# --- /mnt --- #
# create a new partition for /mnt
n
# partition number
2
# first sector
# default
# last sector
# to the end (default)

# final check 
p
# final write
w
```

## Format Partition

Once the partitions have been created, each newly created partition must be 
formatted with an appropriate 
[file system](https://wiki.archlinux.org/index.php/File_system).

The partition of `/mnt/boot` must be `FAT32` format if you are using EFI: 
> EFI system partition: 
> https://wiki.archlinux.org/title/EFI_system_partition#Format_the_partition

```sh
# DEVICE_PARTITION_NAME can get from
# fdisk and use p command to get
# example:
# mkfs.fat -F32 /dev/sda1
mkfs.fat -F32 {DEVICE_PARTITION_NAME}
```

And the root partition `/mnt` can use `ext4` format, other format See
[File systems#Create a file system](https://wiki.archlinux.org/index.php/File_systems#Create_a_file_system)
for details.

```sh
# Example:
# mkfs.ext4 /dev/sda2
mkfs.ext4 {DEVICE_PARTITION_NAME}
```

created a partition for [swap](https://wiki.archlinux.org/index.php/Swap), initialize it with [mkswap(8)](https://man.archlinux.org/man/mkswap.8), enable it with [swapon(8)](https://man.archlinux.org/man/swapon.8):

> About swap partition: https://wiki.archlinux.org/title/Swap#Swap_partition

```sh
# Example
# mkswap /dev/sda3
mkswap {DEVICE_PARTITION_NAME}
swapon {DEVIEC_PARTITION_NAME}
```

## Software manager

manage pacman source:

```sh
vim /etc/pacman.conf

# --- vim user interface --- #
# Under UseSyslog, can delete # before Color section
Color
```

```sh
vim /etc/pacman.d/mirrorlist

# --- vim user interface --- #
# you should put all the wanted source at the top
# like Chinese user can put all the China mirror at the top
# China
http://example.com
```

Also you can add TUNA mirror manually:
https://mirrors.tuna.tsinghua.edu.cn/help/archlinux/

## Mount Partition and Final Install

> All the below device name like `/dev/sda1` or sda2 was mention above, replace
them with your machine specific device name.

[Mount](https://wiki.archlinux.org/index.php/Mount) the root volume to `/mnt`. 
For example, if the root volume is `/dev/sda2`

**CARE OF ALL YOUR MOUNT ACTION**

```sh
# Check your disk name
fdisk -l
# mount the root partition to /mnt
mount /dev/sda2 /mnt
# check mount status
ls /mnt

# if you are BIOS boot user you don't need to mount the boot partition
# create a directory for boot
mkdir /mnt/boot
# mount your boot partition to /mnt/boot or any directory
mount /dev/sda1 /mnt/boot
```

and finally, time to install:

```sh
# install base,linux,linux-firmware to /mnt
pacstrap /mnt base linux linux-firmware
```

If you want any other program, add them after the pacstrap command.
For example, I will use neovim and vi next, so I also added neovim.
Also you will need a
[network tool](https://wiki.archlinux.org/title/Network_configuration#Network_management).
Here I choose the [NetworkManager](https://wiki.archlinux.org/title/NetworkManager).

```bash
pacstrap /mnt base linux linux-firmware neovim vi networkmanager
```

## Configure

### Generate an [fstab](https://wiki.archlinux.org/index.php/Fstab) file

use `-U` or `-L` to define by [UUID](https://wiki.archlinux.org/index.php/UUID) or labels, respectively:

```
genfstab -U /mnt >> /mnt/etc/fstab
```

### Get into system installed

```sh
arch-chroot /mnt
```

Now you are in your system

### Create a symbol link to system time

```sh
# like ln -sf /usr/share/zoneinfo/Asian/Shanhai /etc/localtime
ln -sf /usr/share/zoneinfo/Region/City /etc/localtime
```

### Sync system time:

```sh
hwclock --systohc
```

### Manage locale

```sh
# edit the locale file in root
vim /etc/locale.gen

# find en_US.UTF-8 and remove # before it
en_US.UTF-8 UTF-8

# also if you are Chinese you will need Chinese locale support:
# find zh_CN
zh_CN.GB18030 GB18030
zh_CN.GBK GBK
zh_CN.UTF-8 UTF-8
zh_CN GB2312

# generate locale
locale-gen
# should get output: 
#     en_US.UTF-8... done
#     zh_CN.GB18030 GB18030... done
#     ......
#     Generation complete.
```

### Save a locale configuration

```sh
$ vim /etc/locale.conf
# Here I set English as default language
LANG=en_US.UTF-8
```

### Set keyboard (OPTIONAL)

If you [set the keyboard layout](https://wiki.archlinux.org/index.php/Installation_guide#Set_the_keyboard_layout), make the changes persistent in [vconsole.conf(5)](https://man.archlinux.org/man/vconsole.conf.5):

```text
vim /etc/vconsole.conf
--------------------------
KEYMAP=de-latin1
```

### Set host

Set your host name like: tom, bob, sally...etc

```text
$ vim /etc/hostname
---------------------
tom
```

Set hosts(If your hostname is called tom)

```
$ vim /mnt/etc/hosts
--------------
127.0.0.1     localhost
::1           localhost
127.0.0.1     tom.localdomain tom
```

### Set root password

```sh
# enter your root password
passwd
```

### Boot loader

Choose and install a Linux-capable
[boot loader](https://wiki.archlinux.org/index.php/Boot_loader).
If you have an Intel or AMD CPU, enable
[microcode](https://wiki.archlinux.org/index.php/Microcode)
updates in addition.

UEFI user need to [install grub on UEFI system](https://wiki.archlinux.org/title/GRUB#UEFI_systems)
which needs packages `grub` and `efibootmgr`. BIOS boot user don't need this.

Also I install `intel-ucode` for Intel CPU microcode support. And the
`os-prober` package for multiple system detect.

Use command below to test if you can use UEFI or not:

```console
$ efivar-tester
UEFI variables are not supported on this machine.
# This means you need to use BIOS boot method.
```

#### For BIOS boot user

First you need to change partition type for your boot partition. In the case
above, it is `/dev/sda1`. So using fdisk to change it's type:

```console
# replace with your machine specific name
$ fdisk /dev/sda
```

Press `t` to changed disk partition type. In my case, my boot partition is the first one. So input `1`. Then select `20`, which is the `BIOS boot` type. You can input `L` to list all available type.

And finally install `grub` with command:

```console
pacman -S grub intel-ucode
```

Then:

```console
grub-install --target=i386-pc /dev/sda
```

replace the `/dev/sda` with your disk name.

And finally generate the config:

```console
grub-mkconfig -o /boot/grub/grub.cfg
```

#### For UEFI boot user

```sh
pacman -S grub efibootmgr intel-ucode os-prober
```

Then install GRUB EFI application.

```console
# check platform use uname
$ uname -m
# x86-64

# install
grub-install --target=x86_64-efi --efi-directory=/efi

# generate config
grub-mkconfig -o /boot/grub/grub.cfg
```

## Before reboot:

You have something that must be checked:

- Network manage software (dhcpcd, NetworkManager...) installed or not.
- Set your root password with `passwd` or not.

## Then...reboot and enjoy!

```sh
exit
reboot
```

