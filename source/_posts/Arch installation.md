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

get from https://archlinux.org/download/

install from https://wiki.archlinux.org/index.php/Installation_guide

press `e` to edit install configuration like:

```sh
nomodset video=800x450
```

## Setting font 

for more comfort look
`setfont /usr/share/kbd/consolefonts/LatGrkCyr-12x12.gz`

## Configure key:

**This is optional**

```sh
touch keys.conf
#-------edit--------#
keycode 1 = Caps_Lock
keycode 58 = Escape
#----end of edit-----#
loadkeys keys.conf
```

## Setting Wireless

```sh
# check up device
ip link 
# open up a device
# example:
# ip link set wlan0 up

# check wireless device
# sort only ESSID
iwlist wlan0 scan | grep ESSID

# generate a wifi cofiguration
wpa_passphrase {WIFI_NAME} {WIFI_PASSWORD} > wifi.conf

# connect to wifi 
# -c use a configuration
# -i specific a device
# & running in background
wpa_supplicant -c wifi.conf -i wlan0 &

# dynamic ip address
dhcpcd &
```

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

| Mount point | Partition               | [Partition type](https://en.wikipedia.org/wiki/Partition_type) | Suggested size          |
| ----------- | ----------------------- | ------------------------------------------------------------ | ----------------------- |
| `[SWAP]`    | `/dev/*swap_partition*` | Linux swap                                                   | More than 512 MiB       |
| `/mnt`      | `/dev/*root_partition*` | Linux                                                        | Remainder of the device |

#### UEFI with GPT

| Mount point               | Partition                     | [Partition type](https://en.wikipedia.org/wiki/GUID_Partition_Table#Partition_type_GUIDs) | Suggested size          |
| ------------------------- | ----------------------------- | ------------------------------------------------------------ | ----------------------- |
| `/mnt/boot` or `/mnt/efi` | `/dev/*efi_system_partition*` | [EFI system partition](https://wiki.archlinux.org/index.php/EFI_system_partition) | At least 260 MiB        |
| `[SWAP]`                  | `/dev/*swap_partition*`       | Linux swap                                                   | More than 512 MiB       |
| `/mnt`                    | `/dev/*root_partition*`       | Linux x86-64 root (/)                                        | Remainder of the device |

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

Once the partitions have been created, each newly created partition must be formatted with an appropriate [file system](https://wiki.archlinux.org/index.php/File_system).

The partition of `/mnt/boot` must be `FAT32` format:

```sh
# DEVICE_PARTITION_NAME can get from
# fdisk and use p command to get
# example:
# mkfs.fat -F32 /dev/sda1
mkfs.fat -F32 {DEVICE_PARTITION_NAME}
```

And the root partition `/mnt` can use `ext4` format, other format See [File systems#Create a file system](https://wiki.archlinux.org/index.php/File_systems#Create_a_file_system) for details.

```sh
# Example:
# mkfs.ext4 /dev/sda2
mkfs.ext4 {DEVICE_PARTITION_NAME}
```

created a partition for [swap](https://wiki.archlinux.org/index.php/Swap), initialize it with [mkswap(8)](https://man.archlinux.org/man/mkswap.8), enable it with [swapon(8)](https://man.archlinux.org/man/swapon.8):

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

## Final Install

> all the /dev/sda1 or sda2 is mention above, replace them with your machines specific device name.

[Mount](https://wiki.archlinux.org/index.php/Mount) the root volume to `/mnt`. For example, if the root volume is `/dev/sda2`

**CARE OF ALL YOUR MOUNT ACTION**

```sh
# Check your disk name
fdisk -l
# mount **root** to /mnt
mount /dev/sda2 /mnt
# check mount status
ls /mnt
# create a directory for boot
mkdir /mnt/boot
# mount your **boot device** to it
mount /dev/sda1 /mnt/boot
```

and finally, time to install:

```sh
# install base,linux,linux-firmware to /mnt
pacstrap /mnt base linux linux-firmware
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
# leave chroot first(because of the lack of text editor)
exit
# edit the locale file in root
vim /mnt/etc/locale.gen
# find en_US.UTF-8
/en_US <Enter>
# remove # before
en_US.UTF-8 UTF-8
# get back to root
arch-chroot /mnt
# generate locale
# should get output: 
#     en_US.UTF-8... done
```

### Save a locale configuration

```sh
# leave chroot
exit
# edit
vim /mnt/etc/locale.conf
# add line below, this will automatically 
# set English as default language
LANG=en_US.UTF-8
```

### Set keyboard

If you [set the keyboard layout](https://wiki.archlinux.org/index.php/Installation_guide#Set_the_keyboard_layout), make the changes persistent in [vconsole.conf(5)](https://man.archlinux.org/man/vconsole.conf.5):

```text
vim /mnt/etc/vconsole.conf
--------------------------
KEYMAP=de-latin1
```

### Set host

set your host name like: tom, bob, sally...etc

```text
vim /mnt/etc/hostname
---------------------
tom
```

Set hosts

```
vim /mnt/etc/hosts
--------------
127.0.0.1	localhost
::1		localhost
127.0.0.1	tom.localdomain	tom
```

### Set root password

```sh
# get back to root
arch-chroot /mnt
# enter your root password
passwd
```

### Boot loader

Choose and install a Linux-capable [boot loader](https://wiki.archlinux.org/index.php/Boot_loader). If you have an Intel or AMD CPU, enable [microcode](https://wiki.archlinux.org/index.php/Microcode) updates in addition.

```sh
pacman -S grub efibootmgr intel-ucode os-prober
mkdir /boot/grub
grub-mkconfig > /boot/grub/grub.cfg
# check platform use uname
# uname -m
# x86-64
grub-install --target=x86_64-efi --efi-directory=/boot
```

### Install network manager

if you have WiFi only you must set up this software else you will get no Internet and no way to install them.

```sh
pacman -S wpa_supplicant dhcpcd
```

### End

```sh
exit
reboot
```

