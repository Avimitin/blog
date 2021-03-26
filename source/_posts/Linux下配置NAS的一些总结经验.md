---
title: 使用Linux搭建NAS
date: 2019-08-22 15:05
categories:
- [system, linux]
tags:
	- nas
	- samba
	- linux
tldr: "配置组合使用 Samba + Aria2 + Transmission 来实现 NAS 功能"
---

# Linux下配置NAS的一些总结经验

## 前言

经历了上一次的Windows Server的痛苦配置以后，我决定换成Linux来配置NAS。相对于Windows平台，Linux平台会更加自由和简洁。而且清晰的权限和后台也要比Win更加干净，唯一的弊端就是要去学不少命令。但是我这里会帮你总结好大部分要用到的命令，快速帮你搭建好Linux平台下的NAS。

## Linux的系统选择

我选择的是[Ubuntu 18.04](www.ubuntu.com)。但是对于NAS来说，稳定且更新少是必要需求，所以我更加推荐[Arch Linux](www.archlinux.org)。简洁干净，最好减少桌面配置，减少GPU占用。

## 配置思路

* SMB高速传输
* 基于Web的离线下载
* SSH命令行和Web页面管理系统

---

## 系统安装

下载好.iso文件以后，用[UltraISO](https://cn.ultraiso.net)烧录进U盘里。然后重启，按F12进入BIOS，检查启动列表，要打开UEFI启动选项。然后保存设置退出，按F8打开启动列表，选择UEFI 你的U盘。进入系统选择第一项，直接安装。关于硬盘配置，我只有一个SSD和一个HDD，所以把系统安装在SSD，让安装程序自动分区，HDD自动挂载用来做数据盘。其余没什么好说的，按照自己的需求一直下一部就行了。

## NAS正式配置

### SAMBA配置

一、安装Ubuntu SAMBA服务器

* 更新软件源：输入命令` sudo apt-get update `更新软件源
* 安装SAMBA服务：输入命令` sudo apt-get install samba `安装SAMBA服务器。

二、创建SAMBA配置文件

* 备份原配置文件：` sudo cp /etc/samba/smb.conf /etc/samba/smb.conf.bak`
* 创建共享目录：`sudo mkdir -p /你的共享文件夹路径` `#mkdir -p 用来创建路径中不存在的路径。`
* 更新目录权限：`sudo chmod -R 777 /你的共享文件夹路径` `#chmod -R 可以给你共享的目录和该目录下所有文件和子目录进行相同的权限变更。777即所有用户对该目录都有读写权。`
* 修改配置文件：`sudo vim /etc/samba/smb.conf`  #没有vim的请输入命令`sudo apt-get vim ` [vim的使用方法](https://www.runoob.com/linux/linux-vim.html) 

进入配置文件之后，按i进入编辑模式，把所有字段全部删除，输入以下配置(#号后注释文字要删除）：

```bash
[global] #这里是全局设置
workgroup = WORKGROUP #与Windows的工作组名保持一致
security = user #这里是访问安全级别，user为最低安全等级，需要输入用户名和密码。(网上的教程中的的share权限在更新之后已经关闭了，输入share权限默认最高安全等级。)
usershare owner only = false #给予其他设备访问权限
public = yes
browseable = yes
[你的NAS Name] #这里是分享路径配置
comment = User's NAS #这一段是标记，对配置没有影响。
path = /你的共享文件夹路径 #写上你自己的共享路径
read only = no #是否只读
writeable = yes #是否可写
browseable = yes #是否可浏览
guest ok = yes #是否可以给其他用户使用
public = yes #是否公开
create mask = 0777 #创建权限
directory mask= 0777 #目录权限
vaild users = user #输入当前用户名 
[你的NAS名字]
#如果同一台机子你想分开共享路径，就把上面的配置复制到这里。
```

三、创建SAMBA用户

* 输入命令 

`sudo smbpasswd -a smbuser` 

**注意！在创建samba用户之前请确保有一个同名的linux用户。**

如果想创建其他linux用户来使用samba，请输入命令` sudo adduser username `来创建新用户

四、重启SAMBA服务

* 输入命令 

` sudo systemctl restart smbd ` 

五、检查SAMBA服务是否正在运行

* 要检查samba 服务是否正在运行，请输入命令：

` systemctl status smbd `

` systemctl status nmbd `

* 要启用这两个服务，请运行以下命令：

` sudo systemctl start smbd `

` sudo systemctl start nmbd `

开始运行后，smbd将在139和445端口上侦听，若有无法访问，可以检查是否为端口封锁。

PS：Manjaro等Arch用户使用以下命令启用samba服务
```bash
systemctl enable smb nmb
systemctl start smb nmb
```

六、从其他设备访问SAMBA文件夹
* 在同一网络的Windows 设备上，打开此电脑，点击上方选项卡**计算机** ，选择选项__映射网络驱动器__，在文件栏输入\\Host ip #你的NAS ip地址\\你的共享文件夹名 (此处可以不输入根目录）

然后就可以直接使用了。

>###### 参考文章：
>>###### linux与window文件目录共享——samba配置及在windows映射  [2013-01-21]https://blog.csdn.net/mengfanbo123/article/details/8524924
>>###### Ubuntu下配置支持Windows访问的samba共享  [2014-02-14] https://blog.csdn.net/i_chips/article/details/19191957
>>###### samba配置文件注释 [2015-02-06]https://blog.csdn.net/dhgao38/article/details/43567403
>>###### 如何在Ubuntu 16.04上安装和配置Samba服务器以进行文件共享 [2017-11-02] https://www.linuxidc.com/Linux/2017-11/148194.htm

---

## 离线下载和远程控制的配置

### BT下载

一、BT软件下载
> BT软件推荐：
> Transmission、Deluge

以下以Transmission为例进行介绍

*  输入命令：` sudo apt-get install transmission ` 下载transmission。
*  输入命令：` sudo apt-get install transmission-daemon ` 下载transmission的web管理端

这样你就可以在桌面打开了，可以在应用程序页面中找到启动应用程序应用，把transmission勾选进开机自启动。

二、web管理BT下载

* 打开Transmission，点击编辑选项，点击首选项选项卡，点击远程选项卡。打开远程连接选项，输入你觉得比较好记的端口。

* 在其他设备上打开浏览器，在地址栏输入你的NASIP和你刚刚设置好的端口，例如`192.168.1.100:12345`, 你就已经可以使用web来管理BT下载了。

* 如果想要更加美观的界面和更多的设置选项，可以继续以下步骤：

> [Transmission-web-control](https://github.com/ronggang/transmission-web-control) 安装
> 1.获取最新脚本
>
> > 输入命令

```
wget https://github.com/ronggang/transmission-web-control/raw/master/release/install-tr-control-cn.sh

```

>> 请留意执行结果，如果出现`install-tr-control-cn.sh.1`之类的提示，表示文件已存在，请使用 `rm install-tr-control-cn.sh*` 删除之前的脚本再重新执行上面的命令。
>> 如果提示 https 获取失败，请使用以下命令获取安装脚本：

```
wget https://github.com/ronggang/transmission-web-control/raw/master/release/install-tr-control-cn.sh --no-check-certificate

```
>> 如果提示文件已存在，可以通过 `rm install-tr-control-cn.sh` 进行删除后再执行下载；或者在 wget 后面添加 -N 参数，如：
```
wget -N https://github.com/ronggang/transmission-web-control/raw/master/release/install-tr-control-cn.sh --no-check-certificate

```
> 2.执行安装脚本
> > 执行安装脚本（如果系统不支持 bash 命令，请尝试将 bash 改为 sh ）：
> > `bash install-tr-control-cn.sh`
> > 如果出现 Permission denied 之类的提示，表示没有权限，可尝试添加执行权限：
> > `chmod +x install-tr-control-cn.sh`
> > 如果命令成功执行，将出现文字安装界面：
> > 按照提示，输入相应的数字，按回车即可。

> 安装完成后，用浏览器访问 Transmission Web Interface（如：http://192.168.1.1:9091/ ）即可看到新的界面；如果无法看到新界面，可能是浏览器缓存了，请按 Ctrl + F5 强制刷新页面或 清空缓存 后再重新打开；注意，路径最后不要加web

### 离线下载
一、关于离线下载软件
> 离线下载推荐使用[aria2](https://aria2.github.io)，功能齐全，下载性能强悍，比迅雷会员下载还猛。

二、Aria2安装与配置
* Aria2 下载
  输入命令：
  ` sudo apt-get install aria2 `

Aria2完整安装：

```
sudo mkdir /etc/aria2 #新建aria2文件夹
sudo touch /etc/aria2/aria2.session #新建session文件
sudo chmod 777 /etc/aria2/aria2.session    #设置aria2.session可写 
sudo vim /etc/aria2/aria2.conf    #创建配置文件

```
* Aria2 配置
  vim 打开aria2.conf，将下列配置直接拷贝进文档内再自行进行编辑。

**注意！注释号内的配置皆为不生效使用默认配置，如果要自定义配置一定要把配置前的注释号删除！**

```
#'#'开头为注释内容, 选项都有相应的注释说明, 根据需要修改
#被注释的选项填写的是默认值, 建议在需要修改时再取消注释

#文件保存相关

# 文件的保存路径(可使用绝对路径或相对路径), 默认: 当前启动位置
dir=~/downloads
# 启用磁盘缓存, 0为禁用缓存, 需1.16以上版本, 默认:16M
#disk-cache=32M
# 文件预分配方式, 能有效降低磁盘碎片, 默认:prealloc
# 预分配所需时间: none < falloc ? trunc < prealloc
# falloc和trunc则需要文件系统和内核支持
# NTFS建议使用falloc, EXT3/4建议trunc, MAC 下需要注释此项
#file-allocation=none
# 断点续传
continue=true

#下载连接相关

# 最大同时下载任务数, 运行时可修改, 默认:5
#max-concurrent-downloads=5
# 同一服务器连接数, 添加时可指定, 默认:1
max-connection-per-server=5
# 最小文件分片大小, 添加时可指定, 取值范围1M -1024M, 默认:20M
# 假定size=10M, 文件为20MiB 则使用两个来源下载; 文件为15MiB 则使用一个来源下载
min-split-size=10M
# 单个任务最大线程数, 添加时可指定, 默认:5
#split=5
# 整体下载速度限制, 运行时可修改, 默认:0
#max-overall-download-limit=0
# 单个任务下载速度限制, 默认:0
#max-download-limit=0
# 整体上传速度限制, 运行时可修改, 默认:0
#max-overall-upload-limit=0
# 单个任务上传速度限制, 默认:0
#max-upload-limit=0
# 禁用IPv6, 默认:false
#disable-ipv6=true
# 连接超时时间, 默认:60
#timeout=60
# 最大重试次数, 设置为0表示不限制重试次数, 默认:5
#max-tries=5
# 设置重试等待的秒数, 默认:0
#retry-wait=0

#进度保存相关

# 从会话文件中读取下载任务
input-file=/etc/aria2/aria2.session
# 在Aria2退出时保存`错误/未完成`的下载任务到会话文件
save-session=/etc/aria2/aria2.session
# 定时保存会话, 0为退出时才保存, 需1.16.1以上版本, 默认:0
#save-session-interval=60

#RPC相关设置

# 启用RPC, 默认:false
enable-rpc=true
# 允许所有来源, 默认:false
rpc-allow-origin-all=true
# 允许非外部访问, 默认:false
rpc-listen-all=true
# 事件轮询方式, 取值:[epoll, kqueue, port, poll, select], 不同系统默认值不同
#event-poll=select
# RPC监听端口, 端口被占用时可以修改, 默认:6800
#rpc-listen-port=6800
# 设置的RPC授权令牌, v1.18.4新增功能, 取代 --rpc-user 和 --rpc-passwd 选项
#rpc-secret=<TOKEN>
# 设置的RPC访问用户名, 此选项新版已废弃, 建议改用 --rpc-secret 选项
#rpc-user=<USER>
# 设置的RPC访问密码, 此选项新版已废弃, 建议改用 --rpc-secret 选项
#rpc-passwd=<PASSWD>
# 是否启用 RPC 服务的 SSL/TLS 加密,
# 启用加密后 RPC 服务需要使用 https 或者 wss 协议连接
#rpc-secure=true
# 在 RPC 服务中启用 SSL/TLS 加密时的证书文件,
# 使用 PEM 格式时，您必须通过 --rpc-private-key 指定私钥
#rpc-certificate=/path/to/certificate.pem
# 在 RPC 服务中启用 SSL/TLS 加密时的私钥文件
#rpc-private-key=/path/to/certificate.key

#BT/PT下载相关

# 当下载的是一个种子(以.torrent结尾)时, 自动开始BT任务, 默认:true
#follow-torrent=true
# BT监听端口, 当端口被屏蔽时使用, 默认:6881-6999
listen-port=51413
# 单个种子最大连接数, 默认:55
#bt-max-peers=55
# 打开DHT功能, PT需要禁用, 默认:true
enable-dht=false
# 打开IPv6 DHT功能, PT需要禁用
#enable-dht6=false
# DHT网络监听端口, 默认:6881-6999
#dht-listen-port=6881-6999
# 本地节点查找, PT需要禁用, 默认:false
#bt-enable-lpd=false
# 种子交换, PT需要禁用, 默认:true
enable-peer-exchange=false
# 每个种子限速, 对少种的PT很有用, 默认:50K
#bt-request-peer-speed-limit=50K
# 客户端伪装, PT需要
peer-id-prefix=-TR2770-
user-agent=Transmission/2.77
# 当种子的分享率达到这个数时, 自动停止做种, 0为一直做种, 默认:1.0
seed-ratio=0
# 强制保存会话, 即使任务已经完成, 默认:false
# 较新的版本开启后会在任务完成后依然保留.aria2文件
#force-save=false
# BT校验相关, 默认:true
#bt-hash-check-seed=true
# 继续之前的BT任务时, 无需再次校验, 默认:false
bt-seed-unverified=true
# 保存磁力链接元数据为种子文件(.torrent文件), 默认:false
bt-save-metadata=true

```
* 启动aria2 

输入命令：` sudo aria2c --conf-path=/etc/aria2/aria2.conf `

如果没有提示错误，按` ctrl+c `停止运行命令，转为后台运行：

` sudo aria2c --conf-path=/etc/aria2/aria2.conf -D `

* 设置开机自动启动

输入命令创建：` sudo vim /etc/init.d/aria2c `

添加以下内容
```
#!/bin/sh
### BEGIN INIT INFO
# Provides: aria2
# Required-Start: $remote_fs $network
# Required-Stop: $remote_fs $network
# Default-Start: 2 3 4 5
# Default-Stop: 0 1 6
# Short-Description: Aria2 Downloader
### END INIT INFO
 
case "$1" in
start)
 
 echo -n "已开启Aria2c"
 sudo aria2c --conf-path=/etc/aria2/aria2.conf -D
;;
stop)
 
 echo -n "已关闭Aria2c"
 killall aria2c
;;
restart)
 
 killall aria2c
 sudo aria2c --conf-path=/etc/aria2/aria2.conf -D
;;
esac
exit

```
修改文件权限：` sudo chmod 755 /etc/init.d/aria2c `

添加aria2c服务到开机启动：` sudo update-rc.d aria2c defaults `

启动服务：` sudo service aria2c start `

查看服务状态：` sudo systemctl status aria2c `

* Aria2的使用

打开浏览器，在地址栏输入` http://aria2c.com `打开aria2的web管理器。打开右上角的设置，输入你的NAS机地址和你的aria2配置文件的端口，如果没有更改就是6800。

在chrome下载aria2插件，可以方便直接调用aria2下载东西，如果无响应可以直接拷贝链接下载。

三、关于Docker

Docker功能正在研究学习，后续会更新。

> ###### 参考文章：
> > ###### ubuntu安装配置aria2[2016-08-14] https://blog.csdn.net/crazycui/article/details/52205908
> > ###### ubuntu18.04 aria2的安装及使用详解 [2018-08-23] https://blog.csdn.net/qq_29117915/article/details/81986509
> > ###### Ubuntu安装aira2及开机启动配置[2018-03-01]https://www.jianshu.com/p/3c1286c8a19d

---

### 配置NAS的远程管理

一、关于远程管理
> 对Linux的远程连接可以使用SSH连接terminal来控制，也可以用Webmin来图形化控制

个人推荐SSH多一些，毕竟Linux用命令还是多一些的，而Webmin会稍微直观一些。

二、SSH的配置

* 安装SSH

输入命令:`sudo apt-get install ssh`

启动服务:`service sshd start`

配置端口:` vim /etc/ssh/sshd_config`

去除Port前的注释键，自定义端口。

查看服务是否启动：` ps -e | grep ssh `

无报错且` ssh-agent `和` sshd `两个程序在运行即可。

* Windows下载[putty](https://www.putty.org/)，输入你的NAS地址和端口即可远程使用Terminal来管理NAS主机。

三、Web管理

*Webmin安装

>由于包管理器中的源并没有webmin，我们需要去官网下载软件的包来进行安装

使用wget来下载包：

`cd /你的下载目录`

` wget https://prdownloads.sourceforge.net/webadmin/webmin_1.910_all.deb `

然后运行安装命令

` dpkg --install webmin_1.910_all.deb `

安装程序将会自动把Webmin安装进 `/usr/share/webmin`目录内。如果没有任何报错，你就可以在其他设备中输入`http://NASIPAddress:10000/`，请自行更换NAS的IP。

若遇到缺少依赖的报错，请输入该命令解决依赖：`apt-get install perl libnet-ssleay-perl openssl libauthen-pam-perl libpam-runtime libio-pty-perl apt-show-versions python`

若安装依赖库的过程中报错为无法找到包，请输入`vim /etc/apt/sources.list `，检查最后一行是否以universe结尾。

>###### 参考文章：Linux配置SSH服务实现远程远程访问[2018-03-30]https://blog.csdn.net/liguangxianbin/article/details/79759498
