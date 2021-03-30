---
title: 使用 FRP 实现一机多设备编程
date: 2020-10-25 09:20
categories:
	- tools
tags:
	- frp
	- cooporate
---


## 前言

最近配了台 ChromeBook 用来方便平常通勤写代码，但是 ChromeBook 羸弱的性能和拉跨的散热使得他日常编程体验极其垃圾。同时，回到宿舍之后用主机写代码和 ChromeBook 之间也会极其脱节。

也有考虑过用 git 分支来管理代码，但是有时候代码写了一半还有很多问题没调试，不想提交 commit 。或者使用 SAMBA 直接对笔记本的代码进行修改，这样又会遇到跨平台编译的问题。

于是我用了一段时间的 VSCode 的 Remote SSH 插件，在外用 ChromeBook，回到宿舍之后主机 SSH 连接 ChromeBook 来写代码。但是这样有些本末倒置了：本来性能更好的主机却只做了显示器的功能。于是我这周和朋友了解了一下 frp 并重新设计了部署方案。

## 设计

一机多设备编程部署方案：

![Design By XMind](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20201025132138.png)

一台高性能的本地服务器主机用作编译主机，部署 frp 服务到一台有公网 IP 的 VPS主机上，然后别的设备通过 VPS 转发流量到本地主机上来操作本地服务器。

## 平台

我的本地服务器用的 Ubuntu 20.4，公网服务器用的 Debian Buster。

## 部署

首先下载 [FRP](https://github.com/fatedier/frp) ，根据自己的系统和处理器选择，我这边选的 [linux-amd64](https://github.com/fatedier/frp/releases/download/v0.34.1/frp_0.34.1_linux_amd64.tar.gz) 版本。解压之后里面有两个程序，一个是 `frps` ，一个是 `frpc` ，分别对应服务器和客户端两个程序。

> 下面所述的字段在 [FRP Docs](https://gofrp.org/docs/) 里都有详细介绍，你可以根据自己需求更改。

### FRPS（服务器端）

将 `frps` 和 `frps.ini` 用 SFTP 发送到公网IP服务器，然后修改 `frps.ini` ：

```ini
;common 就是通用设置
[common]
;服务端监听端口，建议修改防止被扫
bind_port = 7000
;token设置
token = 12345qwert
;端口复用设置
tcp_mux = true
tcpmux_httpconnect_port = 7001
;log文件设置
;我这边没有设置，我使用的另外一个方法获取log
;有需求可以去掉前面的分号来启用自定义log
;log_file = /run/logs/frps/frps.log
;log_level = info
```

然后使用 `./frps -c frps.ini` 测试启动成功与否。按 `Ctrl + C` 停止程序。你还要记得在VPS面板更改防火墙设置把自己设置的端口打开。

### FRPC（客户端）

然后把 `frpc` 和 `frpc.ini` 复制到本地服务器，修改 `frpc.ini` ：

```ini
[common]
;这里填公网IP的机子的地址还有端口
server_addr = xxx.xx.xxx.xxx
server_port = 7000
;token服务端和客户端要一致
token = 12345qwert
;然后自定义一个方案名
[SSH]
type = tcp
;如果想要访问局域网内别的设备这里就填局域网别的设备的ip
local_ip = 127.0.0.1
local_port = 22
;remote_port 建议修改，防止被扫
remote_port = 6000
```

> 这里的 remote_port 是公网服务器监听的端口，即若有对 xxx.xxx.xxx.xxx:6000 的请求，公网服务器就会把请求转发到内网的 127.0.0.1:22 地址上。所以 remote_port 也要记得在公网服务器的防火墙里打开。

在公网服务器启动 frps 服务端，然后使用 `./frpc -c frpc.ini` 测试连接成功与否，如果成功连接公网服务器的 frps 服务端会有 `client login info` 的提示，然后会看到两边都有 `new proxy [ssh] success` 的提示，就代表转发代理建立成功了。然后你可以在任意一台设备上使用

```bash
ssh xxx@xxx.xxx.xxx.xxx -p 6000
```

来测试 SSH  转发成功与否。大部分失败都是因为防火墙没有打开对应端口。

## 安全性

由于将内网机器暴露在公网，SSH 尽量不使用明文密码登录，建议使用 pubkey 进行设备认证。

### 公钥生成并监听

```bash
#生成公钥
ssh-keygen -t rsa -C "example@gmail.com"
eval "$(ssh-agent -s)"
#私钥监听
ssh-add ~/.ssh/id_rsa
```

### 公私钥认证

在服务器端下的 `.ssh` 文件夹修改或者创建文件 `authorized_keys` 。在里面追加公钥(`.pub`结尾的文件)

```bash
# 客户端查看并复制公钥
cat ~/.ssh/id_rsa.pub
# 追加到 authorized_keys 里
```

每个设备的公钥占一行，隔行来分辨不同设备。

### 设置仅公钥登录

在本地服务器打开 `/etc/ssh/sshd_config` , 首先建议把默认的 22 端口修改为高位的端口减少被扫的次数（这里假如修改了记得要去 `frpc.ini` 修改本地端口）。然后找到 `PubKeyAuthentication` 字段，删掉前面的 # 号注释，并改为 `yes` 。找到 `PasswordAuthentication` 字段并改为 `no` 。这样就不能再用密码登录了。

## 持续可用性

我使用 `systemd` 来进行进程维护。下载的压缩包里已经有写好的 `.service` 文件，根据文件名复制到对应机子的 `/etc/systemd/system/` 文件夹下。并根据 `.service` 文件里的内容把程序和配置文件复制到对应文件，我选择修改 `.service ` 文件。拿服务器端举例：

1. 首先 `mv frps /usr/bin/` 移动 frps 到 `/usr/bin` 目录下。
2. 然后 `cp ./frps.ini /usr/local/frps` 复制配置文件到 `/usr/local/frps` 下。
3. 打开 `frps.service` 文件，将 ExecStart 字段修改为 `/usr/bin/frps -c /usr/local/frps/frps.ini`
4. 将 ExecReload 字段修改为 `/usr/bin/frps reload -c /usr/local/frps/frps.ini`
5. 然后 `cp frps.service /etc/systemd/system` 复制到 systemd 目录下。
6. 每次修改 service 文件都需要 `sudo systemctl daemon-reload` 。
7. 启动 frp  :   `sudo systemctl start frps`
8. 重启 frp : `sudo systemctl restart frps`
9. 关闭 frp：`sudo systemctl stop frps`
10. 查看 frp 状态 : `sudo systemctl status frps`
11. frp 开机自启 : `sudo systemctl enable frps`
12. frp log 查询 : `journalctl -au frps.service` 或 `journalctl -au frps.service -f` 查看最新日志并等待新日志输出。

## VSCode SSH Remote

部署完毕之后就是 VSCode 的设置了，安装好插件之后点击左下角的 SSH 按钮，在弹出菜单中点选 `Remote-SSH Open Configuration File` ，选择 config 存放位置，在里面填入

```yaml
Host xxx
	HostName xxx.xxx.xxx.xxx
	User xxx
	Port 6000
	IdentityFile "~/.ssh/id_rsa"
```

> - Host 任意填名字
> - HostName 填公网IP
> - User 就是登录用户名
> - Port 远程端口，即我们设置的转发端口 6000
> - IdentityFile 填私钥地址

然后就可以多设备同机编程了。

