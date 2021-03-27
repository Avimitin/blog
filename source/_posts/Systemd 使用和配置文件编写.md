---
title: Systemd User Guide
date: 2020-04-03 15:52
categories:
	- [system, linux]
tag:
	- systemd
---
# Systemd 使用和配置文件编写

## Systemd 常用命令

```bash
//启动服务
systemctl start xxx.service
//关闭服务
systemctl stop xxx.service
//重启服务
systemctl restart xxx.service
//检查服务状态
systemctl status xxx.service
//更改自启动状态
systemctl enable/disable xxx.service
//查看目前启动服务
systemctl list-units --type=service
//重启systemd
systemctl daemon-reload
```

## xxx.service编写详解

```bash
[Unit]
Description=Telegram Bot Second Python Program	//服务描述，随便写
Wants=network-online.target	//服务需要的依赖
After=network-online.target	//服务启动的时间，在某插件启动后再启动服务

[Service]
Type=simple //默认值，以主程序启动服务[1]
User=root	
Group=root
WorkingDirectory=/root/Telegram_bot/Telegram_Bot/venv	//服务主程序的路径，需要绝对路径
ExecStart=/usr/bin/python3 Bot2.py	//启动服务的命令，前半段写启动程序的软件，后半段写程序名字
PrivateTmp=true	//是否给独立空间运行
Killsignal=SIGINT	
TimeoutStopSec=10s
Restart=always	//必填字段，使用自动重启帮助守护进程

[Install]
WantedBy=multi-user.target	//默认启动target，填multi-user可以使用enable设置开机自启动
```

