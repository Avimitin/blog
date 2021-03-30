---
title: 用阿里云OSS做图床
date: 2020-04-19 20:21
categories:
	- tools
tag:
	- oss
	- photo
---
# 购买资源包

直接买LRS标准就可以了，按照自己需求买，不过存几张图而已40G够用了。

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/Snipaste_2020-04-19_16-29-26.png)

然后买下行流量包，这里的流量是每月刷新的。

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/Snipaste_2020-04-19_16-44-37.png)

> 嫌贵的话其实可以不买，直接按实际流量扣费，0.25/GB。但是要做好被DDos或者突然高访问量然后一夜负债房子卖掉。

# Bucket创建

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/Snipaste_2020-04-19_16-56-51.png)

> 1. Bucket名称要填独一无二的，同时也不要太长，方便自己
> 2. 选择哪里都无所谓，但是一定要选国内的
> 3. 存储空间选标准

# 下载PicGo

PicGo是一个方便的批量上传图片的软件，而且也能管理自己的图片链接，在作者的[GitHub Release](https://github.com/Molunerfinn/PicGo/releases)里下载安装。

# 配置图床

## AccessKey获取

首先要取得OSS的访问秘钥，鼠标移至阿里云控制台右上角的头像处，点击Access Key管理，点击创建Access Key。然后点击开始使用子用户Access Key。

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/Snipaste_2020-04-19_17-10-16.png)

接着就会跳转创建用户，输入一个账户名，勾选编程访问，点击确定创建。

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/Snipaste_2020-04-19_17-12-10.png)

创建完成会跳转到用户信息，请一定要现在就保存好这个ID 和Secret，这里的信息只会显示现在这一次。可以下载CSV或者点复制保存。

然后点添加权限，选择管理对象存储服务权限一个就够了。

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/Snipaste_2020-04-19_17-14-52.png)

## 配置PicGo

然后打开PicGo，选择阿里云图床，输入刚刚的AccessKey和其他信息：

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/Snipaste_2020-04-19_17-19-15.png)

> 1. 存储空间名是刚刚的bucket名。
> 2. 存储区域是一开始选择的区域，可以在bucket信息里面找endpoint信息。
> 3. 存储路径请一定保证OSS有这个目录。

然后就在上传区直接拖动照片上传就可以了。
