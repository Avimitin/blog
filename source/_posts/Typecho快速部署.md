---
title: 快速配置Typecho
date: 2020-04-18 20:09
categories:
	- blog
tag:
	- blog
	- typecho
---
# 前言

因为Hexo迁库太麻烦了，而我经常忘记备份。而且每次写文章都要先渲染成静态页面。所以整了个Typecho，随时随地写文章，迁站只要宝塔一键就完事。

<!--more-->

# 资源预备

## 选购服务器的一些建议

假如你有国内备案的就最方便了。但是假如没有的话服务器尽量选香港服务器。尽量不要选`Vultr`了，延迟太高。可以尝试华为云香港或者阿里云香港。能白嫖就白嫖。

假如还是硬头皮选`Vultr`请走我`aff`：<https://www.vultr.com/?ref=8527098>

服务器买和配置啥的自己找教程了，太小白的教不动，系统可选CentOS或者Debian，服务器就不要选别的发行版了，方便方便自己。

然后只是作为博客而已，不用买配置太牛掰的，省点钱吃点好吃的对自己好一点。`Vultr`的话选个5美金的配置就完全够用了。华为云的话选个`1vCPU|1GB|1Mbit/s`就可以了。最近华为云有在做活动，可以嫖两个月，试试他们的香港机子吧。

## 选购域名

首先推荐[namesilo](https://www.namesilo.com/)，这个网站不需要实名注册，而且支持支付宝，没有外币卡也能快乐pay。

![namesilo域名购买界面](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-18_16-39-55.png)

> 1. 在这里搜索你想要的域名关键词，比如我叫avimitin就搜索avimitin
> 2. 勾选你想要的域名后缀
> 3. 假如没有满意的后缀，就点击这个按钮看更多。

勾选完之后点击下面`Register Checked Name`，域名保护可选可不选，有钱你就买。

![购物车页面](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-18_16-58-28.png)

> 1. 自动续费选择no
> 2. 隐私设置选择WhoIS Privacy，是永久免费的域名隐私保护
> 3. 购买年限自己挑
> 4. 填okoff可以便宜一点
> 5. 然后点击continue注册

注册部分我就不讲了，你自己搞，然后注册完支付页面选支付宝，那个邮箱那里填个能联系上你的，扫码支付，搞定。

## 域名解析

域名购买完之后，点击右上角的`Manage My Domain`，勾选你想要用的域名，在最右边的Options选项栏里选择那个小地球，第五个按钮。

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-18_17-16-21.png)

> 删除原来的域名记录
>
> 1. 在这里选择一个类型
> 2. 常规解析先用一个A类型，地址填你刚刚买的服务器的ip
> 3. 再用一个CNAME 类型，主机名填WWW，地址填写你买的域名
>
> 别的都不用管，TTL3600就行了。

DNS服务器可以先不改，假如你要用到CDN再改，我会在CloudFlare的教程里面讲到。

# 站点建设

## 服务器防火墙设置

请记得到服务器页面打开防火墙，然后打开以下TCP端口：`22|80|443|8888`

> Tips：
>
> - 22：SSH端口
> - 80：HTTP端口
> - 443： HTTPS端口
> - 8888：宝塔端口
>
> 假如连接不上服务器的某个服务，就去防火墙检查是不是相对应端口没打开。

## SSH连接

下载个XShell，还有配套的XFTP。说实话我觉得XShell比Putty好用一些。然后到你VPS的页面复制用户名和密码。

> 假如你不想下载软件的话，用PowerShell输入 `SSH root@123.123.123.123 -p 22`
>
> 把IP改成自己的回车就行了。

## 安装宝塔和LNMP

**以下所有命令和操作建立在Debian10系统基础上**

首先同步更新一下库和软件：

```bash
apt-get update
apt-get upgrade
```

然后安装宝塔：

```bash
wget -O install.sh http://download.bt.cn/install/install-ubuntu_6.0.sh && bash install.sh
```

宝塔安装完之后复制界面上的`URL`，用户名和密码，到浏览器登录。假如第二次登录忘记账号密码了使用这个命令查询用户名密码:

```bash
bt default
```

登录网站之后会自动弹出LNMP安装页面，请注意**PHP安装7.1及以上版本！！**

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-18_17-16-21.png)

然后泡杯咖啡等他安装，先去干点别的。

## Typecho配置

### Typecho下载

点击[Typecho](https://typecho.org/download)下载页下载1.1正式版

### 添加站点

打开宝塔界面，点击左边菜单栏的网站选项，点击添加站点。

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-18_19-34-05.png)

> 1. 写下域名
> 2. 选择MySQL
> 3. 记住这里是下划线

然后创建完成。

### 数据库

点击左边数据库，点击选项root密码，把root密码复制了。

> 没必要改，忘记了过来复制就行，改成你熟悉的密码还会增加密码泄露风险

### 安装Typecho

点击左边的文件按钮，进入你的网站目录，也就是刚刚创建的/www/wwwroot/example.com目录。把里面的东西全部删除了，然后点击上传，把刚刚下载的压缩包上传到网站目录下，解压。

为了防止混淆，啰嗦多几句，写个流程出来：

- 打开`www/wwwroot/example.com`目录
- 删除原有所有文件
- 上传下载的Typecho压缩包
- 解压缩
- 删除压缩包
- 打开build文件夹
- 全选文件，剪切
- 返回上一级(返回到`www/wwwroot/example.com`目录)
- 粘贴，删除build文件夹。

具体操作可以看这个图。

![img](https://pic2.zhimg.com/v2-082727a103520cd686cbd397ed6e86ed_b.webp)

### 配置Typecho

然后在浏览器访问`example.com/install.php`

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/installphp.jpg)

> - 数据库地址填localhost
> - 端口3306不要改
> - 用户名填root
> - 数据库密码把刚刚复制的密码粘贴上去
> - 填example_com
> - 账户密码是管理员权限的，不要忘记了。

然后打开`example.com`就是你的博客啦

管理页面在`example.com/admin`里面。现在就可以把个人信息自定义一下了。

# 美化主题

## 主题下载

主题在这个网站里面可以找：[Typecho Themes](https://typecho.me/)，找到喜欢的主题，点击demo看看效果是否满意。然后到作者的Github库里找新发布的release，下载zip。假如作者没有上传release的话就直接clone整个库。

我个人是推荐直接clone，之后作者有什么改动只要输入`git pull`就可以平滑更新了。

下面以`Rorical`主题举例来示范如何安装主题，这个主题在我不断鼓励(~~鞭策~~作者之后已经非常简约完美了。

## `Rorical`主题安装

1. 首先打开`Rorical主题项目`的仓库网页：<https://github.com/Liupaperbox/Rorical>

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-19_11-49-23.png)

> 1. 单击这个按钮
> 2. 复制这个Git URL
> 3. 或者点这个按钮复制

2. 然后打开你的SSH软件，进入网站根目录(也就是那个`www/wwwroot/example.com/`)，然后输入:

```bash
# 进入主题目录
cd usr/themes
# 复制整个仓库
git clone https://github.com/Liupaperbox/Rorical.git
```

3. 然后打开后台管理页面，点击控制台-外观，把主题勾选成`Rorical`主题.

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-19_11-46-44.png)

然后点击设置外观：自己设置一些图片路径

3. 底下的widget设置用的`Font-Awesome`里的图标，[FontAwesome图标库网址](http://www.fontawesome.com.cn/faicons/)。一个widget写一行。格式为：

```bash
fa fa-link$$example words$$URL
```

# 网站安全设置

## 宝塔登录安全

返回宝塔面板，点击面板设置，勾选上Google验证，然后把面板端口改成你记得住的端口号，不要一直用8888。

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-19_12-05-07.png)

> 1. 手机下载Google Authority,扫码认证
> 2. 面板端口改成好记但是不是常规端口。
> 3. 然后把8888给删除了
>
> PS：**记得记得记得先在防火墙打开你准备设置的端口**

## Typecho登录安全

首先把要把登录页面改掉，也就是把`example.com/admin`改成`example.com/abcdefg`：

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-19_12-09-59.png)

> 进入网站根目录，用mv命令把admin改名成abcdefg

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-19_12-13-19.png)

> 然后用vim打开`config.inc.php`，把原来'/admin/'后台路径改成'/abcdefg/'

之后登录后台就用`example.com/abcdefg`就可以了。

## 原站保护

关于如何使用CDN保护原站IP可以看我这篇文章->

# 其他设置

## 图床

**建议选完图床之后备份好自己的图片，免得图床炸了之后修复死链找不到图**

- 我这里有两个免费图床的建议：

> <https://sm.ms>
>
> 这个的速度更快一点点:<https://img.vim-cn.com/>。

- 或者也可以看[这篇文章](https://codein.icu/Github-PicGo-Jsdelivr/)学习使用`github`配合`jsdelivr`搭建自己的图床。速度是很不错的，就算在国内裸连也有不错的速度。
- 亦或是像我一样花十几块钱买一个OSS搭建图床：[使用OSS搭建图床教程](none)

## SSL证书

SSL证书是为了让别人访问你的站点的时候，能够使用HTTPS访问，加密传输并提供身份验证。HTTPS其实就是在HTTP通道的基础上加入了SSL层用来加密。更详细原理请谷歌查询，反正最直观的表现就是出现这个锁头了

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-19_13-19-06.png)

---

SSl证书申请有两种办法：

### 宝塔一键申请

打开宝塔控制面板，选择网站，点击对应域名的设置按钮，找到SSl选项面板，点击Let's Encrypt签名，然后打开强制HTTPS。完工。

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-19_13-23-32.png)

### Certbot申请

首先安装Certbot，不是Debian的用你的包管理器查询一下有没有，没有就用编译安装:

```bash
apt install certbot
```

然后就用Certbot申请证书,可以一直`-d`添加所有你想要注册的域名，一般来说这俩就够了：

```bash
certbot certonly --webroot -w /www/wwwroot/example -d example.com -d www.example.com
```

假如出现`443`端口被占用，基本都是被Nginx占用了，那么就先`kill`掉应用：

```bash
#使用netstat -anp|grep 443查询占用的软件
$ netstat -anp|grep 443
tcp    0    0 0.0.0.0:443    0.0.0.0:*    LISTEN    21395/nginx: worker
```

这个`21395`就是占用端口程序的PID，使用kill命令关闭

```bash
kill 21395
```

然后再申请就可以了，记得重启Nginx。

## 重定向

假如你有一大堆域名没用，想全部解析到博客上，可以用宝塔的重定向将其他域名指向博客。

> **注意：不要忘记域名解析了**

1. 打开宝塔，点击网站的设置，添加一个新的域名

![](https://avi-pic-storage.oss-cn-shenzhen.aliyuncs.com/pic/Snipaste_2020-04-19_13-54-00.png)

2. 点击下面重定向，访问域名选择`example.org`，目标URL填入`example.com`，然后点击启用301。
3. 假如还有更多域名就点底下的测试版，重复1，2步骤。
