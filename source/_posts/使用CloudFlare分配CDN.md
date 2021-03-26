# 前言

CloudFlare是真的减速器，假如服务器在香港可能上了CF会很慢，所以假如像我一样用完延迟从60ms蹦到500ms的，就不要搞CF了。

<!--more-->

# 添加域名

首先先注册CloudFlare，然后点击添加站点，输入你的域名。然后就会跳转计划，选择0美元的免费计划就可以了。选完之后CloudFlare就会开始扫描你现在的DNS缓存。

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/Snipaste_2020-04-19_17-56-04.png)

一般来说CloudFlare不用CNAME类型，把CNAME类型删了，再添加一条A类型记录，名称为www，指向VPS，改完应该长这样：

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20200419175954.png)

点击继续，使用默认方法。然后跳转到域名面板，去域名购买的网站改一下域名DNS服务器。

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20200419180406.png)

假如使用namesilo的用户就到`manage my domain`页面里，勾选域名，然后点击上面第二个图标`change nameserver`更换DNS服务器。

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20200419180516.png)

把原来的`NameServer`删除，填入CloudFlare页面提供的名称服务器：

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20200419180556.png)

然后等待DNS生效就可以了。

# IP测试

打开CMD，先刷新本地DNS：

```bash
ipconfig/flushdns
```

然后ping你的域名：

```bash 
ping example.com
```

假如ping出来的不是VPS的IP而是CDN的IP那就是生效了。

# CloudFlare设置

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20200419180742.png)

点击`SSL/TLS`选项，假如你的服务器有SSL认证的话，选择`完全(严格)`，没有认证的话选择灵活。

CloudFlare也有提供证书认证，而且是永久免费的，可以用宝塔面板自定义证书填入。

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20200419181137.png)

点击右上角的速度，可以把这里面能打钩的全部选上，但是效果的话，只能说... 感 知 不 强。

# 站长工具

- 使用[站长工具网站](https://seo.chinaz.com/)来测试自己的站点。

- [这个网页](https://tool.chinaz.com/sitespeed)可以测试访问站点的速度。