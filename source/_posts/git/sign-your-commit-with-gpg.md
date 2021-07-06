---
title: 'Sign Your Commit With GPG'
author: avimitin
date: 2021/07/06 11:42
tag: [gpg, git, privacy]
---
# 使用 GPG 来签名 Commit

前几天一个朋友在群里提出了一个我从未考虑过的问题：
“如果我在一个项目里用你的邮箱 commit，你会知道吗？”。

这个问题把我问住了。众所周知，在第一次使用 `git`
的时候，我们需要填写以下两个信息：

```console
git config --global user.name "tom"
git config --global user.email "tom@example.com"
```

当你进行项目开源或者和他人合作时，别人就可以在 commit
历史里用这些信息查询到是你提交的这条 commit 。你所
使用的代码托管平台甚至还会依据你填写的这些作者信息，
让你荣登 Contributor 之列。

似乎没什么问题的样子，为什么我还要大费周章的写一篇
《使用 GPG 签名 Commit》，上面提到的逻辑究竟发生了
什么问题呢。

## 小剧场

想象这么一个场景：现在是凌晨两点，你奋战了一天，终于
把一个顽固的代码 Bug 修好了。你感觉到一阵昏昏欲睡，想
赶紧上床好好休息一下。带着对自己代码的满足感，你 commit
了自己的心血： `"fix: fixed the security valnerability"`
你推送到了自己的代码托管平台，等待明早的新版本推送前，
大家对你进展的检查和评价。你爬上床，关好灯，困意逐渐
渗透，你开始意识模糊了起来......

突然，邮件和各种通讯软件的通知声把你轰炸了起来，你
困惑的思考着似乎自己才刚进入梦乡，怎会这么快就到白天。
而接下来屏幕里的内容，更让你震撼的坚信自己还未睡醒：
大家都在责怪你的新 commit 让整个项目都混乱了。

你赶忙打开电脑，检查起 git log。你一行一行的查看 commit
哈希值，你找到了让项目崩坏的 patch，你不认识它，你从未
写过任何一行存在在 patch 里的改动，但令你绝望的是，这个
patch 上签署着你的名字，你的邮箱。毫无疑问，就是你，提交
的这个 commit。

> 以上小剧场改写自 
> [Mike Gerwitz: A Git Horror Story: Repository Integrity With Signed Commits](https://mikegerwitz.com/2012/05/a-git-horror-story-repository-integrity-with-signed-commits)

## 问题

我们都能知道自己做了啥，但是怎样才能让别人知道，我，
真的做过这么个事情。要怎样才能让别人知道，写这些修改
，签署这个修改，提交这些修改的人，真的是我。

所以问题就在于虽然 git 可以证明这是你写的 commit，
但是它缺少了另一个抵抗恶意的环节：应该如何证明这个签
着我的名提交了 commit 的人真的就是我本人？

可能也有人觉得这并不重要：我基本只写个人项目，团队
项目也都是认识的人，搞破坏的人他们没有写权限。那我们
从另外一个角度出发。在 GitHub 上有些项目，我不好说，但
是是政治敏感项目。假设有人冒用你的名字你的邮箱，在这
些项目 “做贡献”，等秋后算账之时，警察叔叔按图索骥找
上你家的门，你应当如何证明你的清白？吃着火锅唱着歌儿，
结果突然背黑锅了，这可如何是好？

## 解决方案

在这里我们将采用 GPG 生成 RSA 公私钥，并使用私钥对你的所
有 commit 进行签名。

> GPG 是什么：
> - [GnuPG](https://gnupg.org/)
>
> 公私钥的实现原理以及其如何保护我：
> - [公钥加密简介](https://privacy.n0ar.ch/Encryption/Pubkey/Pubkey)

### 生成公私钥

首先应当确认你的电脑上已经安装好了 GnuPG，你可以运行
下面的命令来确认：

```console
gpg --version
```

在开始生成密钥前，最好检查一下你的系统是否生成足够多
的熵来生成随机数：

```console
cat /proc/sys/kernel/random/entropy_avail

#--possible output--
#3718
```

这个数尽量要大于 3000，如果不够 3000 你可以随便敲敲键盘
，点点鼠标，到处上网冲浪一下。

然后我们开始生成密钥：

```console
gpg --full-generate-key
```

终端会提示你选择密钥类型：

```text
Please select what kind of key you want:
   (1) RSA and RSA (default)
   (2) DSA and Elgamal
   (3) DSA (sign only)
   (4) RSA (sign only)
Your selection?
```

如果没有别的需求，选择默认的 `RSA and RSA` 即可。

接下来就选择密钥大小。因为现代算力的增强，大部分的报
告推荐选择 2048 及以上的密钥大小。这里我选择了 4096 bits。

```text
Please specify how long the key should be valid.
         0 = key does not expire
      <n>  = key expires in n days
      <n>w = key expires in n weeks
      <n>m = key expires in n months
      <n>y = key expires in n years
Key is valid for? (0)
```

下一步是选择该密钥何时过期，我选择的是一年后过期。选
择不过期也是可以的，但是如果你想撤销以前发布的公钥会
比较麻烦，因为不会自动过期，保存旧公钥的用户可能不会
知道公钥已经被撤销了。

最后一步就是填写身份信息，根据需要发布的身份填写即可。

```text
GnuPG needs to construct a user ID to identify your key.

Real name: sh1marin
Email address: sh1marin@example.com
Comment: code signing
You selected this USER-ID:
    "sh1marin (code signing) <sh1marin@example.com>"

Change (N)ame, (C)omment, (E)mail or (O)kay/(Q)uit?
```

如果填错了，输入提示的大写字母重新填写即可。最后输入
O 结束录入。确认身份之后，gpg 会弹出一个窗口让你输入
私钥密码。非常建议你填写上，防止被人偷窃私钥盗用。

输入下面的命令来查看你刚刚生成的密钥：

```console
gpg --list-keys
```

### 修改密钥

你可以对已生成的密钥进行编辑管理：

```console
gpg --edit-key mail@example.com
```

在弹出的新的提示符后输入 help 查看帮助信息。

下面的例子演示了如何在密钥内加入另一个身份信息：

```gpg
gpg> adduid
Real name: sh1marin
Email address: sh1marin@school.edu
Comment: Student account
You selected this USER-ID:
    "sh1marin (Student account) <bestuser@school.edu>"

Change (N)ame, (C)omment, (E)mail or (O)kay/(Q)uit? O
```

最后输入 save 保存退出。

### 导出公钥

公钥可以用来加密信息，同时也可以用来验证你的私钥签名
。公钥就是被设计出来用作四处分享的。

使用下面的命令来导出你的公钥：

```console
gpg --export --armor --output anyname.pub
```

同时为了防止被篡改，你还可以在你的主页或者任何你可以
信赖的平台发布你的指纹信息，让他人验证公钥正确性和完
整性。

```console
gpg --fingerprint
```

你可以在自己的主页放上指纹和公钥，也可以发布到诸如
[OpenPGP keyserver](https://keyserver.ubuntu.com/)
等公钥托管网站，你甚至可以丢到 GitHub 上用 raw 来
分享。

> 上述公私钥生成部分参考自：
> [Red Hat: How to create GPG keypairs](https://www.redhat.com/sysadmin/creating-gpg-keypairs)

### 签名 commit

长篇大论之后，终于来到最重要的这一步了。首先，你需要
用以下命令取得私钥的 ID：

```console
gpg --list-secret-keys --keyid-format=long
```

在下述例子中，假设你的私钥 Key ID 是 `3AA5C34371567BD2`。

```console
$ gpg --list-secret-keys --keyid-format=long
/Users/hubot/.gnupg/secring.gpg
------------------------------------
sec   4096R/3AA5C34371567BD2 2016-03-10 [expires: 2017-03-10]
uid                          Hubot 
ssb   4096R/42B317FD4BA89E7A 2016-03-10
```

接着需要告诉 Git 你要用这个私钥来签名：

```console
git config --global user.signingkey 3AA5C34371567BD2
```

设置好之后，在每次 commit 时加上 `-S` 参数来签名 commit。

```console
git commit -S -m "feat: some fancy new feature"
```

> 也可以在 git config 中加入：
> `git config --global commit.gpgsign true`
> 来默认打开签名，就不用每次自己手动输入 `-S` 参数了。

最后 Push 到你使用的远程托管仓库。

> 参考自：
> [GitHub: Signing commits](https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification/signing-commits)

### 上传公钥

- GitHub

点击右上角的头像 -> Settings -> SSH and GPG keys -> GPG keys -> New GPG
key

- Gitea

右上角头像 -> Settings -> SSH/GPG Keys -> Manage GPG Keys -> Add Key

## 附

逻辑上说，好像我也可以在举证时再去签名，只要签名不对
那就不是我干的。但是实际上要让社区信任公钥确实来自于
我，也是一个很重要的步骤。如果不先把自己和公钥联系起
来，如何证明公钥是我的呢。

Web of trust 还是基于人的信任机制，漏洞诸多。(By Blare)
理论和应用和现实有很大差别，希望各位读者不会遇到恶意，
也不要成为恶。
