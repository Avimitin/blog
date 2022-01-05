---
title: '白嫖博客指南'
author: avimitin
date: 2021/06/29
tag: [hexo, blog]
---
# 白嫖博客指南 #

因为我的机子又小又弱，顶不住 DDoS，我一直是处于怂逼
状态，不敢把博客公开，都是自己用。然后最近发现，其实
我也没什么国内访问的需要，为啥不转移到托管平台呢？
于是这周稍微折腾了一下，把博客从 vps 的 nginx 转移
到了 CloudFlare Pages 上。

## 具体思路 ##

本地建立 hexo 环境，建立 git 仓库，把源文件 stage
到 master 分支，提交到 GitHub 时使用 Actions 编译，
然后把静态文件 stage 到别的分支。CloudFlare 监听
这个静态文件的分支，有静态文件变动就生成。最后域名
CNAME 到 CloudFlare 就可以了。

## 预备需求 ##

- 一个 GitHub 帐号
- 一个 CloudFlare 帐号
- 一个终端
- 曾经使用过 hexo (没有的话看 hexo 的文档)
- （可选）一个自己的域名

## Hexo 安装 ##

> 我因为常用 Arch Linux 所以就只基于 Arch 环境来讲
> 一些终端操作。如果你发现环境不一样，你可以试着开个
> docker 来运行。或者你也可以到我的 
> [仓库 issues](https://github.com/Avimitin/blog/issues)
> 来询问我。

首先需要安装 [nvm](https://github.com/nvm-sh/nvm) 。
跟着 README 的安装指引一步一步来就好。安装好之后
开始安装 node 和 npm

```console
nvm install node
```

用版本命令查看是否安装成功：

```console
node --version
npm --version
```

确认没啥毛病之后，开始安装 hexo：

```console
npm install -g hexo-cli
```

如果有遇到任何问题请查看 hexo 的 [官方文档](https://hexo.io/docs)。
如何在 hexo 下写文章也请查看官方文档。

## GitHub ##

首先你需要在 GitHub 创建一个新的 Repository，然后进入你的
hexo 文件夹，把所有编译需要的文件 commit 到 master 分支。

### GitHub Actions ###

然后在这个文件夹创建两个目录： `.github/workflows`，并新建
一个 `build.yml` 文件，我们往里面开始写工作流：

首先是给你的工作流取个名，这个不是很重要，就简单叫他 `blog` 好了。

```yaml
name: blog
```

然后我们写一下工作流触发的条件：

```yaml
on:
  push:
    branches: [ master ]
  pull_request:
    branches: [ master ]
```

这里限定工作流在 `push` 和 `PR` 到 master 分支时再触发构建。

然后开始设置工作环境，因为没有什么系统依赖，这里简
单用 ubuntu 镜像就可以了：

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
```

接着开始设置拉取仓库的步骤：

```yaml
steps:
  - uses: actions/checkout@v2
```

如果你像我一样，对主题做了一些魔改，用了 git submodule
，你还需要设置一个步骤拉取 submodule：

```yaml
- name: install theme
  run: git submodule init && git submodule update
```

然后设置 node 环境：

```yaml
- name: setup node
  uses: actions/setup-node@v2
```

设置安装 hexo 并安装所有 node 依赖的步骤：

```yaml
- name: install hexo
  run: sudo npm install -g hexo
- name: setup build environment
  run: npm install
```

最后就是开始构建：

```yaml
- name: build
  run: hexo generate
```

构建完之后，我设置了让他部署到同个仓库的步骤：

```yaml
- name: deploy
  env:
    USER: 'avimitin'
    EMAIL: 'avimitin@gmail.com'
    REPO: 'github.com/Avimitin/blog'
    TOKEN: ${{ secrets.GH_TOKEN }}
  run: |
    git config --global user.name $USER
    git config --global user.email $EMAIL
    cd public && git init -b gh-pages && git add .
    git commit -m "update blog"
    git push --force "https://$TOKEN@$REPO" gh-pages:gh-pages
```

这里面，env 是设置环境变量，方便以后修改。因为 workflows 里的
git 是全新安装的，我需要设置一下 user 信息。

配置好 git config 后，进入静态文件编译后的输出文件夹，在这里我
`cd public`。然后在这个文件夹创建一个新的 git 仓库，并设置默认
分支为 `gh-pages`，最后用 `git add .` 添加所有文件。

最后的 push 步骤，你可以使用常用的 https 用户密码登录，或者
ssh 公私钥验证。我用了 github 的 token 访问来降低步骤，关于
token 如何获得可以看下方的 FAQ。

push 时一定要带上 `--force` 参数，不然会遇到本地历史记录不同步
不允许同步的问题。

完成后的工作流文件可以参考这里： 
[.github/workflows/blog.yml](https://github.com/Avimitin/blog/blob/master/.github/workflows/blog.yml)

## CloudFlare ##

打开 CloudFlare 官网并登录，在下方找到网页选项卡（英文叫 Pages）。
点击进入设置页面。

进入页面后，点击创建项目按钮，绑定你的 GitHub 账户，然后选中
你的博客仓库。

在分支部署选项中，选中你的静态文件分支。在下方的框架预设中选
None，构建输出目录和构建目录都留空，保存后等待 CloudFlare
部署即可，大约 3-4 分钟就可以了。

如果你有自己的域名，跟随 CloudFlare 的指引 CNAME 上去即可。

## FAQ ##

### 如何不把敏感信息写在 workflow 里 ###

在仓库的 settings 页面里，点击左侧的 Secrets 页面，点击
右上角的 new secrets，设置一个隐藏的环境变量。

### 如何使用 GitHub 的 token ###

点击右上角你的头像，点击设置，下拉到 `Developer Settings`
页面，选择 `Personal access tokens` 选项卡，创建一个新的 Token。
这个 Token 只需要仓库的读写权限就够了。Token 如何使用请参阅
workflows 的最后一步。

## 参阅资料 ##

- Hexo 官方文档：https://hexo.io/docs/
- Git 使用方法：https://git-scm.com/book/zh/v2
- GitHub Actions 指引：https://docs.github.com/en/actions/quickstart
- 我的 Workflows 例子：https://github.com/Avimitin/blog/blob/master/.github/workflows/blog.yml
- CloudFlare Pages 文档：https://developers.cloudflare.com/pages/

