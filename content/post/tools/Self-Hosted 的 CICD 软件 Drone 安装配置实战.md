---
title: Self-Hosted 的 CI/CD 软件 Drone 安装配置实战
date: 2021-02-04
tags:
- devops
- ci
---

# Drone 安装配置实战

## 前言

因为穷用不起 GitLab，又想拥有一个自己的代码分发平台，于是搭建了自己的 Gitea，然而因为 Gitea 没有 DevOps，我不得不设置多一条 remote 上游来嫖 GitHub action 来用。这个寒假趁着事情不多，花了2个小时折腾了一下可独立部署的 CI/CD 程序：Drone。

安装 Drone 你需要：一台公网服务器；一台性能不错的服务器；GitHub，GitLab，Gitea 或者任意你正在使用且支持 Drone 的平台；一个域名。

## 下载

为了干净和快，Server 和 Runner 都采用了 Docker 来安装。所以首先要把 Docker 装上

### 安装 Docker

下面的方法仅适用于 Ubuntu，别的 Linux 发行版建议前往官网查询[安装方法](https://docs.docker.com/engine/install/) 。

```bash
# 安装配置 https 源的依赖
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
    
# 增加 Docker 的签名
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# 确认签名是否相同
sudo apt-key fingerprint 0EBFCD88

pub   rsa4096 2017-02-22 [SCEA]
      9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
uid           [ unknown] Docker Release (CE deb) <docker@docker.com>
sub   rsa4096 2017-02-22 [S]

# 然后增加 docker(x86-64 版本) 源
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

# 更新源并安装
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

# 测试安装成功与否
sudo docker run hello-world
```

### 拉取 Drone

```bash
sudo docker pull drone/drone:1
```

### 拉取 Drone Worker

```bash
sudo docker pull drone/drone-runner-docker:1
```

## 配置

### Gitea

首先打开你的 Gitea 主页，右上角点击 settings，点击中间的 application 进入 OAuth 页面

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20210204151736.png)

拖动到下面，点击 Manage OAuth2 Applications 栏目的 Create a new OAuth2 Application 表单。Application Name 随你喜欢的填写，Redirect URI 填写你的 Drone 域名。

> 比如：Application Name = Drone; Redirect URI = https://drone.example.com 。

![](https://docs.drone.io/screenshots/gitea_application_create.png)

然后 Gitea 会生成 Client ID 和 Client Secret，先不要急着点 Save，打开你的机器，先把 server 的信息填好启动。

### Drone Server

```bash
sudo docker run \
  --volume=/var/lib/drone:/data \
  --env=DRONE_GITEA_SERVER=https://gitea.example.com \
  --env=DRONE_GITEA_CLIENT_ID=Client ID \
  --env=DRONE_GITEA_CLIENT_SECRET=Client Secret \
  --env=DRONE_RPC_SECRET=rand Hash \
  --env=DRONE_SERVER_HOST=drone.example.com \
  --env=DRONE_SERVER_PROTO=https \
  --publish=80:80 \
  --publish=443:443 \
  --restart=always \
  --detach=true \
  --name=drone \
  drone/drone:1
```

上面的变量中，你需要自定义并修改

- `DRONE_GITEA_SERVER` 填入你的 Gitea 域名(必须带上 http 或 https 前缀) 
- `DRONE_GITEA_CLIENT_ID` 填入刚刚 Gitea 生成的 Client ID
- `DRONE_GITEA_CLIENT_SECRET` 填入刚刚 Gitea 生成的 Client Secret
- `DRONE_RPC_SECRET` 填入一串密钥，你可以在命令行输入 `openssl rand -hex 16` 来随机生成。
- `DRONE_SERVER_HOST` 填入你的 drone 服务器域名，不要填入 http 或者 https 前缀，只需要域名
- `DRONE_SERVER_PROTO` 填 https 协议就好。

然后回车启动服务器。在 Gitea 的 Application 页面点击 Save 保存这个 OAuth2 应用。

在浏览器访问你的 Drone 域名，页面会先跳转回 Gitea 进行 OAuth2 验证，你需要点击确定来授权 Drone 访问你的仓库。如果一切顺利，在几秒的空白页之后你就能见到一列仓库了。

### Drone Worker

在终端运行：

```bash
sudo docker run -d \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e DRONE_RPC_PROTO=https \
  -e DRONE_RPC_HOST=drone.company.com \
  -e DRONE_RPC_SECRET=super-duper-secret \
  -e DRONE_RUNNER_CAPACITY=2 \
  -e DRONE_RUNNER_NAME=${HOSTNAME} \
  -p 3000:3000 \
  --restart always \
  --name runner \
  drone/drone-runner-docker:1
```

你需要定义的配置有：

- `DRONE_RPC_HOST` 设置为你的 Drone 域名
- `DRONE_RPC_SECRET` 设置为刚刚在服务器填入的随机生成的密钥
- `DRONE_RUNNER_NAME` 随你喜欢设置名字

然后回车运行，输入 `docker logs hash(run 之后弹出的哈希值)`  查看 runner 是否正常运行。

## 各类问题

1. 页面一直空白，没反应

查看一下是不是服务器的 `DRONE_GITEA_SERVER` 变量填写少写了 https 或者 http，导致跳转失败。

2. 在仓库里加入了 `.drone.yml` 之后没反应

打开你的仓库，点击设置，拉到 webhook 选项，试着推送一次，看看有没有错误。

