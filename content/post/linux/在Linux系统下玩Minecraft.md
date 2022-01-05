---
title: 在Linux系统下玩Minecraft
date:  2019-08-11
tag:
- linux
- minecraft
- java
---

## Java环境安装

1. 打开终端

2. 通过包管理器安装OpenJDK8（直接放弃甲骨文的官方JDK，配置无敌麻烦。）

   ```bash
   # Ubuntu/Deepin
   sudo apt-get install openjdk8
   # Arch/Manjaro
   sudo pacman -S jdk8-openjdk
   sudo pacman -S jre8-openjdk
   ```

3.有时候会有报错，记得检查是否安装`openjfx8`

## HMCL下载

基本上大部分的Minecraft资源站都有各种版本HMCL资源，谷歌一下。这里有一份稍微新版本的HMCL备份。—>[HMCL-3.2.130版本备份资源](https://drive.google.com/open?id=19NYiTB2fkzUVvnIsrEZO69ZKf-tDdzEw)<—

## 正式开玩

1. HMCL启动

   ```bash
   # 进入HMCL所在目录
   cd ~/Downloads
   # 启动HMCL
   java -jar HMCL-xxx.jar
   ```

2. MC启动

- 首先先把各种个人信息项设置好。

- 进入游戏列表，打开安装新游戏版本，自己选择一个版本安装。

  ![游戏安装页面](https://img.vim-cn.com/56/5b191f9da21709566bae86e11658bfcedf49f8.png)

  游戏安装页面

- 注意先下载游戏，再安装整合包。
  [![img](https://img.vim-cn.com/06/82d3f1f9dff3adf01adbf119173139c7671f46.png)](https://img.vim-cn.com/06/82d3f1f9dff3adf01adbf119173139c7671f46.png)

- 游戏安装成功之后，点击下载好的游戏右侧的齿轮图标，点击自动安装，上方菜单栏选择在线安装。**先安装Forge，再安装OptiFine**

- 然后点击游戏设置，调整一下游戏内存和窗口分辨率

- 最后点击顶上的主页按钮，左下角开始游戏。

  ## Mod和光影安装

- 打开MC文件夹下载目录
  [![img](https://img.vim-cn.com/fa/5ae45480c59558f28d477175194df1f4b21297.png)](https://img.vim-cn.com/fa/5ae45480c59558f28d477175194df1f4b21297.png)

- 文件夹的作用

  > **mods**: 将mod放入其中，重启游戏生效
  >
  > **resourcepacks**: 将材质包放入其中，重启游戏之后到设置选择。
  >
  > **saves**: Minecraft本地地图存放在这里
  >
  > **shaderpacks**: 光影包存放在这里，重启游戏之后到视频选项中开启。

  ## 服务器宣传

  欢迎来我的服务器游玩，打开多人游戏，输入

  ```bash
  123.207.233.155
  ```

  进入服务器。服务器游戏版本为1.12.2，服务器负载4-5人游玩。24h不间断营业。（其实也没啥好玩的就当做是个云端地图吧。）
