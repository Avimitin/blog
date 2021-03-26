---
title: ZSH的美化配置
date: 2019-08-06 15:56
categories:
- [system, linux]
tags:
- zsh
- linux
---
## ZSH,Oh-My-ZSH安装

1. 安装zsh

   ```
   $ yay -S zsh
   //修改默认shell路径
   $ sudo vim /etc/passwd
   //找到用户名的那一行，将bash路径改为/usr/bin/zsh
   ```

2. 安装Oh-My-ZSH

   ```
   $ yay -S oh-my-zsh
   ```

## ZSH配置

1. 更换主题

```
//输入"/"对字符进行搜寻，查找"ZSH_THEME=***"这一行，将theme后面的改成已存在的主题就行//zsh配置文件位置
$ sudo vim ~/.zshrc
```

或者直接换成火箭主题：

```
//克隆仓库
$ git clone https://github.com/denysdovhan/spaceship-prompt.git "$ZSH_CUSTOM/themes/spaceship-prompt"
//链接主题
$ ln -s "$ZSH_CUSTOM/themes/spaceship-prompt/spaceship.zsh-theme" "$ZSH_CUSTOM/themes/spaceship.zsh-theme"
//修改主题
$ vim ~/.zshrc//ZSH_THEME="spaceship"
//特殊符号字体安装
yay -S powerline-fonts
```

1. 自动补全提示插件

   ```
   $ cd ~/.oh-my-zsh/custom/plugins
   $ git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
   $ vim ~/.zshrc
   //plugins=(git zsh-autosuggestions)
   ```

2. zsh美化

- 字体安装

  ```
  $ yay -S ttf-fira-code
  ```

- 颜色配置
  文本: #BFE9F3
  背景: #282B34
  调色板: 自定义，把标准色往灰度高处调。
