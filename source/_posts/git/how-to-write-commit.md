---
title: How to write commit
date: 2021-10-22 21:47
categories:
	- git
tags:
	- git
	- commit
---
## 前言

对我而言，撰写 git commit 的内容是一个很重要的事情。因为我会把 commit
里的信息当成一份邮件，在开源社区里每个人都需要知道我做了什么，我为什么
要这么做，而编写 commit，就像在群发邮件一样。一份清晰的 commit 信息可以
给后续开发带来很多方便，也是对自己行为的一种声明。

我也一直在实践中找出一份简单但清晰的 commit 规则，以下内容
是我目前总结出来的比较简单但清晰的一份模板，我也会在后续不断重构不断更新。

## 结构

首先需要对 commit 要包含什么内容有一个清晰的概念。一个 commit 需要包含标题、
正文、脚注三个部分。每一部分都应该用一个空行分开。

## 标题

commit 的标题应当简洁明了的陈述你做了什么。注意我用的字眼：“陈述”。最好不要在
标题里包含任何时态，任何语气符号以及任何夸张用词。同时为了历史记录的一致性，
推荐用全小写来书写你的标题。

对比以下两个 commit，你更喜欢那一种呢？

```text
I've fixed almost all the problems!
-----------------------------------
fix(socket): clean up buffer when establish to new socket
```

我个人的品味是更倾向于选择下面这种书写方式，清晰明了的说明了做了什么(fix)，
对什么做了修改(socket)，大概做了哪些修改(正文)。

相信有些读者可能对这种写法非常熟悉，他来自于著名的开源框架 Angular。
Angular 的贡献指南里明确的约定了如下的标题书写格式：

```text
<type>(<scope>): <short summary>
  │       │             │
  │       │             └─⫸ Summary in present tense. Not capitalized. No period at the end.
  │       │
  │       └─⫸ Commit Scope: animations|bazel|benchpress|common|compiler|compiler-cli|core|
  │                          elements|forms|http|language-service|localize|platform-browser|
  │                          platform-browser-dynamic|platform-server|router|service-worker|
  │                          upgrade|zone.js|packaging|changelog|docs-infra|migrations|ngcc|ve
  │
  └─⫸ Commit Type: build|ci|docs|feat|fix|perf|refactor|test
```

你需要先表明这份补丁的类型，涉及哪些模块，和简短的补丁总结。而且每一个 type 恰好也可以对应
项目的版本号：fix 对应版本中最小的一位，比如现在是 `v1.0.2`，提交 fix 类型的补丁后就对应升
级到 `v1.0.3`。feat 意思是 feature，对应版本的中间一位，既提交了新的特性之后，版本号对应升
级到 `v1.1.0`。而带有 BREAKING CHANGE 的 commit 对应版本的主要一位，既提交后版本号升级到
`v2`。

用这种写法写了一段时间之后我发觉他太过于冗长了。我不是说不推荐这种写法，只是对于我来说，我
无法忍受写完 refactor(module) 等等之类的前缀信息之后，就没剩多少空间写概述了。

是的，标题是有限字数的，标题最好能控制在 50 个字符以内，这样在生成 commit 历史时可以一目了
然。于是进行一段时间之后，我选择寻找一个更加简练的书写方案。

我把目光移向了著名的内核项目：Linux。对于这种重量级的世界范围的开源项目，他们应该也有一个
规定好的习惯。打开了 [Torvalds/Linux](https://github.com/torvalds/linux/commits/master) 的
commit 历史，我发现他们是这样写 commit 的。

```text
module: summary
--------------- Example ----------------
drm/kmb: Enable ADV bridge after modeset
```

前缀说明修改的模块，然后跟一个冒号，以及概述。

好！这就是我想要的！足够简短，也足够清晰。我在 Linux 的基础上又给自己多加了一个要求：
他们偶尔会用大写开头的概述，而我继续 Angular 的习惯选择全小写，这样能保证历史记录足够
美观。在后续查找资料的时候，我发现 Golang 团队也有全小写的约定，我的 remix 看来也是
能够被广泛接受的。

同时不管是哪个项目，都不推荐在 commit 里用句号结尾，没有必要，会让标题很难看。

格式确定下来之后，就是如何撰写内容了。结合 Linux 和 Golang 两个项目的习惯，我制定了自
己的一套内容概述方式。首先不要用任何时态，并且用动词开头。根据 Golang 的文档，他们
推荐了一个完形填空的方式来帮助书写内容：“这个补丁如果应用了，将会让项目......”。举个例
子，比如我修改了我爬虫项目的数据部分的本地存储部分，让他在请求失败时自动清除缓存，那么
这个修改的 commit header 就是："data/storage: clean the cache file when request failed"。
套用进去就是“应用了这个补丁之后，这个爬虫项目会在请求失败时清除缓存文件”。

有了这样的完形填空技巧和格式，相信你也应该能很顺手的写出一个规范且易读的 commit 标题了。

## 正文（主体）

主体部分其实相对要求没那么严格。你可以把他当作一篇日记来写。只要描述你为什么这么做，具体
做了什么就可以了。在格式上，建议使用纯文本，不要用 HTML 或者 Markdown 这类标记语言，就像
你在代码里写的注释那样，用正确的标点符号和完整的句子叙述即可。

如果你有类似于这次修改后的测试跑分，也可以用来作为 “为什么“ 的例证添加到主体里。

如果你对如何写补丁的修改理由还有疑惑，可以看看 git 的发明者 Torvalds Linus 是怎么编写他的
emacs 的 commit 的: [torvalds/uemacs](https://github.com/torvalds/uemacs/commits/master)。
他的主体部分详细的讲解了他当时的思路，和他具体的修改，我觉得是一个非常好的参考资料。

## 脚注

脚注部分我参考了 Linux 项目，在末尾用 `Signed-off-by: Username <Email>` 的形式签署自己的
commit。这里的 Username 和 Email 推荐和你 git config 保持一致。这样写的好处是可以在社区
提前建立好一个信任链，降低被人冒名顶替的风险。其次是方便一些对项目有想法的人能快速找到你的
联系方式，和你电邮沟通。

在签名的上方，如果你的项目有什么破坏性更改，或者弃用了什么功能，记得用 `BREAKING CHANGE`
和 `DEPRECATED` 标注。这算是好习惯的一部分，可以帮助你方便的生成改动日志，也能让一些更新
后出问题的用户有迹可循。

## 总结

最后你的 commit 信息大概长这样：

```text
module: do some modification

I want to make the project blablabla...

# 如果有重大修改，如果没有可以不写
BREAKING CHANGE: API some_fn() need 3 arguments now

Signed-off-by: MyName <name@example.com>
```

## 模版

git 提供了模版功能，在你 commit 的时候以这个文本文件为模板来生成 commit。你可以在里面注
释长度，辅助文本，以及提前生成好格式。
具体的内容可以查看我的模版：https://github.com/Avimitin/dotfile/blob/master/.gittemplate

你可以自己写一份符合自己胃口的模板，保存为文本文件，然后用命令
`git config --global commit.template PATH/TO/TEMPLATE` 设置好模板路径，下一次 commit 的
时候就会加载好了。
