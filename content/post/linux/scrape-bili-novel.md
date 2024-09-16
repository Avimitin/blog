+++
title = '从 bilinovel 扒小说'
date = '2024-09-16'
tag = ['novel']
author = 'sh1marin'
+++

最近买了个电子书看漫画和轻小说，漫画从 mox.moe 上下载了，但是轻小说没找到啥好资源站。
最后盯上了对 bilinovel.com 爬虫的方案。

主要用的 <https://github.com/lightnovel-center/linovelib2epub> 这个项目。
项目本身只是一个纯库，需要额外自己写 python 调用。
不过好在库封装的很简单，基本只需要两行就能搞定。

```bash
git clone https://github.com/lightnovel-center/linovelib2epub.git --depth=1
cd linovellib2epub
python -m venv venv
source venv/bin/activate.fish # 我用的 fish
pip install -r requirement-dev.txt # 他们的 requirement.txt 很久没更新过了，直接用 dev
pip install -e . # 把这个库装进 venv 里
```

接下来写个调用就好：

```python
from linovelib2epub import Linovelib2Epub, TargetSite

if __name__ == "__main__":
    linovelib_epub = Linovelib2Epub(
        book_id=177,
        divide_volume=True,
        target_site=TargetSite.LINOVELIB_PC
    )
    linovelib_epub.run()
```

把 `book_id` 换成网站上对应轻小说的书号即可。
`divide_volume` 也推荐加上，不然所有书卷都会封装在同一个 epub 文件。

执行的时候，库会尝试启动 chrome 去爬虫，如果挂着个窗口很麻烦，或者是在服务器上爬，
可以 chrome 加一行配置文件，让 chrome 默认 headless 启动：

```conf
# ~/.config/chromium-flags.conf
--headless
```

chrome 本身吃代理环境变量，因此如果想换 IP 可以简单的：

```bash
https_proxy=http://... python download.py
```
