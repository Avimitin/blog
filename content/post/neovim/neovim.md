---
title: 让你的 neovim 像 IDE 一样强大
date: 2021-02-04
tags:
- neovim
- coc.nvim
- ide
---

# 让你的 NeoVim 和 IDE 一样强大

## 前言

Neovim 是一款完全拥有 vim 特性且拥有高度支持插件的编辑器软件，而他的高可自定义能力也深深吸引了像我这样的人。这篇文章我会带着你编写一个漂亮且强大的终端编辑器的配置文件，同时稍微的带你入门一点 neovim，但没那么零基础哦。

我的配置文件在 [Avimitin/nvim](https://github.com/avimitin/nvim) , 欢迎 fork 使用。

## 下载安装

neovim 有一个很重要的 `pop_up_windows` 弹出窗口特性在 0.5.0 版本才有，还有很多 `TUI` 的支持，蛮多插件依赖于这些特性的，所以我不建议在默认源的包管理器中下载稳定版的 0.4 版本。关于 neovim 的安装指南在 [wiki](https://github.com/neovim/neovim/wiki/Installing-Neovim) 里面有，Ubuntu 的话也可以通过添加 PPA 源的方式，添加 unstable 源用 apt 包管理器安装：

```bash
# 如果你已经安装了，先卸载
sudo apt remove neovim 

# 添加 0.5.0 的源
sudo add-apt-repository ppa:neovim-ppa/unstable
sudo apt-get update
sudo apt install neovim
```

安装好 neovim 之后，还需要安装 Python3 和 NodeJS 来获得完全的插件支持：

```bash
sudo apt install python3 python3-pip
```

NodeJS 非常不推荐在包管理器中安装，建议查看 NVM 安装教程或参考[这篇文章](https://github.com/Avimitin/nvim/blob/master/docs/nodejs_install.md) 。

安装好 Python3 和 NodeJS 之后，安装 neovim 的依赖：

```bash
pip3 install pynvim
npm install -g neovim
```

如果想要获得像我配置一样的更多的插件支持，你还需要安装:

```bash
sudo apt install lazygit ctags ranger
```

## 开始配置

### 检查依赖

在配置文件之前，在终端输入 `nvim` 打开 neovim 。直接键盘输入 `:checkhealth` ，然后按下回车执行来检查依赖是否齐全：

![直接敲就行](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20210204171124.png)

回车之后你会看到这样的页面：

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20210204171212.png)

checkhealth 指令会帮你检查所有插件的依赖，在没安装依赖之前，找到`health#nvim#check` 这一行，全部显示 OK 即可，没有显示 OK 的就把缺的依赖安装好。之后安装插件如果运行不正确你都可以输入这个指令来检查是不是依赖丢失。然后输入 `:q` 退出该页面。

### 编写设置

linux 的 neovim 配置通常在 `~/.config/nvim/` 目录下，是一个命名为 `init.vim` 的文件。Windows 的版本通常在 `C:/Users/$USERNAME/AppData/Local/nvim` 下。

在对应系统的目录下创建 `init.vim` 文件，我们需要写一些设置。我会在设置背后表明注释，你可以参考设置。对于具体的设置选项你可以输入 `:help specific_setting` 来了解详细，比如你不懂 `lazyredraw` 是什么，你可以输入 `:help lazyredraw` 来翻看文档。一般来说，你直接复制下面的所有设置即可，大部分非常贴近普通 IDE 的设置了。然后如果你喜欢每个 tab = 4个空格的话，你可以设置 `tabstop` | `shiftwidth` | `softtabstop` 的值为 4。

```Vimscript
set exrc                                                  " exec command in init.vim
set secure                                                " safely do command above
set autochdir                                             " auto change directory
set number                                                " setting line
set relativenumber                                        " setting line in relative mode
set cursorline                                            " set line below cursor
set noexpandtab                                           " use only '\t' as tab
set tabstop=2                                             " show how many space for a '\t'
set shiftwidth=2                                          " use how many space for >> or << key
set softtabstop=2                                         " use how many space when pressing tab
set autoindent
set list                                                  " show hiding char
set listchars=tab:\|\ ,trail:·                            " define tab and space show
set scrolloff=4                                           " least amount line below and above the cursor
set ttimeoutlen=0                                         " set never wait for key
set notimeout
set viewoptions=cursor,folds,slash,unix                   " remember where to recover cursor
set wrap                                                  " auto line feed
set tw=0                                                  " text width for automatically wrapping
set indentexpr=
set splitright
set splitbelow
set noshowmode                                            " not showing current mode
set showcmd                                               " show cmd inputing like key combine
set wildmenu                                              " auto finish vim command
set ignorecase                                            " ignore case when searching
set smartcase                                             " ignore case only on searching
set shortmess+=c                                          " don't show useless msg
set inccommand=split                                      " show substitution automatically
set completeopt=longest,noinsert,menuone,noselect,preview " complete opject with a menue
set ttyfast                                               " make scrolling faster
set visualbell                                            " flash screen to notify error
set updatetime=100
set virtualedit=block
set colorcolumn=100
set lazyredraw
set re=0                                                  "make increase speed

" set folding paragraph
set foldmethod=indent
set foldlevel=99
set foldenable
set formatoptions-=tc

" keep undo or temp file
set hidden
silent !mkdir -p ~/.config/nvim/tmp/backup
silent !mkdir -p ~/.config/nvim/tmp/undo
set backupdir=~/.config/nvim/tmp/backup,.
set directory=~/.config/nvim/tmp/backup,.
if has('persistent_undo')
	set undofile
	set undodir=~/.config/nvim/tmp/undo,.
endif
```

### 编写快捷键

> 这边你可以直接复制我的设置，也可以根据个人喜好设定。

首先需要介绍几个概念，一般的快捷键语法非常简单，录入 `map key function` 三部分就可以了，比如我要让 d 键变成删除整行，就可以 `map d Vd` ，但是你如果你熟悉 vim 你就会知道，vim 有几种不同的模式，只用 map 可能会键位冲突，所以 vim 里还有 `imap` | `vmap` | `nmap` 分别对应输入时，选中时，普通模式时三种键位。而只是这样还不太够，假如我设置 `nmap r d` 来让 r 键代替原来的 d 键，会出现递归的命令传递，结果你输入 r 键也变成删除整行了。所以 vim 还有另外一个概念：norecusive map，你可以设置 `noremap | nnoremap | inoremap | vnoremap`  ，分别对应不递归的快捷键，普通模式限定快捷键，插入模式限定快捷键，选中模式限定快捷键，从而避免递归指令。

作为一个懒癌，我不想移动手去摸鼠标，甚至不想去右移到方向键，默认的 `hjkl` 键位我用的不是很顺手，所以我把指针移动配置改成了 `uhjk` ：

```vimscript
noremap <silent> u k
noremap <silent> k l
```

> 因为 vim 会显示所有你输入的指令，`<silent>` 可以避免一些频繁且不会出错的操作一直不停输出显示。

改了这两个键之后，指针操作就顺手多了，为了移动的更快，我又加入了倍数操作：

```vimscript
noremap <silent> U 5k
noremap <silent> H 0
noremap <silent> J 5j
noremap <silent> K $
```

这样一来，输入大写 U 就可以往上飞 5 行，输入大写 H 就可以跳转到行首，以此类推，让指针操作更符合直觉。

但是这样设置完之后，我把原来撤销操作的 u 键位给霸占了，所以我又加入了 Windows 常用键位 `Ctrl - z` 来替代撤销操作：

```vimscript
noremap <silent> <C-z> u
```

其中的 c 代表着 Ctrl，对于 Shift 和 Alt，也有 s 和 a 分别对应，比如 `ctrl alt shift a` 就可以设置成 `<c-a-s-q>` ，这里的大小写不敏感。

你也可以把一些常用命令写进去，比如我们需要经常写一点保存一点，但是老是输入 `:w` 加回车也过于麻烦了，所以你可以这样写：

```vimscript
noremap <silent> <C-s> :w<CR>
```

这里的 `<CR>` 代表着回车。然后你按下 Ctrl + S 就能保存文件啦。

除了以上键位，vim 还引入了一个 `<LEADER>` 键位的概念，你可以设置 leader 键来实现更多快捷键：

```vimscript
let mapleader=" "
```

这里我把 leader 设置为空格，在普通模式下他就可以帮我实现更加多的快捷键设置了，比如常常苦恼人的 “vim 怎么复制粘贴的问题” ，你不需要输入繁杂的命令，配置一个:

```vimscript
vnoremap <LEADER>y "+y
nnoremap <LEADER>p "+p
```

就可以在选中文字后按下 **空格+y** 来复制文字，在普通模式中按下 **空格+p** 来粘贴文字。应该是非常简单且容易理解的，关于更多键位实现，你可以参考我的配置文件复制粘贴即可。

### 配置插件

只是实现一堆快捷键并没有本质上改变 neovim 只是编辑器的本质，我们还需要安装各种插件来让 neovim 更加强大。大部分的插件在 github 上都能搜索到，vim 插件都可以在 neovim 上使用的，反过来不行。但是大量插件，管理是一个很麻烦的事情，于是我在这里给你们推荐一个很好用的 vim 插件管理插件，[vim-plug](https://github.com/junegunn/vim-plug) 。你可以参考 README 安装。

为了高度可迁移能力，你可以在 init.vim 中加入这么一行来避免每次都要手动安装 vim plug：

```vimscript
if empty(glob('~/.config/nvim/autoload/plug.vim'))
	!curl -fLo ~/.config/nvim/autoload/plug.vim --create-dirs
			\ https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
	autocmd VimEnter * PlugInstall --sync | source $MYVIMRC
endif
```

这段命令会自动下载并启用 vim plug ，顺便帮你把插件都安装好。但是由于国内不可言说的原因，你可能会遇到 `no resolve host` 的错误，就只能自己动手爬梯子手动安装了。

vim plug 安装好之后，在你的 init.vim 中加入一段命令：

```vimscript
call plug#begin('~/.config/nvim/plugged')
call plug#end()
```

这段命令会激活所有包裹在里面的插件，比如：

```vimscript
call plug#begin('~/.config/nvim/plugged')
Plug 'bpietravalle/vim-bolt'
call plug#end()
```

就可以激活 vim-bolt 这个插件。如果是第一次安装插件，你可以依照上面的格式写好 `Plug 'github author/repo'` 的格式填进去，保存之后输入 `:PlugInstall` 命令，vim-plug 会帮你下载并处理好插件：

![依赖 vim-plug，你可以下载安装并管理几乎所有的 github 上的 vim 插件。](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20210204181653.png)

想要关闭插件就在 `Plug...` 字段之前加上 `"` 注释掉。如果想要删除在 `~/.config/nvim/plugged` 目录里找到并删除就可以了。更多好用的插件你可以参考我的配置文件，或者持续关注下栏的插件推荐。

## 插件推荐

### coc.nvim

![](https://user-images.githubusercontent.com/251450/55285193-400a9000-53b9-11e9-8cff-ffe4983c5947.gif)

coc.nvim 是一个利用 Language Server Protocol (LSP) 实现代码补全功能的插件，VSCode 也是用的这个 LSP 来代码补全的。它自身也带有一个插件系统，帮你安装各类用 JS 写的语法补全以及好用的插件。

用 vim-plug 安装 coc.nvim : `Plug 'neoclide/coc.nvim', {'branch': 'release'}`

coc.nvim 你需要配置他的各种键位操作，比如 tab 选择补全栏目和错误跳转，因为设置太多了，我建议你直接复制粘贴，然后再来自行修改需要的键位：

```vimscript
"coc.nvim
let g:coc_global_extensions = [
	\ 'coc-diagnostic',
	\ 'coc-explorer',
	\ 'coc-gitignore',
	\ 'coc-html',
	\ 'coc-json',
	\ 'coc-lists',
	\ 'coc-prettier',
	\ 'coc-snippets',
	\ 'coc-syntax',
	\ 'coc-tasks',
	\ 'coc-translator',
	\ 'coc-vimlsp',
	\ 'coc-yaml',
	\ 'coc-yank']

inoremap <silent><expr> <TAB>
			\ pumvisible() ? "\<C-n>" :
			\ <SID>check_back_space() ? "\<TAB>" :
			\ coc#refresh()
inoremap <expr><S-TAB> pumvisible() ? "\<C-p>" : "\<C-h>"
inoremap <expr> <cr> complete_info()["selected"] != "-1" ? "\<C-y>" : "\<C-g>u\<CR>"

function! s:check_back_space() abort
	let col = col('.') - 1
	return !col || getline('.')[col - 1]  =~# '\s'
endfunction

function! Show_documentation()
	call CocActionAsync('highlight')
	if (index(['vim','help'], &filetype) >= 0)
		execute 'h '.expand('<cword>')
	else
		call CocAction('doHover')
	endif
endfunction

inoremap <silent><expr> <c-o> coc#refresh()
nnoremap <LEADER>h :call Show_documentation()<CR>
nnoremap <silent><nowait> <LEADER>d :CocList diagnostics<cr>
nmap <silent> <LEADER>- <Plug>(coc-diagnostic-prev)
nmap <silent> <LEADER>= <Plug>(coc-diagnostic-next)
nnoremap <c-c> :CocCommand<CR>
nnoremap <silent> <space>y :<C-u>CocList -A --normal yank<cr>
nmap <silent> gd <Plug>(coc-definition)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gr <Plug>(coc-references)
nmap <leader>rn <Plug>(coc-rename)
nmap tt :CocCommand explorer<CR>
nmap ts <Plug>(coc-translator-p)
imap <C-l> <Plug>(coc-snippets-expand)
vmap <C-e> <Plug>(coc-snippets-select)
let g:coc_snippet_next = '<c-e>'
let g:coc_snippet_prev = '<c-n>'
imap <C-e> <Plug>(coc-snippets-expand-jump)
let g:snips_author = 'Avimitin'
autocmd BufRead,BufNewFile tsconfig.json set filetype=jsonc

function! s:cocActionsOpenFromSelected(type) abort
	execute 'CocCommand actions.open ' . a:type
endfunction
xmap <leader>a  <Plug>(coc-codeaction-selected)
nmap <leader>aw  <Plug>(coc-codeaction-selected)
```

其中 coc_global_extension 是自定义的插件变量，可以在你每次迁移的时候自动安装列表里的所有插件，除此之外，你还可以输入 `:CocInstall xxx` 来安装 coc 插件，也可以输入 `:CocList extensions` 来查看插件仓库有什么好用的插件，带 * 号的就是已经安装好的插件。

在上面定义的大部分命令中，最好用的有几个快捷键：在普通模式中输入 gd，gy 会分别跳转到光标下的函数或者变量的定义和指向，gr 可以修改变量名字或者函数名字，空格加 `-` 或者 `=` 可以在错误信息之间跳转。ts 可以翻译光标下的英语单词。关于 coc 的 setting 由于篇幅就不在这里写了，我会找时间另开篇目。

### 咕咕咕

## 后话

我时常会听到有人说：你为啥要 vim，你为啥要用终端，IDE 他不香吗，GUI他不香吗。这些质问我到现在也没法给出很好的回应，因为说的确实不错：对于普通人来说，不要说 Jetbrains 的 IDE ，MicroSoft 的 Visual Studio Code 都已经能够完全的胜任个人项目或是团队合作项目了。一个如此简陋的需要耗费大量时间配置的软件，根本不是现代选择。如果你只是看到了一个漂亮的截图或者什么论坛帖子推荐，就想要耗费时间去折腾自己的 neovim，我也会建议你思考一下上面抛出的问题。虽说磨刀不误砍柴工，但是折腾 neovim ，是一个会耽误砍柴的历程。

我使用 neovim 的契机来自于一台 ChromeBook，因为经费不足但是又很有带笔记本出门的需求，我在诸多轻薄本中选中了 ChromeBook 13，一台 X86 平台的上网本。但是因为其羸弱的性能和便秘的散热，只是打开浏览器就能烫的脚发麻。于是我只好抛弃 IDE 开始使用 VSCode，啃各种 CLI 文档并慢慢习惯于各种终端操作。但是 ChromeBook 的触摸板也是极度的反人类，我每次移动光标都像回到了2008年第一次摸到黑莓的时光，在无奈之下我就开始寻找纯键盘和终端的解决方案。

在 B站 Up 主 TheCW 的熏陶下，我开始尝试学习使用 i3wm ，给 ChromeBook 刷上了 i3wm + Ubuntu 的发行版 Regolith，关于这个系统的配置可以参考：[如何配置 Ubuntu + i3wm](https://avimitin.com/system/regolith.html) 。逐渐习惯全快捷键的系统之后就自然而然地，开始走向魔改 neovim 的道路。

所以最后总结起来我愿意花时间魔改 neovim 的原因很简单，我想要一个能够随手启动，随手打开，速度飞快的编辑器，他还能拥有绝大多数 IDE 的功能，这些要求别的编辑器多多少少能做到一些，但是像 neovim 一样什么都有我全都要的，却只有 neovim 一个。那么你呢，你心中的答案能够值得你花这个时间去折腾吗。



