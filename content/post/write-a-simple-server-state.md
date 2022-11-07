+++
title = '边造车轮边学习'
date = '2022-11-07'
tag = ['react', 'ruby', 'typescript']
author = 'sh1marin'
+++
国庆几天把 Ruby，TypeScript 和 React 都粗粗浅浅地学了一下。
光看肯定不够，而学习新语言的一个办法之一就是造个轮子。
学 Ruby 的原因是想拿它来代替 Perl 和 BASH。
那有啥东西是又需要前端又和系统维护相关的呢？探针！我来写个
CPU 占用率的探针吧。用 Ruby 写脚本获取服务器 CPU 占用率，
前端用 React 写个展示页，完美的结合两个刚学的新知识。

* 前端的 Demo 可以看这里： <https://unmatched-status.sh1mar.in>，
* 整个项目的源码在这里：<https://github.com/Avimitin/uptime-collector>。

我把一些我写项目时用到的教程列在这里，这些教程写的都很好很详尽。

* [Ruby in Twenty Minutes](https://www.ruby-lang.org/en/documentation/quickstart/https://www.ruby-lang.org/en/documentation/quickstart/)
* [The TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
    * [学 TypeScript 之前先学 JavaScript](https://developer.mozilla.org/en-US/docs/Web/javascript)
* [React JS Tutorial](https://beta.reactjs.org/learn)

这篇博客主要记录一些写项目时学到的知识，不细谈语言本身的内容。

## 项目设计

我的想法是不要在服务器跑自己写的 daemon，服务器只跑一个 sshd。
然后再开一台机器，用 systemd timer 定时跑一个脚本，
这个脚本 ssh 到服务器上跑 uptime 获取 load 信息。
获取完信息后，把数据存进 sqlite，
再用一个独立的脚本和 systemd timer 跑数据导出和上传。

前端则纯静态托管在 CloudFlare 上，用 client side routing 做路由。
每次访问前端的时候从用户端向 GitHub 发出 raw 文件下载请求获取数据，
然后在用户端渲染图表。

这样的设计有几个好处：

1. sshd 稳
2. 数据获取出问题只需要到一台机器上找问题
3. 纯静态托管 CloudFlare 的好处不需要愁数据公开时的访问问题（带宽，DDOS...）
4. systemd-timer 比自己手写探测频率方便得多，也比 crontab 方便
5. 可以利用一些已经有的轮子，专注于语言的使用学习上

最后写出来的项目八百行出头，感觉还不错：

```text
===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 Ruby                    3          297          202           41           54
 TSX                     5          567          511            2           54
 TypeScript              4          117          105            2           10
===============================================================================
 Total                  12          981          818           45          118
===============================================================================
```

## 后端设计

### 数据获取

这个脚本需要做这几件事情：

1. 开 ssh 跑 uptime 获取返回值
2. 用 regex 拿到 load
3. 计算实际使用率并塞进 sqlite 数据库

Ruby 里面有一个很好用的 shell 交互语法糖，所以获取数据非常简单，
在本地配好 ssh config 之后直接在脚本里遍历机器地址，然后执行

```ruby
response=`ssh #{machine} uptime`
```

这里的 `response` 将会拿到远程机器的 uptime 命令输出的 stdout 值。
用反引号包住的字符会当做 command 传给 shell 执行，而 `#{machine}` 是
模板语法，类似于 JS 的 `${var}`，可以把 machine 变量拼接进字符串里。

Uptime 命令的输出类似于：

```text
13:05:59 up  3:24,  2 users,  load average: 1.42, 1.33, 1.33
```

其中 `2 users` 指代这台机器上目前已登录的用户数量，load average 后面
的浮点数是这台机器前 1，5，15 分钟的平均负载。这个负载是实际上是正在
等待 CPU 或者其他计算资源的程序数量。对于一个单核 CPU 的机器而言，
load 1 意味着整个 CPU 都在被占用，而对于四核 CPU 而言，load 1 意味着
75% 的时间 CPU 都在等待分配任务。

解压这些数据也很简单，ruby 自带 regex 支持，我用两个 capture groups
分别获取登录的用户数和前 5 分钟的平均 load

```ruby
result = /(?<user>\d+) users?,\s+load average: [\d.]+, (?<load>[\d.]+)/
        .match(response)

return [result['user'], result['load']]
```

Ruby 的 Regex 语法糖 `/{regex pattern}/` 可以创建一个新的 Regex object，
塞入刚刚获取的 response 字符串，因为我给这两个 capture groups 都起了别名，
match 方法会返回一个可以用字符串 index 的 capture groups object。

之后就可以下载 `ruby-sqlite3` 这个 gem 然后把获取到的数据加上时间戳塞进
数据库里了。

### 数据导出

数据导出需要做以下几件事情：

1. 从数据库获取数据
2. 过滤数据并格式化数据
3. 定时上传

获取数据就是一些基础的 SQL query，就不展开说了。在过滤日期的时候 SQLite 有一个
很好用的函数可以过滤掉非本月的记录：

```SQL
SELECT ttime, users, load
FROM record
WHERE machine=?
AND DATE(ttime, 'unixepoch')
BETWEEN
  DATE('now', 'start of month')
AND
  DATE('now', 'start of month', '+1 month', '-1 day')"
```

`DATE` 函数可以帮助调整日期 offset，而后面这一句 BETWEEN 可以很轻松的将日期调整到月初和月末来生成
日期限制条件。

因为计划是每隔 5 分钟获取一次数据，但实际使用的时候其实只需要一个数据
来代表这一天的使用量。这里我采用了 95th percentile 计算法。95th percentile
是一个在集合里大于其他 95% 数值的值。这个计算法常用在带宽计算上，因为它
去掉了 5% 极端数值，能相对准确的表现出一台机器最大需要多少的带宽，这个算法
能比较直观的给用户反馈一台机器某个使用量的最大值，方便做预算计划。

这里的计算也很简单，将输入数值集合排序，取数组长度 * 0.95 作为 index。

Ruby 的实现：

```ruby
sorted = records.sort_by do |rec|
  rec[:machine_load]
end
index = (sorted.length * 0.95).ceil - 1
return sorted[index][:machine_load]
```

`:machine_load` 是 Ruby 用来表达 hash key 的语法糖，records 是一组
序列化后的，以时间排序的 uptime 数据值。你可以把 `rec` 看成一个 HashMap。

我用 `:machine_load` 作为 key 重写排序，让记录以负载的大小来排序。
然后向上取整数组长度乘以 0.95 的结果作为 index 值。
Ruby 里的数组下标是从 0 开始的，所以要记得 -1 来获得正确的下标。
最后返回这个第 95% 位的值就行。

### 定时脚本

两个脚本写完之后就要开始~~献祭底裤了~~。systemd 有 timer 服务，
可以用来触发同名的 service 服务。比如假设 service 文件名为
`load-exporter.service`，可以用 `load-exporter.timer` 来触发这个服务。
这个 timer 文件支持类 crontab 的语法，而且因为是 systemd 的服务，
可以隔离环境，隔离不同的任务，还有 journalctl 查日志，
管理起来也非常方便，可以永远丢掉那个混乱且问题很多的 crontab 了。

因为脚本是执行一次获取一次获取一次类型的，这里用 oneshot 类型，
让 systemd 每次起服务执行完就结束服务。

```systemd
[Unit]
Description=The CPU usage fetcher
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/fetch
EnvironmentFile=/etc/fetcher/runtime_env

[Install]
WantedBy=multi-user.target
```

systemd 有 EnvironmentFile 属性，可以指定一个 env 文件，运行时将里面的键值对作为
环境变量。这样就可以开发时用 dotenv + 一个开发用 env 文件，生产环境让 systemd 准备
好生产环境需要的环境变量，无缝衔接非常方便。

timer 文件语法和 service 类似，但只需要设置好触发时机就行。

```systemd
[Unit]
Description=Scheduler for fetcher

[Timer]
OnCalendar=*:1/5
Persistent=true

[Install]
WantedBy=timers.target
```

`OnCalendar` 根据实际时间来触发，这里的 `*` 指代每一天，后面的 `1/5` 指的是每过五分钟触发一次。
`Persistent` 设置为 true 可以让服务在不知道上一次什么时候执行的时候立刻执行一次。

详细的时间解析语法可以看 [systemd.time(7)](https://man.archlinux.org/man/systemd.time.7#CALENDAR_EVENTS)。
systemd 的各种用法可以参考 [Arch Wiki](https://wiki.archlinux.org/title/Systemd)。

定时上传也很简单，参照定时获取再写一个新的 systemd service 就行。

## 前端

我使用的前端框架是 [vite](https://vitejs.dev/) + [React](https://reactjs.org/) + 
[react-router](https://reactrouter.com/en/main)，其中 vite 用来创建开发环境，
React 用来画前端，react-router 用来做客户端路由，数据获取则是用了 [SWR](https://swr.vercel.app/)
提供的 `useSWR` React Hook。

这里同样也不细谈他们的使用方法，这些框架的使用教程也很详尽很易读。
主要讲一下我在写 React 时写的一些错误。

### React Hook 调用顺序

在 React 里有个很重要的 hook 使用规则：hook 调用一定要在函数组件的最顶层。
因为 React 是通过 `useHook` 的调用顺序来判断每个 hook 的状态的。把 hook
调用放在顶层可以确保所有的 hook 依次执行，保证不会有任何的条件跳转或者
异常抛出，导致某个 hook 用到了其他 hook 的状态而产生渲染错误。

```javascript
useState(null)              //  1. 创建一个新的 state
useEffect((...) => { ... }) //  2. 创建一个新的 side effect
useState(selected)          //  3. 创建一个新的 state
```

像这样依次调用，React 可以将本地存储的状态和这些 hook 调用建立联系。
但是假如我写了一点条件跳转呢？

```
useState(null)
if (maybeFalse) {
  useEffect(...)
}
useState(selected)
```

如果 init 的时候 if 条件是 true，那么状态依旧是依次生成的，
但当某次渲染时条件为 false，那么第三行的 `useState` 就不能正确
读取自己的状态，而会读取到 useEffect 存的状态。

所以其实这个规则有另一个更精准的解读：一定要保证每个 hook 在每次
渲染都能被依次调用到。

### useState 存放 Map 类型的 state

在写数据日期选取的时候，因为不同的机器生成的数据时间不同，要给
每个机器都存一个自己的可选日期状态。比如 A 机器有 10 月和 11 月
的数据，B 机器只有 11 月的数据，混用 state 就会导致获取数据时出错。
（我就犯了这个错）

这个需求听起来就很键值对，机器名作为键，一组日期作为值。于是我就
想当然的写了下面这样的封装：

```typescript
function useDate(machID: string): [DateMenuOption | null, (opt: DateMenuOption) => void] {
  const [storage, update] = useState<Map<string, DateMenuOption>>(new Map());

  const setDate = (opt: DateMenuOption) => {
    const latest = storage.set(machID, opt);
    update(latest);
  }

  return [storage.get(machID) || null, setDate];
}
```

实际执行的时候发现 `setDate` 函数在每次更改菜单的时候都能触发，
但就是没有办法触发界面重渲染。

这是因为在 JavaScript 里，所有的 Object 都会传引用，而不是传值。
这里调用 `storage.set()` 拿到的返回值 `latest` 和 `storage` 是
同一个引用值。对于 React 而言，它只能看到用户传了一个和旧值
一模一样的引用，它并不知道 Map 里面的数据值变化了。

所以为了强制触发重渲染，这里可以靠创建新的 Map 并传递新的引用：

```diff
function useDate(machID: string): [DateMenuOption | null, (opt: DateMenuOption) => void] {
  const [storage, update] = useState<Map<string, DateMenuOption>>(new Map());

  const setDate = (opt: DateMenuOption) => {
-    const latest = storage.set(machID, opt);
+    storage.set(machID, opt);
-    update(latest);
+    update(new Map(storage))
  }

  return [storage.get(machID) || null, setDate];
}
```

Map 类的初始化函数可以接受一个 iterable 的对象用来作为初始化的
数据来源，我们可以先更新旧的 map，传给初始化函数创建新的 Map。

在用 Array 存 State 的时候则可以用 Array.prototype.slice() 创建一个
新的 Array Object。

### CSS 限制滚动条

默认不加限制的话，超出可显示范围的内容会延长网页。滚动的时候像
看一张长长的卷轴。但我想要画面保持固定长度，超出显示长度的页面依旧
可以滚动查看。但视觉效果上更加稳定，看起来就像在使用一个 App 一样。

实现这个也很简单，需要从 HTML 的根节点 `<body>` 开始加一个

```css
... {
  width: 100%;
  height: 100%;
}
```

的长宽高限制。渲染时这个值会换算成实际外部 container 的大小。
让这一个匹配上的节点大小和它父节点的大小保持一致。
一层一层向子节点限制，最后就能让各个组件的大小都不会超过可显示区域。

### 将 JSON 数据变成 Map

初学 JavaScript 的我因为没有认真研读 `JSON.parse` 的用法，先入为主
的认为这个就是一个 Deserialize 的函数，可以把 JSON 字符串反序列化我想要的类型。

实际是这样吗？

```typescript
const raw = '{ "a": "foo", "b": "bar", "c": "baz"}';
const val = JSON.parse(raw) as Map<string, string>;
console.log(val.get("a"));
```

上面这个代码块会报错 `val.get` is not a function。我对这个函数的期待
是像 Rust 的 `serde_json::from_str::<HashMap<String, String>>(...)` 那样
帮我把这一串 JSON 字符串反序列化成一个 Map，而 Map 是有 get 这个方法的。
这中间发生了什么事情呢？

实际上这部分就是我不学无术了，[`JSON.parse`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/parse#return_value)
会将字符串解析成 JavaScript 的 Object，Array...类型，但它并不接收泛型，
不会将 JSON 字符串反序列化用户强行 cast 的类型。

除此之外，`Map<string, string>` 也不是一个 Map 类型，而是一个函数类型，
在上面的例子里，我错误地将一个 Object 类型强行转换成了 Map 的初始化函数类型，
自然就不可能正常运行。

实际上想把 JSON Object 转换成 Map 应该使用 `Object.entries()` 将 object
转换成可以枚举的键值对属性，再用 `new Map()` 将这个属性值转换成 Map。
