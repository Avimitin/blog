---
title: JavaScript 学习笔记
date: 2021-04-12 20:14
categories:
	- [coding, js]
tag:
	- js
	- javascript
---

## JavaScript 是什么

JavaScript 就是一种脚本编程语言，能帮助在网页上实现各种复杂的操作，比如页面实时刷新和动画。它包含常规编程的特性，同时可以利用许多 API 接口实现有用的功能。

## JavaScript 是如何工作的

浏览器在读取一个网页时，HTML，CSS， JavaScript 会在一个浏览器标签页内执行，把源代码加工成一个网页。在 HTML 和 CSS 加载完毕后，浏览器的 JS 引擎会执行 JavaScript 代码。在网页的结构和样式就位之后再执行，可以防止引发找不到元素的错误。

> 每个标签页就是一个独立的运行环境，一个标签页内的代码不会影响别的标签页的代码。

### JavaScript 执行次序

浏览器会从上自下执行浏览器代码，假如在声明变量之前调用，会发生对象未声明的错误。

### 解释与编译代码

JavaScript 是轻量级的解释语言，虽然浏览器在运行 JS 代码时会把它编译成二进制形式，但是编译在运行时发生，所以 JS 还是一门解释语言。

### 动态与静态代码

动态代码即页面会随时根据不同的数据进行页面更新，而静态页面指页面显示内容永远不会变。

## 如何添加一段 JavaScript 代码

可以像 CSS 那样将 JS 添加进 HTML， 使用 `<script>` 元素即可嵌入。

### 内部 JS

可以在 `<body>` 标签之前插入 `<script>` 标签来嵌入 JS 代码。比如以下例子：

```html
<!DOCTYPE html>
<html lang="zh-Hans">
  <head>
    <meta charset="utf-8">
    <title>使用 JavaScript 的示例</title>
  </head>
  <body>
    <button id="btn-exp">点我呀</button>
    <script>
      const btn = document.getElementById("btn-exp");
      function changeText() {
        btn.textContent = "你点了我！";
      }
      btn.addEventListener("click", changeText);
   </script>
  </body>
</html>
```

### 外部 JS

在和 HTML文件同个路径中，把 JS 代码放入一个后缀为 `.js` 的文件里，然后在 HTML 文件中，把 `<script>` 替换成:

```html
<script src="script.js" async></script>
```

刷新页面后会发现页面实现效果是完全一样的，同时 HTML 代码也变得更加有序易读。

### 内联 JS

可以在 HTML 页面中直接调用 JS 代码：

`HTML` :

```html
<button onclick="changedText()">
  点我鸭！
</button>
```

`JavaScript`:

```javascript
function changeText() {
  btn.textContent = "你点了我！";
}
```

这样写和之前实现的作用是一模一样的，只是在 `<button>` 元素中内联的处理器。

**但是这样非常不好**，一，这使得 HTML 代码维护成本升高。二，这样实现的 HTML 代码效率低下，假设页面有多个按钮，不可能去一个一个的给 `<button>` 元素添加属性。

### 脚本调用策略

HTML 元素是按照出现的次序依次进行加载的，如果 JS 代码在 HTML 元素之前先加载，则代码会出错。比如以下例子：

`example.html`:

```html
<!DOCTYPE html>
<html lang="zh-Hans">
  <head>
    <meta charset="utf-8">
    <title>使用 JavaScript 的示例</title>
    <script src="example.js"></script>
  </head>
  <body>
    <button id="btn-exp">点我呀</button>
  </body>
</html>
```

`example.js`:

```javascript
const btn = document.getElementById("btn-exp");
function changeText() {
  btn.textContent = "你点了我！";
}
btn.addEventListener("click", changeText);
```

这个例子与 [内部 JS](#内部 JS) 一节中的例子作用是相同的，但是 `<script>` 标签在 `<button>` 标签之前出现了，这会导致 `btn` 这个变量找不到 `id="btn-exp"`  的元素。

旧的方法如内部 JS 一节所示，把 `<script>` 标签放在依赖的元素之后即可，但是对于现代网页而言，可能会有大量 JavaScript 需要编译执行，如果放在 DOM 元素之后进行加载可能会带来肉眼可见的巨大损耗。

现代技术中，可以在 `<script>`  元素中写入 `async` 属性来进行异步加载来解决这个问题：

```html
<script src="example.js" async></script>
```

### async 和 defer

脚本阻塞问题有两种解决方案: `async` 和 `defer` 

`async` 用在脚本调用没有次序要求时的情况，`defer` 用在脚本调用有次序要求的情况。比如以下例子:

```html
<script async src="js/vendor/jquery.js"></script>

<script async src="js/script2.js"></script>

<script async src="js/script3.js"></script>
```

如果使用 `async` 属性，三个 JS 脚本的调用次序是完全不定的，`jquery.js` 完全有可能在下面两个脚本之后再加载，那么假如 `script2.js` 或 `script3.js` 依赖 `jquery.js` ，页面加载就会出错，像这种情况，就可以使用 `defer`  属性：

```html
<script defer src="js/vendor/jquery.js"></script>

<script defer src="js/script2.js"></script>

<script defer src="js/script3.js"></script>
```

上面的例子，三个脚本会依照次序一个一个加载。

所以当你的脚本不需要等页面解析，也不依赖别的脚本，就用 `async` 属性，如果脚本要等待页面解析，又依赖别的脚本，就用 `defer` 属性。

### 注释

JavaScript 的注释类似于 C 的风格:

```javascript
// 这是单行注释

/*
这是多行
注释
*/
```

