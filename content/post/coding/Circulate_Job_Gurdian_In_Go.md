---
title: Golang Polling Control Flow
date: 2021-02-21
categories:
- useless
tags:
- golang
- goroutine
- polling
---

# Go 的循环任务守护

## 前言

这几天在做的一个项目里，有一个功能需求是：长期运行一个轮询任务，通过轮询得到的返回的信息进行下一步操作。这个问题初一看是蛮简单的，就一个 `for` 循环的事情嘛。但是实际上还有很多细节的地方需要仔细处理。本篇文章建立在你有足够的 Go 语言基础且懂得 `chan` 、 `select` 、`go` 关键字的用法的基础上。文章代码是逐步完善的伪代码，**请自行推演，不要复制粘贴**。

## 简单的无限循环

基于分治的思想，我们把这个任务给分成几个小任务：

- 执行一次请求
- 把一次请求改为轮询
- 对执行中的轮询进行控制

首先先写好请求的代码：

```go
package main

import (
    "net/http"
    "log"
    "fmt"
)

func Request(url string) ([]byte, error) {
    resp, err := http.Get(url)
    if err != nil {
        log.Println("[Error]Http get error,", err)
        return nil, err
    }
    return resp, nil
}

func main() {
    resp, err := Request()
    if err != nil {
        log.Println("[Error]Request error,", err)
        return
    }
    fmt.Println(string(resp))
}

```

执行一下，程序会发出一次请求，如果没有错误，打印返回的数据。

接着我们将这个请求改为无限请求来达成轮询：

```go
func main() {
    for {
        var url = "http://127.0.0.1:11451"
        resp, err := Request(url)
        if err != nil {
            log.Printf("request url %s: %v", url, err)
            continue
        }
        fmt.Println(string(resp))
    }
}
```

程序现在就会一直无限循环的发出请求了。请求的太快？我们加入一个 `sleep` ：

```go
import "time"

func main() {
    for {
        ...
        fmt.Println(string(resp))
        time.Sleep(1)
    }
}
```

好，一个简单的轮询就写完了。

## 轮询控制

上面那个简单的轮询，我们可以随时 `Ctrl+C` 停止。但是在实际生产环境中我们要保证程序尽量的 7x24h 在线，因此我们需要给这个轮询进行更多修改操作。比如我们新增一个变量 `pause` :

```go
package main

import (
	"fmt"
    "log"
    "net/http"
    "time"
)

var pause bool

func Pause() {
    pause = true
}

func Start() {
    pause = false
}

func Request() ([]byte, error) {
    // 同上
    ...
}

func main() {
    for {
        if pause {
            return
        }
        ...
    }
}
```

使用 `pause` 变量我们可以在程序内逻辑控制暂停了。但是不够即时，不管暂不暂停，我们都要先等上一次的请求结束之后才能判断 `pause` 变量。同时，因为我们用的是 `async` 服务器，意味着可能我们请求的服务不是即时返回的，在这种情况下，我们就要调用 Go 的关键字：`go` 协程 + `chan` channel 通信。

## 使用 Channel 进行流程控制

还是用分治的思想细化任务，我们要达成可控的轮询需要：

- 开辟一个新的协程来运行请求
- 新建一个通道来传输请求返回值
- 新建一个通道来传输暂停与否
- 同时为了让轮询的间隔也变得可控，我们再新建一个通道来控制间隔时间

建立一个新的协程非常容易，基于语言层面的支持，我们直接使用 `go` 关键字就可以开辟新的协程了。同时我们需要把轮询的代码移动出来，防止主进程结束导致子协程全部退出。

```go
package main

func Run() {
    for {
        go Request()
        ...
    }
}

func main() {
    Run()
}
```

然后我们要修改一下 `Request()` 的代码，让他能够通讯：

```go
func Request(response chan string) {
    resp, err := http.Get("127.0.0.1:11451")
    if err != nil {
        log.Println("[Error]Http get error,", err)
        response <- err.Error()
        return
    }
    response <- string(resp)
}
```

假设服务器只会返回 `done` 值：

```go
func Run() {
    resp := make(chan string)
    for {
        go Request(resp)
        for r := range resp {
            if r == "done" {
                log.Println("[Info]Request success")
                time.Sleep(1)
            } else {
                log.Println("[Error]", r)
            }
        }
    }
}

func main() {
    go func() {
        Run()
    }
    for i:=0; i<100; i++ {
        //做点什么来保持主进程
    }
}
```

然后我们就完成了用协程进行轮询的任务，同时整个任务都是在后台执行完毕的，我们也能获取到想要的消息。

接着实现暂停功能，这里我们要用上 `select` 关键字来控制通道：

```go
var stop chan bool

func Run() {
    resp := make(chan string)
    for {
        go Request(resp)
        select{
        case <- stop:
            return
        case r := resp:
            if r == done {
                log.Println("[Info]Request success")
                time.Sleep(1)
            } else {
                log.Println("[Error]Request fail")
                return
            }
        }
    }
}
```

因为 `select` 带来的通道响应的即时性，现在就算是正在请求中，我们也可以随时暂停程序了。

最后一步就是改造间隔时间，我们可以直接传参然后让程序 sleep。但是这样会导致一个情况：当你想要停止循环的时候，刚好这个循环正在睡眠，那么你就得等这个睡眠完全结束了程序才会停止，假如你设置了每隔2h轮询一次，那么最坏情况下循环要等2h之后才能收到你要停止的消息。

因此我们使用一个 `time` 包里的触发器 `time.Ticker` ，间隔一定时间发送一个信号，循环收到信号时再启动程序，这样也能实现间隔的效果，也能随时退出循环：

```go
package main

import (
	"time"
    "log"
    "net/http"
)

// 你还可以构建一个结构体把这些通道或是你项目需要的信息封装起来，实现更优雅的代码。
var stop chan bool

func Run(interval int) {
    period := time.Duration(interval) * time.Second
    t := time.NewTicker(period)
    for {
        select {
        // t.C 是一个 channel，会在给定的间隔时间发送一个 tick
        // 利用这个特性既能实现轮询冷却，又能保证轮询的控制性
        case <- t.C:
            log.Println("[Info]New request started")
            go Request()
            // 启动轮询之后就停止 ticker 
            t.Stop()
        case r := <- resp:
            if r == done {
                log.Println("[Info]Request success")
                // 一次请求成功之后重置定时器
                t.Reset(Period)
            } else {
                log.Println("[Error]Request fail")
                return
            }
        case <- stop:
            log.Println("[Info]loop stopped")
            return
        }
    }
}
```

## 总结

至此，我们实现了一个后台轮询的易用性和可控性。代码仅作抛砖引玉，实际操作上还有各种细节需要补充，比如对多个后台任务进行统一管理，以及错误出现后的自动恢复等等，就需要你自行根据情况进行设计。总而言之，多实践，在实践中学习，在学习中总结。
