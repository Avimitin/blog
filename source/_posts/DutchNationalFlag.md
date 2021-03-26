---
title: Solving Dutch National Flag
date: 2020-08-12 23:29
categories:
	- [coding, algorithm]
	- [coding, java]
tags:
	- coding
	- algorithm
	- java
thumbnail: "https://gss0.baidu.com/-vo3dSag_xI4khGko9WTAnF6hhy/zhidao/wh%3D600%2C800/sign=94eeecbbd5a20cf446c5f6d94639670e/f603918fa0ec08fa9c51789b5cee3d6d55fbdabd.jpg"
tnalt: "dutch flag"
tldr: "using java to solve the dutch national flag problem"
---

# DutchNationalFlag

## 题目

> Given an array of *n* buckets, each containing a red, white, or blue pebble, sort them by color.

大意：给定一个 *n* 大小的数组，里面包含了无序的红白蓝三种颜色，将其以颜色为组进行排序。

e.g: `int[] array = {r, w, b, w, r, w, b, r}`

需要将上述数组过滤成 `{r, r, r, w, w, w, b, b}`

## 要求

复杂度至多为：
$$
O(n)
$$
内存占用至多为 *n* 。

## API

| 类型   | 方法                       | 描述                           |
| ------ | -------------------------- | ------------------------------ |
|        | `DutchNationalFlag(int N)` | 初始化数组                     |
| `void` | `swap(int i, int j)`       | 交换对应参数角标的数组的两个值 |
| `void` | `color(int i)`             | 生成数组元素颜色               |

## 思路

由于只有红白蓝三种颜色，使红白蓝分别赋值为 0, 1, 2。然后排序数组即可。

---

由于需要保持在 `O(n)` 复杂度，只能遍历一次，因此使用类似于 *3way-quicksort* 的方式来排序。把数组分为三个部分，前面都是红，后面全是蓝，因此遍历数组遇到红的就往前移，遇到蓝的就往后移动。剩下中间的就全部是白色了。

## 程序设计

设置指针 `begin`, `end`, `current`。其中 `begin` 为数组首节点，`end` 为数组尾节点，`current` 为遍历指针。将 `current` 放在首节点，不断往后遍历。

```java
// 初始化三个指针
int begin = 0;
int end = d.len -1;
int current = 0;
```

- 当 `current` 遇到红色时，将其与 `begin` 交换位置，然后把 `begin`  往前进一位，因此 `begin` 指针之前的所有节点都会是红色。
- 当 `current` 遇到白色时，只需继续遍历下一次循环，因为排序完白色一定在中间。
- 当 `current` 遇到蓝色时，将其与 `end` 交换位置，然后把 `end`  往后退一位，因此 `end` 之后的所有节点都会是蓝色。但是**不能**直接进行下一轮循环，因为你并不知道换过来的 `end` 节点是什么颜色，可能会打乱前面的顺序。让 `current` 指针不动，再进行一次条件判断来排序红色和白色。

```java
while(current <= end){
    // 遍历判断 array 数组里的值
    switch (d.a[current]){
        case RED:
            // 遇到红色交换 current 和 begin 对应值并同时进一位
            d.swap(current++, begin++);
            continue;
        case WHITE:
            // 遇到白色 current 指针直接进一位
            current++;
            continue;
        case BLUE:
            // 遇到蓝色 end 指针退一位， current 指针不动再进行一次条件判断
            d.swap(current, end--);
            continue;
    }
}
```

- 图解

![](http://lh4.ggpht.com/_VYXCTGdavk8/TWctp9wB7uI/AAAAAAAAAEI/2D9HvA9pGno/[UNSET].jpg)

## 完整程序

```JAVA
package pratise.ElementarySorts;
// 导入的是随机类和输出类，不影响算法
import edu.princeton.cs.algs4.StdRandom;
import edu.princeton.cs.algs4.StdOut;

public class DutchNationalFlag {
    private static final int RED = 0;
    private static final int WHITE = 1;
    private static final int BLUE = 2;
    private int[] a;
    private int len;

    public DutchNationalFlag(int N){
        a = new int[N];
        len = N;
    }

    public void swap(int i, int j){
        int pebble = a[i];
        a[i] = a[j];
        a[j] = pebble;
    }

    public void color(int i){
        a[i] = StdRandom.uniform(3);
    }

    public static void main(String[] args){
        DutchNationalFlag d = new DutchNationalFlag(20);
        // 往数组塞东西
        for (int i = 0; i < d.len; i++){
            d.color(i);
            StdOut.print(d.a[i]);
        }
        StdOut.print("\n");
        // 初始化三个指针
        int begin = 0;
        int end = d.len -1;
        int current = 0;
        // 过滤
        while(current <= end){
            // 遍历判断 array 数组里的值
            switch (d.a[current]){
                case RED:
                    // 遇到红色交换 current 和 begin 对应值并同时进一位
                    d.swap(current++, begin++);
                    continue;
                case WHITE:
                    // 遇到白色 current 指针直接进一位
                    current++;
                    continue;
                case BLUE:
                    // 遇到蓝色 end 指针退一位， current 指针不动再进行一次条件判断
                    d.swap(current, end--);
                    continue;
            }
        }
        for (int i = 0; i < d.len; i++){
            StdOut.print(d.a[i]);
        }
    }
}
```

