---
title: 排序数列并进行排列比较 
date: 2020-08-10 21:13
categories:
- [coding, algorithm]
- [coding, java]
tags:
- java
- algorithm
---

# 排序数列并进行排列比较

## 题目

> 英文原题：Given two integer arrays of size **N**, design a subquadratic algorithm to determine whether one is a permutation of the other. That is, do they contain exactly the same entries but, possibly, in a different order.

给定两个相同大小且大小为N的数组，设计一个复杂度为平方级别的算法来判定两者是否全排列。

## 思路

对两个数组先进行排序，因为给定数组大小相同，假如排序之后的两个数组不是完全相等的就非全排列，反之则是。

## API设计

- **`public class Permutation`**

| 类型      | 方法                              | 描述                             |
| --------- | --------------------------------- | -------------------------------- |
| `boolean` | `less(int i, int j)`              | 返回 `i` 是否大于 `j`            |
| `void`    | `exch(int i, int j, int[] a)`     | 将 `int[i]` 和 `int[j]` 交换位置 |
| `boolean` | `isSorted(int[] a)`               | 判断 `a` 是否是顺序              |
| `int[]`   | `sort(int[] a)`                   | 排序 `a` 数组                    |
| `boolean` | `isPermutation(int[] a, int[] b)` | 判断 `a` 和 `b` 数组是否全排列。 |

- **`public class StopWatch`**

| 类型     | 方法        | 描述                                                         |
| -------- | ----------- | ------------------------------------------------------------ |
|          | `StopWatch` | 初始化变量 `start` ，赋值实例初始化时的系统时间。            |
| `double` | `stop()`    | 初始化 `now ` 变量并赋值当前系统时间，返回 `(now - start) / 1000.0 ` 的值 |

## 程序设计

- 设计排序

初始化变量 `h` 并以 `3 * h + 1`  增量。让 `a` 数组在间隔 `h`的所有值都是有序排列的，既分割多个 `h` 数组。然后缩小 `h` 的值将整个 `a` 数组进行完整排序。

由于 *Shell排序* 是基于插入排序的，当数组处于一定有序时插入排序速度会非常快。由于 `h` 增量的改变会大大影响 *Shell排序* 的速度，无法正确估计排序的复杂度，但是能够确定的是用 `3 * h + 1` 增量的 `h` 间隔排序的复杂度为
$$
O(N^\frac{1}{5})
$$

```java
// 这里使用 Shell 排序
public int[] sort(int[] a){
	int N = a.length;
    int h = 1;
    while (h < N/3) h = 3*h +1;
    while (h >= 1){
        for (int i = h; i < N; i++){
            for (int j = i; j >=h && less(a[j], a[j-h]); j-=h){
                exch(j, j-h, a);
            }
        }
        h/=3;
    }
    return a;
}
```

- 设计比较

遍历两个数组并比较两个数组是否元素相同即可

## 完整程序

```java
package pratise.ElementarySorts;

import edu.princeton.cs.algs4.StdOut;
import programTest.StopWatch;

public class Permutation {
    // Target: Confirm array a is permutation of array b
    public boolean less(int i, int j){
        return i < j;
    }
    
    public void exch(int i, int j, int[] a){
        int x = a[i];
        a[i] = a[j];
        a[j] = x;
    }

    public boolean isSorted(int[] a){
        for (int i = 1; i < a.length; i++){
            if (a[i] < a[i-1]){
                return false;
            }
        }
        return true;
    }

    public int[] sort(int[] a){
        int N = a.length;
        int h = 1;
        while (h < N/3) h = 3*h +1;
        while (h >= 1){
            for (int i = h; i < N; i++){
                for (int j = i; j >=h && less(a[j], a[j-h]); j-=h){
                    exch(j, j-h, a);
                }
            }

            h/=3;
        }
        return a;
    }

    public boolean isPermutation(int[] a, int[] b){
        // nonsense methods, comparing two array is enough
        int[] bigger;
        int[] smaller;
        if (a.length >= b.length){
            bigger = a;
            smaller = b;
        } else {
            bigger = b;
            smaller = a;
        }

        boolean val = false;

        for (int num : smaller){
            val = false;
            for (int num2 : bigger){
                if (num == num2){
                    val = true;
                    break;
                }
            }
        }
        return val;
    }

    public static void main(String[] args){
        // initialize
        int[] a = {3, 5, 8, 9, 2, 4, 1, 7, 6, 0};
        int[] b = {3, 5, 8, 9, 2, 4, 1};
        Permutation p = new Permutation();
        // sort and count time
        StopWatch s = new StopWatch();
        a = p.sort(a);
        b = p.sort(b);
        // judge
        if (p.isSorted(a)) StdOut.print("a IS SORTED\n");
        if (p.isSorted(b)) StdOut.print("b IS SORTED\n");
        if (p.isPermutation(a, b)) StdOut.print("a AND b IS PERMUTATION\n");
        // stop watch
        double cost = s.stop();
        StdOut.printf("SORT AND CONFIRM IS PERMUTATION COST %.2f SECONDS", cost);
    }
}
```

