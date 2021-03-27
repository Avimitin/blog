---
title: 二叉查找树（Binary Search Tree）
date: 2021-03-27 11:02
categories:
	- [coding, algorithms]
	- [coding, c]
tags:
	- coding
	- c
	- algorithms
---

## 概念

二叉查找树是二叉树，树中的每个节点都是可以比较的对象，节点中都存放着一组数据。且每个节点的键都大于它左边子树的任意节点，小于其右边子树的任意节点。

## 数据表示

每个节点含有一个键，一个值，一条左键和一个右键，以及一个以该节点为根的所有子树的节点总和的计数器。

```c
typedef struct node {
  KeyType  key;
  ValType  val;
  NodeType left;
  NodeType right;
  int      N
} NodeType;
```

> - 左链指向左子树，右链指向右子树
> - 计数器 `N = size(x.left) + size(x.right) + 1` （1是x自己）

## 递归

把递归看作是在移动节点，每次递归都是在把根节点向下移动。

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20201102151820.png)

## 查找

如果树是空的，则查找未命中，如果查找的键等于根节点的键，则命中，如果小于，则从根节点左边递归开始查找，反之则从右边。

## 插入

如果树是空的，则直接插入新节点，如果插入的键等于根节点的键，则更新键的值。如果插入的键小于根节点的键，则递归的从左子树插入，大于则从右子树，然后更新计数器的值。

```c
node *put(NodeType *x, KeyType *key, ValType *val) {
    if (x == NULL) {
        return NewNode(key, val, 1);
    }
    int cmp = Compare(key, x->key);
    if		(cmp < 0) {	x->left = put(x->left, key, val);	}
    else if (cmp > 0) {	x->right = put(x->right, key, val); }
    else {
        x->val = val;
    }
    x->N = size(x->left) + size(x->right) + 1;
    return x;
}
```

