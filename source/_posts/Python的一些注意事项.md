---
title: Python Coding Notice
date: 2020-04-04 15:51
categories:
	- [coding, python]
tags:
	- python
	- notes
---

# 易错易忘点

~~每周更新新的易错知识点，巩固记忆。~~

不写 Python 了, 停更

## 语法

### 随机数生成

- 随机数生成需要`import random`这个模块，使用`random.randrange(int值)`来获取指定范围内的数值。

> Example:

```python
import random
···
ran_num = random.randrange(3)
print(ran_num)
```

### 列表元组和字典

- 列表是可变化的，元组是不可变化的列表。
- 列表第一个元素是`list[0]`，最后一个元素是`list[-1]`
- 字典是用`{key:value}`来对应生成的列表。列表使用中括号：`list = ["abc","123",456]`，字典使用大括号:`dic = {value1:"abc",value2:"123"}`。
- **列表元素可重复，字典键值不可重复。**若有重复的字典元素，将会由后面的元素替代指向前面的元素。

> Example:

```python
list1 = ["abc",'abc','123',321]
>>> list1
['abc','abc','123',321]
dic1 = {name:'zhangsan',age:30,name:"lisi"}
>>> print(dic1['name'])
lisi
```

### 键盘输入输出

- `input`默认输入String类型，有时候需要记得强制转换类型。
- 格式化输出`format`的用法就是使用大括号来占位，使用后面的括号内的元素替代。

> Example:

```python
>>> print("我的名字叫{},年龄是{}".format("李华","19"))
我的名字叫李华，年龄是19
>>> print("我的名字叫{0},年龄是{1}".format("李华","19"))
我的名字叫李华，年龄是19
>>> print("我的名字叫{1},年龄是{0}".format("李华","19"))
我的名字叫19，年龄是李华
>>> print("我的名字叫{name},年龄是{age}".format(name="李华",age="19"))
我的名字叫19，年龄是李华
```

- 由于`open`方法打开文件需要`close`，而使用close会经常蹦出奇怪的错误且占用一行，所以常用`with open() as xxx`语句来自动关闭文件。

> Example:

```python
"""
假设要创建一个文件叫test.txt，打开并写入:
Hello
World
然后再读取并输出到终端
"""
# 打开并读写文件，W+模式用来新建并读写文件
with open("test.txt","w+") as test_file:
    test_file.write("Hello\nWorld")
    
# 打开并读取文件，输出到终端，r+用来读写文件，指针置于文件开头
with open("test.txt","r+") as test_file:
    test_words = test_file.read()
    print(test_word)
```



### While循环

- `break`直接跳出循环，`continue`回到循环开头

### 异常与错误

- 使用`try`语句来尝试执行某语句，假如语句返回异常则进行`except`中的错误值匹配判断，假如返回`True`则执行`except`中的语句，假如返回`False`则中断并退出程序。
- 类型错误是`ValueError`,不是`TypeError`。

### 迭代器

- 迭代器执行完之后不可以再继续读取执行

### 函数

- Python中的赋值是将某个变量指向一个对象，变量本身不会变动。

> Example

```python
"""
Python只引用对象，赋值本身属于将变量指向对象
a本身是没有任何类型的
"""
a = 2  # 将a变量指向int 2对象
a = 3  # 将a变量指向int 3对象


def change_num(b):
    b = 10
    return b


print(change_num(a))  # a = 10(将a指向了10）
change_num(a)
print(a)  # a = 3 （a本身不改变，还是指向3）

```

- `def`定义函数时，参数有标识的传参可以无所谓顺序。
- 默认参数需要写在最后面。
- *号后的不定长参数默认传元组。

```python
def test(a, b, c, *var):
    """*号后传入的数据指向元组类
    所以如果想加多余值不可以直接sum = aa + b + vd + var
    需要使用循环语句将元组内数字传出"""
    print(a + b + c)
    print(var)
    count = len(var)
    num = 0
    while count != 0:
        num += var[count-1]
        count -= 1
    sum = a + b + c + num
    return sum
```

### YAML和JSON

- 都需要导入模块才能使用，YAML需要额外安装`PyYaml`
- JSON需要学习，但是推荐使用YAML，简单强大不易错误。
- 使用`json.dumps`把Python转换成JSON格式，使用`json.loads`把JSON文件转换成Python可识别格式。
- 使用`yaml.dumps`把Python转换成YAML格式，使用`yaml.load`把YAML文件转换成Python可识别格式。
- 读取YAML文件时可能会出现安全性错误，需要在load方法中传入参数`Loader=yaml.FullLoader`

### 面向对象

- 类中的变量本身可以被其他函数改变值
- 当想使用普通函数时需要用装饰器`@staticmethod`
