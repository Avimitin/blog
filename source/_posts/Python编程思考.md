---
title: Python coding learning
date: 2020-05-10 17:49
categories:
	- [coding, python]
tags:
	- python
	- notes
---

# 转换思考方式压缩代码

## 多行`if...else...`

偶尔会遇到非常多的事项，需要进行条件判断，假如一行一行的 `if...elif...else`将会占用大量空间。但是其实可以利用 `Python` 的一个常用映射特性： `dict`，来解决多行 `if..else...` 的复杂度。

比如现在要求你写一个温度反馈程序，当温度低于0℃时，程序输出寒冷；当温度在0℃到10℃时，程序返回冰凉；当温度在10℃到20℃时，程序返回凉爽；当温度在20摄氏度到30摄氏度时，程序返回温暖；当温度在30到40摄氏度时，程序返回炎热。

如果是刚入门的我，会这么写：

```python
def temperature(temp):
    if temp <= 0:
        print('freeze')
    elif 0 < temp <= 10:
        print('cold')
    elif 10 < temp <= 20:
        print('calm')
    elif 20 < temp <= 30:
        print('warm')
    elif 30 < temp <= 40:
        print('hot')
    else:
        raise ValueError('illegal Value!')
```

这也是我一直以来的写法，对每一个数据进行条件判断。但是这样的写法又长又臭，而且假如现在突然间工作要求，让你把0-100℃，每隔10℃就输出一个判断，你就又要把原来的`if...else...`语句拆开补写，并增加了这串屎山。

所以在这里要非常推荐另外一种写法：利用Python独有的映射关键字：`dict`。

众所周知，字典的键可以存任意值，只要可以进行比较。我们可以利用这一特性，往里面存入变量和方法，来帮助降低难度。比如上面的程序我们可以写成这样：

```python
# 这里存放所有的范围和返回值
temperature_dict = {
    range(0, -100, -1): 'freeze',
    range(0, 10): 'cold',
    range(11, 20): 'calm',
    range(21, 30): 'warm',
    range(31, 40): 'hot'
}

# num就是我们要输入的数值
temp = -80
# key_list是存放在temperature里的所有键的列表
# 字典的key方法返回的是个键值对象，需要转换成列表
key_list = list(temperature_dict.keys())

# 然后遍历这个键值列表里所有的键，判断temp是否在键的范围里，
# 如果在，则返回键对应的值
for x in range(len(key_list)):
    if temp in key_list[x]:
        key = key_list[x]
        print(test_dict[key])
```

用上面的这个办法，就可以彻底压缩代码，需要执行输出的代码只有下面的`for`循环语句，一旦以后需要增改判断条件，也不需要拆开代码加`elif`，只需要维护一个字典即可。

---

同时值也可以存方法，也可以存入一个列表来执行多种操作，拿我的一个工作项目举例：

```python
# InlineKeyboardButton总控
call_func = {
    'reboot': [set_title, '正在重启程序'],
    'redo_title': [set_title, '请重新输入标题'],
    'set_dscp': [set_dscp, '请输入事故描述'],
    'redo_dscp': [set_dscp, '请重新输入事故描述'],
    'set_status': [set_status, '请设置一个事故状态'],
    'redo_status': [set_status, '请重新设置一个事故状态'],
    'set_component_id': [set_component_id, '请设置组件ID'],
    'redo_component_id': [set_component_id, '请重新设置组件ID'],
    'set_component_status': [set_component_status, '请设置组件的状态'],
    'redo_component_status': [set_component_status, '请重新设置组件的状态'],
    'final_check': [final_check, '请进行最后确认']
}


@bot.callback_query_handler(func=lambda call: True)
def check_title(call):
    if call.data in call_func.keys():
        bot.answer_callback_query(call.id, call_func[call.data][1])
        call_func[call.data][0](call.message)
    else:
        bot.send_message(call.message.chat.id, '输入错误')
```

`call_func`这个字典里的键，是程序可能会收到的消息。值则是收到消息时需要执行的方法和返回值的列表。当`check_title`这个方法接收到一个消息时，会使用 `in`关键字，判断收到的消息是否等于`call_func`字典里的键，如果`True`，则将执行后面列表里的方法，并将列表的文字输出。

比如当程序收到`'reboot'`这个消息时，将消息与`call_func`的键比对，比对到`reboot`这个键，于是将调用`call_func['reboot'][1]`这个值，也就是对应的`'正在重启程序'`。然后再执行列表第0项的方法`set_title`。因为`set_title`变量只是一个`set_title`方法的实例，所以需要加上括号并输入参数来执行对应的方法。

如果我不用字典的话，则会变成下面这样的屎山：

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20200510170307.png)

一旦我再需要处理别的信息，就又要一条一条拆`if...else...`并且反复写一样的方法。维护成本奇高。使用字典我就可以简简单单的增删改查。

## 简单的列表多变量赋值

假如我现在有一个列表，里面存着4个值，然后要把列表里的4个值都分别赋给不同的变量，初学的我就会：

```python
test_list = [1, 2, 3, 4]
test1 = test_list[0]
test2 = test_list[1]
test3 = test_list[2]
test4 = test_list[3]
print('数字分别为{0}{1}{2}{3}').format(test1, test2, test3, test4)
```

但是其实上面这个程序，只需要三行就行：

```python
test_list = [1, 2, 3, 4]
# 需要注意，这里的变量一旦少于或者多余列表存在的值会报错
test1, test2, test3, test4 = test_list
print('数字分别为{0}{1}{2}{3}').format(test1, test2, test3, test4)
```

Python 就是有这么好用的特性，让代码看起来更加简洁并减少工作量。基于这种特性，我可以把原来这个：

![](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20200510173820.png)

变成更加简洁的程序：

![只需要两行代码](https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20200510173901.png)

甚至可以利用这种特性，减少更多工作量：

```python
# 比如现在有这么一串每月收入数据
month = [100, 50, 60, 80, 200, 90]
# 我只想要最后一个月和前面几个月的平均值
# 可以用*变量名来融入更多数据，与函数的参数相同
*before, last = month
avg_before = sum(before) / len(before)
print('前几月平均收入：{}，这个月的收入：{}'.format(avg_before, last))
```

非常有趣不是吗，了解了Python的许多特性之后就会发现Python比起别的静态语言少了很多造轮子的枯燥代码，作为工具而言非常得心应手。
