---
title: Solving Digital Roots
date: 2021-02-13 18:17
tags:
	- coding
	- algorithm
	- golang
thumbnail: https://cdn.jsdelivr.net/gh/Avimitin/PicStorage/pic/20200810211539.png
tnalt: "random photo"
tldr: "using TDD to solve the digital roots problem"
---

# Sum of Digits / Digital Root

## Detail

[Digital root](https://en.wikipedia.org/wiki/Digital_root) is the *recursive sum of all the digits in a number.*

Given `n`, take the sum of the digits of `n`.  If that value has more than one digit, continue reducing in this way  until a single-digit number is produced. The input will be a non-negative integer.

For example:

```text
    16  -->  1 + 6 = 7
   942  -->  9 + 4 + 2 = 15  -->  1 + 5 = 6
132189  -->  1 + 3 + 2 + 1 + 8 + 9 = 24  -->  2 + 4 = 6
493193  -->  4 + 9 + 3 + 1 + 9 + 3 = 29  -->  2 + 9 = 11  -->  1 + 1 = 2
```

## Ideas

首先基于分治思想，把这个问题分成几个小的部分：

```text
分离各数位上的数字 -> 求和 -> 检查是否不为单位数
```

## Solution

首先细分第一个任务：分离数字。我的思路是把每个数位上的第一个数字给抽取出来，放进一个数组里，然后将其减少为剩下的数位，继续这样的循环直到数字降到最低一位。比如对于数字 114514，先去 1 塞进数组，然后令数字等于 14514，继续遍历。

根据以上思路，第一个任务可以细分为：抽取首位，塞入数组，减少数字大小。

## 抽取首位

### Test first

基于 TDD 的思想，我先写测试 `TestGetBit` , 测试对于数字 16 能不能正确取得首位 1:

```go
func TestGetDigit(t *testing.T) {
	t.Run("get 16", func(t *testing.T) {
		got := getDigit(16)
		want := 1
		if got != want {
			t.Errorf("got %d want %d", got, want)
		}
	})
}
```

运行一下测试：

```text
./DigitalRoot_test.go: undefined: getDigit
```

### Then code

然后再来写函数 `getDigit` ：

```go
func getDigit(n int) int {
    return 1
}
```

再来跑一次测试，测试应该能够通过了。于是我们再写后面的测试，对于输入 6 能不能获得首位 6：

```go
t.Run("get 6", func(t *testing.T) {
		got := getDigit(6)
		want := 6
		if got != want {
			t.Errorf("got %d want %d", got, want)
		}
})
```

然后重新跑测试，结果自然会错误：

```text
./DigitalRoot_test.go: got 1 want 6
```

### Refactor

然后要对代码进行重构：想要对任何数字都能取得第一位，我们只需要把他和10的幂运算后的结果相除即可，比如对于数 `1,919,810`，可以将其除以 10 的 6 次方，也就是 `1,000,000` 并舍弃余数就能获得结果 1 。所以在重构 `getDigit` 前，需要先写一个函数来进行对 10 的幂运算。

## 幂运算

### Test first

先写测试 `TestPow`:

```go
func TestPow(t *testing.T) {
	got := pow(6)
	want := 1000000
	if got != want {
		t.Errorf("got %d want %d", got, want)
	}
}
```

跑一下测试：

```test
./DigitalRoot_test.go: undefined: pow
```

### Then code

然后写函数

```go
func pow(p int) int {
	var pow = 10
	if p == 0 {
		return 1
	}
	for p > 1 {
		pow *= 10
		p--
	}
	return pow
}
```

当 `p = 0` 也就是十的 0 次方时，我们返回 1。如果输入是正的，我们不断将 pow 变量和 10 相乘并递减输入。由于题干给出，输入一定是正整数，所以我们不需要考虑 `p<0` 的情况。

### Refactor

然后对函数 `getDigit` 进行重构，多输入一个 e 来获得这个数有几位，且因为对于 7 位的数字我们只需要求 10 的 6 倍即可，因此要先把 `e -= 1`：

```go
func getDigit(n int, e int) int {
	e -= 1
	pow := pow(e)
	return n / pow
}
```

同时对测试进行完整修改补充：

```go
func TestGetDigit(t *testing.T) {
	t.Run("get 16", func(t *testing.T) {
		got := getDigit(16, 2)
		want := 1
		if got != want {
			t.Errorf("got %d want %d", got, want)
		}
	})

	t.Run("get 114514", func(t *testing.T) {
		got := getDigit(114514, 6)
		want := 1
		if got != want {
			t.Errorf("got %d want %d", got, want)
		}
	})

	t.Run("get 6", func(t *testing.T) {
		got := getDigit(6, 1)
		want := 6
		if got != want {
			t.Errorf("got %d want %d", got, want)
		}
	})
}
```

然后跑一下测试，测试就应该是能通过的了。

然后我们就需要知道每个输入都有多少位，所以需要写一个 `count` 函数来获得数位。

## 计算数位

### Test first

二话不说，先写测试：

```go
func TestCount(t *testing.T) {
	got := count(114514)
	want := 6
	if got != want {
		t.Errorf("got %d want %d", got, want)
	}
}
```

### Then code

```go
func count(n int) int {
	var digitCount int
	for n > 0 {
		n /= 10
		digitCount++
	}
	return digitCount
}
```

不断令 n 退 1 位直到其不为正整数，并在每次遍历时递增变量 `digitCount` 就能获得数字的位数。

### Refactor

加个注释？

```go
// count return the number of input n
func count(n int) int {
	var digitCount int
	for n > 0 {
		n /= 10
		digitCount++
	}
	return digitCount
}
```

## 获得余数

### Test first

```go
func TestGetPart(t *testing.T) {
	t.Run("get part 114514", func(t *testing.T) {
		got := getPart(114514)
		want := 14514
		if got != want {
			t.Errorf("got %d want %d", got, want)
		}
	})
}
```

### Then code

```go
func getPart(n int) int {
	return 14514
}
```

### Refactor

重新写测试，让原方法失败

```go
func TestGetPart(t *testing.T) {
	t.Run("get part 114514", func(t *testing.T) {
		got := getPart(114514)
		want := 14514
		if got != want {
			t.Errorf("got %d want %d", got, want)
		}
	})

	t.Run("get part 16", func(t *testing.T) {
		got := getPart(16)
		want := 6
		if got != want {
			t.Errorf("got %d want %d", got, want)
		}
	})
}
```

对于任何正整数输入，可以对其进行取余操作来获得他除首位以外的数位。比如对于输入 `1,919,810`，可以对其除以 `1,000,000` ，余数 `919,810` 就是我们需要的值。

```go
func getPart(n int, bit int) int {
	return n % pow(bit-1)
}
```

## 插入数组

然后我们把这些数分别插入数组，还是先写测试：

### Test first

```go
func TestInsertDigit(t *testing.T) {
	t.Run("get 16", func(t *testing.T) {
		got := insertDigit(16, 2)
		want := []int{1, 6}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v want %v", got, want)
		}
	})
	t.Run("get 114514", func(t *testing.T) {
		got := insertDigit(114514, 6)
		want := []int{1, 1, 4, 5, 1, 4}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v want %v", got, want)
		}
	})
}
```

运行上面的测试，使用 `reflect` 包的 `DeepEqual` 函数来检查输出数组是否时我们想要的。

### Then code

先初始化一个数组，然后我们利用上面写好的函数，抽取首位插入数组，将余数赋值 n，循环至位数降至 0：

```go
func insertDigit(n int, amount int) []int {
	var digits []int
	for amount > 0 {
		i := getDigit(n, amount)
		n = getPart(n, amount)
		amount--
		digits = append(digits, i)
	}
	return digits
}
```

### Refactor

因为每次 `getDigit` 和 `getPart` 都需要循环进行一次 10 的幂运算，所以我们可以重构一下比较占用时间的这一部分，把原本各函数的计算提取出来只计算一次，并把这两个函数的第二个参数改为传入 10 幂运算后的值：

```go
func insertDigit(n int, amount int) []int {
	var digits []int
    divisor := pow(amount - 1)
	for amount > 0 {
		i := getDigit(n, divisor)
		n = getPart(n, divisor)
		amount--
		digits = append(digits, i)
	}
	return digits
}
```

## 主函数

### Test first

```go
func TestDigitRoot(t *testing.T) {
	t.Run("test 16", func(t *testing.T) {
		got := DigitalRoot(16)
		want := 7

		if got != want {
			t.Errorf("got %d want %d", got, want)
		}
	})
}
```

### Then code

首先我们需要检测输入有几位，然后传到 `insertDigit` 函数里来获得每一位的值，最后再遍历求值：

```go
func DigitalRoot(n int) int {
    cm := countNum(n) {
	digits := insertDigit(n, cm)
	n = sum(digits)
	return n
}
```

### Refactor

然后加入更多测试：

```go
	t.Run("942", func(t *testing.T) {
		got := DigitalRoot(942)
		want := 6

		if got != want {
			t.Errorf("got %d want %d", got, want)
		}
	})
	t.Run("132189", func(t *testing.T) {
		got := DigitalRoot(132189)
		want := 6

		if got != want {
			t.Errorf("got %d want %d", got, want)
		}
	})
```

毫不意外，这个测试一定会 failed，因为我们没有把求和完的值继续分开计算，所以继续重构：

```go
func DigitalRoot(n int) int {
	for cm := countNum(n); cm > 1; cm = countNum(n) {
		digits := insertDigit(n, cm)
		n = sum(digits)
	}
	return n
}
```

把输入进行数位计算，只要输入大于1就分离求和，并把求和结果重新赋值回变量 n，直到 n 的数位降至 1，返回结果。

## 最后

这是我第一次基于 TDD 写算法题，出乎意料的是，TDD 不仅没有降低写题速度，甚至帮助我避免了大量简单错误，而且基于 TDD，我也不需要架上 "BREAK POINTS MACHINE GUN" 在 IDE 打上密密麻麻的红点来判断逻辑错误，TDD 帮助了我很多，所以这篇文章我也基于 TDD 的写法，希望能带更多人入坑。
