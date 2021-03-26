# 使用 Python 实现 JSON Web Token

## 前言

涉及到无状态鉴权时，`Token` 是比较好的验证用户身份的方案，而 `Token` 如何制作也是一个设计难点。只是一个 `Token` 去下载模块又增加依赖的负担，于是我去学习了 `JWT(JSON Web Token)` 的设计方式。

<!--more-->

## JWT 简介

> `JWT (JSON Web Token)` 是一个定义安全信息传输的公开标准([RFC 7519](https://tools.ietf.org/html/rfc7519))。使用 `JWT` 标准设计的信息因为其数字签名，都可被验证和信任。

`JWT` 使用 `HMAC` 算法或者一个 `RSA` 或 `ECDSA` 加密的公/私钥来进行签名认证。使用 `JWT` 可以帮助实现无状态鉴权和 `OAuth 2.0` 的实现，也可以用来传输信息。

## JWT 结构

`JWT` 由三部分组成：

- Header
- Payload
- Signature

分别用点 `.` 隔开组合，一个 JWT 应该长得像下面这样:

`xxxx.yyyy.zzzz`

### Header

`header` 内包含两个信息，一般来说都应该是相同的，用来告示自己使用的算法如 `HS256` 。下面是一个 `header` 的例子：

```json
{
    "alg": "HS256",
    "typ": "JWT"
}
```

**然后使用 Base64Url 将这个 `JSON` 编码。**

### Payload

Payload 用来装载关于比如用户的信息或其他更多信息，包含三部分：

- Registered claims: 这里是对 `Token` 的签发的说明，如 `Token` 签发方，`Token` 签发时间，有效时间等等。可以在 [这里](https://tools.ietf.org/html/rfc7519#section-4.1) 看到更多关于 *Registered claims* 名词的说明。
- Public claims: 这里可以由签发人随意定义，但是要尽量避开 `Registered claims` 所用到的专有名词。
- Private claims: 这里可以装载想要使用 `Token` 分享的信息。

下面是关于 `Payload` 的例子：

```json
{
    "iss": "Avimitin Studio",
    "exp": "1600483010",
    "user": "Tom",
    "admin": false
}
```

**然后使用 Base64Url 将这个 `JSON` 编码。**

### Signature

接着使用 `HMAC` 算法对上面编码的两个 `JSON` 进行加密：

```python
HMACSHA256(header_b64+"."+payload_b64, secret_key)
```

### 组合：

将上面三个部分得到的 base64 使用 `.` 组合起来，就能获得最终的 JWT 了：

## Python 实现

虽然 `JWT` 已经有成熟的模块了，但是在一些环境中能够原生实现相比起让用户安装依赖会更加合适一些，于是我根据上面标准，使用自带模块实现了 JWT。大概步骤如下：

### 导入包

制作 `Token ` 将需要以下依赖：

```python
# 加密
import hmac
# 获取时间戳
import time
# base64加密
import base64
# 获取字符
import string
# 比 random 更安全的随机
import secrets
# 加密算法
from hashlib import sha256
```

### base64加密

因为 `Token` 常用于 URL中传递，普通的 base64 加密中的 `\` `=` 等字符会造成歧义，所以我们需要加工一下：

```python 
def _safe_base64_url_encode(text):
    # 判断输入
    if isinstance(text, str):
        text = text.encode("utf-8")
    elif isinstance(text, bytes):
        pass
    else:
        raise TypeError("Expected string or bytes but got others")
	# 使用urlsafe方法得到无歧义的base64字节
    text_b64 = base64.urlsafe_b64encode(text)
    # 用replace方法将字节里的 = 去掉
    return text_b64.replace(b"=", b"")
```

### 生成 Header

```python
def header_generate(self):
    # 生成 header 之后传递base64加密后的字节 
	header = """{"alg": "HS256", "typ": "jwt"}"""
    return self._safe_base64_url_encode(header)
```

### 生成 Payload

```python
def payload_generate(self, username: str, permission: str):
    # 参数名是示例，可以自行调整
    exp = round(time.time()) + 50
    payload = """{"iss": "Avimitin Studio", "exp": "%d", "user": "%s", "admin": "%s"}""" % (exp, username, permission)
    return self._safe_base64_url_encode(payload)
```

### 生成加密密钥

加密用的密钥我是用的是一次性密钥的方法，你也可以换成自己熟记的密码串用来解密。

- 一次性随机密码串

```python
def _secret_key_generate(len: int):
    current_time = round(time.time())
    combine_text = string.ascii_letters + str(current_time)
    salt = ""
    while len > 0:
        salt += secrets.choice(combine_text)
        len -= 1
    return salt.encode("utf-8")
```

- 或者直接换成自己熟记的密码（尽量不要明文）

```python
def _secret_key_generate():
	with open("config/password.json", "r") as password_file:
        return json.loads(password_file)["password"]
```

- 关于 Secrets 模块的[更多说明](https://docs.python.org/3/library/secrets.html)

### 算法加密

```python
def encrypt(key, msg):
    # new 方法生成一个新的HMAC对象，用digest返回加密后的抽样
    sign = hmac.new(key, msg, sha256).digest()
    return sign
```

- 关于HMAC的[更多API说明](https://docs.python.org/3/library/hmac.html)

### 最终合成

```python
def generate(self, username, permission):
    # 生成加密用的密钥
    key = self._secret_key_generate(16)
    print("加密密钥： " + key.decode("utf-8"))
	# 将前两段信息的 base64 用 . 合并起来    
    message = self.header_generate() + b"." + self.payload_generate(username, permission)
    # 用密钥把message加密
    signature = self.encrypt(key, message)
    # 将加密后获得的抽样 base64 加密
    signature_b64 = self._safe_base64_url_encode(signature)
    part = [message, signature_b64]
    # 最终把所有的base64合并并解码为 string 字符串
    return b".".join(part).decode("utf-8")
```

### 输出样例

最终程序输出样例：

```
加密密钥： XWBCv6bT8FXNKV2z

JWT: eyJhbGciOiAiSFMyNTYiLCAidHlwIjogImp3dCJ9.eyJpc3MiOiAiQXZpbWl0aW4gU3R1ZGlvIiwgImV4cCI6ICIxNjAwNDg1Nzc4IiwgInVzZXIiOiAiYXZpbWl0aW4iLCAiYWRtaW4iOiAiVHJ1ZSJ9.Kph2-Qin9Xrd3LwLWX5mOtGiei-1Cp2mtWuhzps3ub0
```

## 代码样本：

你可以在我的 [GitHub](https://github.com/Avimitin/JWTPY) 看到完整的代码演示。



