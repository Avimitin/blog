---
title: 'Useful shell script'
author: avimitin
date: 2021/06/28 11:20
tag: [shell, linux]
categories: [shell]
---
# Useful Shell Scripts

Here store some script that I have used for long time and which are really useful.

## Get exactly what your OS named

```bash
#!/bin/bash

cat /etc/os-release | grep '^ID=' | awk -F= '{print $2}'
```

