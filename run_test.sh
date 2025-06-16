#!/bin/bash

python test.py

if [ $? -ne 0 ]; then
    echo "Python脚本执行失败！"
    exit 1
else
    echo "执行成功"
fi