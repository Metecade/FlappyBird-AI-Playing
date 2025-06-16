#!/bin/bash

# 可以选择运行的模型文件
MODEL_PATH="models/final_model.pth"

python test.py \
      "$MODEL_PATH"

if [ $? -ne 0 ]; then
    echo "Python脚本执行失败！"
    exit 1
else
    echo "执行成功"
fi