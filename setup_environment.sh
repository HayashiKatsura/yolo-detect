#!/bin/bash

# 确保以root或sudo权限运行
if [ "$EUID" -ne 0 ]; then
  echo "请以root用户或使用sudo运行此脚本。"
  exit 1
fi

echo "更新软件包列表..."
apt update -y

echo "安装必要的系统包..."
apt install -y vim libgl1
apt-get update && apt-get install -y libc6
apt-get install -y libglib2.0-0

echo "升级pip，并使用清华镜像..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "安装其他Python库..."
pip install flask
pip install pyyaml
pip install flask_cors
pip install py7zr
pip install pandas
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install opencv-python
pip install matplotlib
pip install tqdm
pip install scipy
pip install numpy==1.24


echo "环境安装完成！"
