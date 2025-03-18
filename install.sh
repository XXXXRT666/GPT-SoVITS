#!/bin/bash

# 安装构建工具
# Install build tools
conda install -c conda-forge gcc=14 -y
conda install -c conda-forge gxx -y
conda install ffmpeg cmake -y

# 设置编译环境
# Set up build environment
export CMAKE_MAKE_PROGRAM="$CONDA_PREFIX/bin/cmake"
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"

pip install torch==2.5.1 torchaudio==2.5.1

# 刷新环境
# Refresh environment
hash -r

pip install -r requirements.txt
