#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

set -e

trap 'echo "Error Occured at \"$BASH_COMMAND\" with exit code $?"; exit 1' ERR

OS_TYPE=$(uname)
ARCHITECTURE=$(uname -m)

if command -v conda >/dev/null 2>&1; then
    echo "conda installed"
else
    echo "installing conda"

    if [ "$OS_TYPE" = "Darwin" ] && [[ "$ARCHITECTURE" == "arm64" ]]; then
        xcode-select --install
        wget --tries=25 --wait=3 --read-timeout=40 -O anaconda.sh "https://repo.anaconda.com/archive/Anaconda3-2024.10-1-MacOSX-arm64.sh"
    elif [ "$OS_TYPE" = "Linux" ] && [[ "$ARCHITECTURE" == "x86_64" ]]; then
        wget --tries=25 --wait=3 --read-timeout=40 -O anaconda.sh "https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh"
    else
        echo "Unsupported System for install.sh：$OS_TYPE with $ARCHITECTURE"
        exit 1
    fi

    bash "anaconda.sh" -b -p "$HOME/anaconda3"

    rm -rf "anaconda.sh"

    if [ "$OS_TYPE" = "Darwin" ]; then
        "$HOME/anaconda3/condabin/conda" init zsh
        source "$HOME/.zshrc"
    elif [ "$OS_TYPE" = "Linux" ]; then
        "$HOME/anaconda3/condabin/conda" init bash
        source "$HOME/.bashrc"
    fi

fi

CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")

source "$CONDA_PATH/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "GPT-SoVITS"; then
    :
else
    conda create -n GPT-SoVITS python=3.10 -y
fi

sudo apt install -y git-lfs
sudo apt install -y zip

conda activate GPT-SoVITS

git-lfs install

if find "GPT_SoVITS/pretrained_models" -mindepth 1 ! -name '.gitignore' | grep -q .; then
    echo "Pretrained Model Exists"
else
    echo "Download Pretrained Models"
    git clone "https://huggingface.co/lj1995/GPT-SoVITS" 111
    mv 111/* GPT_SoVITS/pretrained_models
    rm -rf 111
fi

if [ ! -d "GPT_SoVITS/text/G2PWModel" ]; then
    echo "Download G2PWModel"
    wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404 "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip"

    unzip G2PWModel_1.1.zip
    rm -rf G2PWModel_1.1.zip
    mv G2PWModel_1.1 GPT_SoVITS/text/G2PWModel
else
    echo "G2PWModel Exists"
fi

# Install build tools
echo "Installing GCC..."
conda install -c conda-forge gcc -y

echo "Installing G++..."
conda install -c conda-forge gxx -y

echo "Installing ffmpeg and cmake..."
conda install ffmpeg cmake -y

conda install jq -y

# 设置编译环境
# Set up build environment
export CMAKE_MAKE_PROGRAM="$CONDA_PREFIX/bin/cmake"
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"

echo "Checking for CUDA installation..."
if command -v nvidia-smi &>/dev/null; then
    USE_CUDA=true

    echo "CUDA found."
else
    echo "CUDA not found."

    USE_CUDA=false
fi

if [ "$USE_CUDA" = false ]; then
    echo "Checking for ROCm installation..."
    if [ -d "/opt/rocm" ]; then
        USE_ROCM=true
        echo "ROCm found."
        if grep -qi "microsoft" /proc/version; then
            echo "WSL found"
            IS_WSL=true
        else
            IS_WSL=false
        fi
    else
        echo "ROCm not found."
        USE_ROCM=false
    fi
fi

echo "Installing PyTorch"

if [ "$USE_CUDA" = true ]; then
    pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
elif [ "$USE_ROCM" = true ]; then
    pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2
else
    pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
fi

if [ "$USE_ROCM" = true ] && [ "$IS_WSL" = true ]; then
    echo "Update to WSL compatible runtime lib..."
    location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
    cd "${location}/torch/lib/" || exit 1
    rm libhsa-runtime64.so*
    cp /opt/rocm/lib/libhsa-runtime64.so.1.2 libhsa-runtime64.so
fi

# 刷新环境
# Refresh environment
hash -r

echo "Installing Python dependencies from requirements.txt..."

PACKAGE_NAME="pyopenjtalk"

VERSION=$(curl -s https://pypi.org/pypi/$PACKAGE_NAME/json | jq -r .info.version)

wget "https://files.pythonhosted.org/packages/source/${PACKAGE_NAME:0:1}/$PACKAGE_NAME/$PACKAGE_NAME-$VERSION.tar.gz"

TAR_FILE=$(ls ${PACKAGE_NAME}-*.tar.gz)
DIR_NAME="${TAR_FILE%.tar.gz}"

tar -xzf "$TAR_FILE"
rm "$TAR_FILE"

CMAKE_FILE="$DIR_NAME/lib/open_jtalk/src/CMakeLists.txt"

if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' -E 's/cmake_minimum_required\(VERSION[^\)]*\)/cmake_minimum_required(VERSION 3.5...3.31)/' "$CMAKE_FILE"
else
    sed -i -E 's/cmake_minimum_required\(VERSION[^\)]*\)/cmake_minimum_required(VERSION 3.5...3.31)/' "$CMAKE_FILE"
fi

tar -czf "$TAR_FILE" "$DIR_NAME"

pip install "$TAR_FILE"

rm -rf "$TAR_FILE" "$DIR_NAME"

pip install -r extra-req.txt --no-deps

pip install -r requirements.txt

echo "Installation completed successfully!"
