#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

sudo apt install git-lfs

if command -v conda >/dev/null 2>&1; then
    echo "conda installed"
else
    echo "installing conda"

    os_type=$(uname)
    architecture=$(uname -m)

    if [ "$os_type" = "Darwin" ] && [[ "$architecture" == "arm64" ]]; then
        xcode-select --install
        wget -O anaconda.sh "https://repo.anaconda.com/archive/Anaconda3-2024.10-1-MacOSX-arm64.sh"
    elif [ "$os_type" = "Linux" ] && [[ "$architecture" == "x86_64" ]]; then
        wget -O anaconda.sh "https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh"
    else
        echo "Unsupported System for install.sh：$os_type with $architecture"
        exit 1
    fi

    bash "anaconda.sh" -b -p "$HOME/anaconda3"

    rm -rf "anaconda.sh"

    if [ "$os_type" = "Darwin" ]; then
        "$HOME/anaconda3/condabin/conda" init zsh
        source "$HOME/.zshrc"
    elif [ "$os_type" = "Linux" ]; then
        "$HOME/anaconda3/condabin/conda" init bash
        source "$HOME/.bashrc"
    fi

fi

if conda env list | awk '{print $1}' | grep -Fxq "GPT-SoVITS"; then
    :
else
    conda create -n GPT-SoVITS python=3.10 -y
fi

conda activate GPT-SoVITS

conda install git

git-lfs install

git clone "https://huggingface.co/lj1995/GPT-SoVITS" 111

mv 111/* GPT_SoVITS/pretrained_models

wget "https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip"

unzip G2PWModel_1.1.zip

rm -rf G2PWModel_1.1.zip

mv G2PWModel_1.1 GPT_SoVITS/text/G2PWModel

# Install build tools
echo "Installing GCC..."
conda install -c conda-forge gcc -y

echo "Installing G++..."
conda install -c conda-forge gxx -y

echo "Installing ffmpeg and cmake..."
conda install ffmpeg cmake -y

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

pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://download.pytorch.org/whl/rocm6.2.4 --extra-index-url https://download.pytorch.org/whl/cpu

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

pip install -r requirements.txt

echo "Installation completed successfully!"
