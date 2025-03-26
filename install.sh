#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit

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

git lfs install

git clone "https://huggingface.co/lj1995/GPT-SoVITS" 111

mv 111/* GPT_SoVITS/pretrained_models

wget "https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip"

unzip G2PWModel_1.1.zip

rm -rf G2PWModel_1.1.zip

mv G2PWModel_1.1 GPT_SoVITS/text/G2PWModel

# Install build tools
conda install -c conda-forge gcc -y
conda install -c conda-forge gxx -y
conda install ffmpeg cmake -y

# 设置编译环境
# Set up build environment
export CMAKE_MAKE_PROGRAM="$CONDA_PREFIX/bin/cmake"
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"

pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://download.pytorch.org/whl/rocm6.2.4 --extra-index-url https://download.pytorch.org/whl/cpu

# 刷新环境
# Refresh environment
hash -r

pip install -r requirements.txt
