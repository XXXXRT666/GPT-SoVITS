SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit

wget "https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh"

bash "Anaconda3-2024.10-1-Linux-x86_64.sh" -b

rm -rf "Anaconda3-2024.10-1-Linux-x86_64.sh"

"$HOME/anaconda3/condabin/conda" init bash

source "$HOME/.bashrc"

conda create -n GPT-SoVITS python=3.10 -y

conda activate GPT-SoVITS

git lfs install

git clone "https://huggingface.co/lj1995/GPT-SoVITS" 111

mv 111/* GPT_SoVITS/pretrained_models

wget "https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip"

unzip G2PWModel_1.1.zip

rm -rf G2PWModel_1.1.zip

mv G2PWModel_1.1 GPT_SoVITS/text/G2PWModel

bash install.sh

pip install py-cpuinfo
