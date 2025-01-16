#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/GPT_SoVITS:$SCRIPT_DIR:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

cfg_path="GPT_SoVITS/configs/Cfg.json"
speaker_cfg_path="GPT_SoVITS/configs/Speakers.json"
api=false
infer=false

print_help() {
  cat <<EOF
Usage: $0 [OPTIONS]
Running Tarin webui as default
Options:
  -c, --cfg-path <path>          Path to the configuration file
  -s, --speaker-cfg-path <path>  Path to the speaker configuration file
  -i, --infer                  Run inference web UI
  -a, --api                     Run the API 
  -h, --help                    Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
  --cfg-path | -c)
    cfg_path="$2"
    shift 2
    ;;
  --speaker-cfg-path | -s)
    speaker_cfg_path="$2"
    shift 2
    ;;
  --api | -a)
    api=true
    shift
    ;;
  --infer | -i)
    infer=true
    shift
    ;;
  --help | -h)
    print_help
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    print_help
    exit 1
    ;;
  esac
done

echo "Config path: $cfg_path"
echo "Speaker config path: $speaker_cfg_path"
echo "API enabled: $api"

if $api && $infer; then
  echo "Error: --api (-a) and --infer (-i) cannot be used together."
  print_help
  exit 1
fi

if $infer; then
  echo "Running Inference Web UI..."
  python3 inference_webui.py --cfg-path "$cfg_path" --speaker-cfg-path "$speaker_cfg_path"
  exit 0
elif $api; then
  echo "Running API v2..."
  python3 tools/api.py --api-config "$cfg_path" --speakers-config "$speaker_cfg_path"
  exit 0
else
  echo "Running Web UI..."
  python3 webui.py --cfg-path "$cfg_path" --speaker-cfg-path "$speaker_cfg_path"
  exit 0
fi
