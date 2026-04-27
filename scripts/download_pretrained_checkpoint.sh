#!/usr/bin/env bash
set -euo pipefail

FILE=./ckpts/swin_base_char_aux_1m.pth

if [ -f "$FILE" ]; then
    echo "Skipping — $FILE already exists."
    exit 0
fi

mkdir -p ./ckpts
echo "Downloading MolScribe checkpoint -> $FILE"
huggingface-cli download yujieq/MolScribe swin_base_char_aux_1m.pth --local-dir ./ckpts
