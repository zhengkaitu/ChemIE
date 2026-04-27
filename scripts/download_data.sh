#!/usr/bin/env bash
set -euo pipefail

download_if_missing() {
    local repo="$1"
    local dir="$2"
    if [ -d "$dir" ]; then
        echo "Skipping $repo — $dir already exists."
        return
    fi
    echo "Downloading $repo -> $dir"
    huggingface-cli download "$repo" --local-dir "$dir" --repo-type dataset
}

download_if_missing docling-project/MarkushGrapher-Datasets  ./data/MG1
download_if_missing docling-project/MarkushGrapher-2-Datasets ./data/MG2
