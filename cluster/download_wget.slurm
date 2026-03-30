#!/bin/bash
#SBATCH -J wget-qwen
#SBATCH -o logs/wget_qwen.%j.out
#SBATCH -e logs/wget_qwen.%j.err
#SBATCH -p mi2508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

MODEL_DIR="/work1/harangju/harang/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
SNAP_DIR="${MODEL_DIR}/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
BLOB_DIR="${MODEL_DIR}/blobs"
BASE_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main"

mkdir -p "$SNAP_DIR" "$BLOB_DIR"

# Remove stale incomplete files
rm -f "$BLOB_DIR"/*.incomplete

echo "Downloading model shards at $(date)"
for i in 1 2 3 4; do
    SHARD="model-0000${i}-of-00004.safetensors"
    DEST="${SNAP_DIR}/${SHARD}"
    if [ -f "$DEST" ] && [ -s "$DEST" ]; then
        echo "  ${SHARD} already exists, skipping"
        continue
    fi
    echo "  Downloading ${SHARD}..."
    wget -q --show-progress -O "$DEST" "${BASE_URL}/${SHARD}"
    echo "  Done: $(ls -lh "$DEST" | awk '{print $5}')"
done

# Also grab the index file if missing
IDX="${SNAP_DIR}/model.safetensors.index.json"
if [ ! -f "$IDX" ]; then
    wget -q -O "$IDX" "${BASE_URL}/model.safetensors.index.json"
fi

echo "All downloads complete at $(date)"
ls -lh "$SNAP_DIR"/*.safetensors 2>/dev/null
