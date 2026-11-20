#!/usr/bin/env bash
# repro_v7: re-predict the two sample genomes with the repo's own ESMC path, then
# retrain the five seeds from scratch and compare with repro_v6.
set -uo pipefail
cd /home/gonghang/codes/bgc-detector || exit 1

ROOT=/home/gonghang/codes/bgc-detector/outputs/repro_v7_esm
if [ -e "$ROOT" ]; then
    echo "output root already exists: $ROOT"
    exit 1
fi
mkdir -p "$ROOT/logs"
git rev-parse HEAD > "$ROOT/source_commit.txt"
export https_proxy=http://127.0.0.1:7892 http_proxy=http://127.0.0.1:7892
echo "=== uv sync $(date) ==="
uv sync --locked >> "$ROOT/logs/sync.log" 2>&1 || { echo "uv sync failed"; exit 1; }
unset https_proxy http_proxy
DATA=/home/gonghang/codes/my-bgc-v2/data

echo "=== predictions $(date) ==="
for genome in mycoplasma coelicolor; do
    uv run bgc-detector predict \
        --genome "/tmp/sample/${genome}.gbk" \
        --checkpoint "outputs/repro_v6/runs/final_seed1506/detector_960d_best.pt" \
        --outdir "$ROOT/pred_${genome}" --name "${genome}" \
        --device cuda:2 --threshold checkpoint \
        > "$ROOT/logs/pred_${genome}.log" 2>&1
    echo "${genome} exit=$?"
done

echo "=== retrain five seeds $(date) ==="
DATA_ROOT="$DATA" \
OUTPUT_ROOT="$ROOT/retrain" \
TRAIN_DS="$DATA/derived/train_test/dataset_legacy_train.parquet" \
VAL_DS="$DATA/derived/train_test/dataset_legacy_val.parquet" \
PRETRAIN_REUSE="" \
SEEDS="1506 1507 1508 1509 1510" \
GPUS="0 1 2 0 1" \
bash scripts/train/run_final.sh > "$ROOT/logs/run_final.log" 2>&1
echo "=== run_final exit=$? $(date) ==="
