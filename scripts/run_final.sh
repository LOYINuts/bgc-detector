#!/usr/bin/env bash
# Fixed pipeline: semantic-enabled pretraining -> detector transfer -> overlap NG evaluation.
# Only paths, seeds and GPU placement are runtime inputs. No model/recipe switches.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(${SEEDS:-1301 1302 1303 1304 1305})
GPUS=(${GPUS:-0})
TRAIN_DIR=${TRAIN_DIR:-runs/final_seed}
PRETRAIN_DIR=${PRETRAIN_DIR:-pretrain/seed}
PRETRAIN_REUSE=${PRETRAIN_REUSE:-}  # completed checkpoint directory template with {seed}
TRAIN_DS=${TRAIN_DS:-data/derived/train_test/dataset_train.parquet}
VAL_DS=${VAL_DS:-data/derived/train_test/dataset_val.parquet}
EMB_PATH=${EMB_PATH:-data/derived/catalog/gene_embeddings_960d.parquet}
NG_EMB=data/derived/ng_embeddings/ng_gene_embeddings_960d.parquet
PT_DATA=${PT_DATA:-data/derived/pretraining/antismash_v2_v1}
PT_CACHE=${PT_CACHE:-data/derived/pretraining/antismash_v2_v1_restart_v2_cache}

for f in "$TRAIN_DS" "$VAL_DS" "$EMB_PATH" "$NG_EMB"; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done
if [ -z "$PRETRAIN_REUSE" ]; then
    for f in "$PT_DATA/audit.json" "$PT_CACHE/metadata.json"; do
        [ -f "$f" ] || { echo "missing: $f"; exit 1; }
    done
fi

run_one() {
    local seed=$1 gpu=$2
    local run_dir="${TRAIN_DIR}${seed}" pt_dir="${PRETRAIN_DIR}${seed}"
    local log_dir="${run_dir}.logs" pretrained
    if [ -e "$run_dir" ] || [ -e "$log_dir" ]; then
        echo "[gpu $gpu] seed=$seed: output exists; use a new TRAIN_DIR (no resume)"
        return 1
    fi
    mkdir -p "$log_dir"
    echo "[gpu $gpu] seed=$seed -> $run_dir"
    if [ -n "$PRETRAIN_REUSE" ]; then
        pretrained=${PRETRAIN_REUSE//\{seed\}/$seed}/pretrained_best.pt
    else
        if [ -f "$pt_dir/status.json" ] && grep -q '"status": "completed"' "$pt_dir/status.json"; then
            :
        elif [ -e "$pt_dir" ]; then
            echo "[gpu $gpu] seed=$seed: pretraining output is incomplete; use a fresh PRETRAIN_DIR"
            return 1
        else
            # Logs stay outside the fresh output directory created by pretrain.py.
            uv run python -u -m bgc_gnn.pretrain \
                --data-dir "$PT_DATA" --cache-dir "$PT_CACHE" \
                --output-dir "$pt_dir" --seed "$seed" --device "cuda:${gpu}" \
                > "$log_dir/pretrain.log" 2>&1 \
                || { echo "PRETRAIN FAILED: $log_dir/pretrain.log"; return 1; }
        fi
        pretrained="$pt_dir/pretrained_best.pt"
    fi
    [ -f "$pretrained" ] || { echo "missing: $pretrained"; return 1; }
    uv run python -u -m bgc_gnn.train \
        --fixed-dataset "$TRAIN_DS" --fixed-val-dataset "$VAL_DS" \
        --emb-path "$EMB_PATH" --checkpoint-dir "$run_dir" --seed "$seed" \
        --device "cuda:${gpu}" --pretrained-checkpoint "$pretrained" \
        > "$log_dir/train.log" 2>&1 \
        || { echo "TRAIN FAILED: $log_dir/train.log"; return 1; }

    uv run python -u scripts/eval_ng.py \
        --checkpoint "$run_dir/detector_960d_best.pt" --device "cuda:${gpu}" \
        --metrics-path "$run_dir/ng_metrics.json" \
        > "$log_dir/eval_overlap.log" 2>&1 \
        || { echo "PRIMARY EVAL FAILED: $log_dir/eval_overlap.log"; return 1; }
    echo "[gpu $gpu] seed=$seed: done"
}

n_gpus=${#GPUS[@]}
[ "$n_gpus" -gt 0 ] || { echo "GPUS cannot be empty"; exit 1; }
pids=()
for g in "${!GPUS[@]}"; do
(
    for i in "${!SEEDS[@]}"; do
        if (( i % n_gpus == g )); then
            run_one "${SEEDS[$i]}" "${GPUS[$g]}"
        fi
    done
) &
pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
[ "$failed" -eq 0 ] || exit 1
echo "All seeds completed; metrics: ng_metrics.json (stride64/mean)"
