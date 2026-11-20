#!/usr/bin/env bash
# Fixed pipeline: semantic-enabled pretraining -> detector transfer -> overlap NG evaluation.
# Only paths, seeds and GPU placement are runtime inputs. No model/recipe switches.
set -euo pipefail
cd "$(dirname "$0")/.."

# Historical execution settings: prevent concurrent seeds creating huge CPU pools.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

SEEDS=(${SEEDS:-1301 1302 1303 1304 1305})
GPUS=(${GPUS:-0})
DATA_ROOT=${DATA_ROOT:-data}
OUTPUT_ROOT=${OUTPUT_ROOT:-.}
TRAIN_DIR=${TRAIN_DIR:-${OUTPUT_ROOT}/runs/final_seed}
PRETRAIN_DIR=${PRETRAIN_DIR:-${OUTPUT_ROOT}/pretrain/seed}
PRETRAIN_REUSE=${PRETRAIN_REUSE:-}  # empty means new pretraining for every seed
TRAIN_DS=${TRAIN_DS:-${DATA_ROOT}/derived/train_test/dataset_train.parquet}
VAL_DS=${VAL_DS:-${DATA_ROOT}/derived/train_test/dataset_val.parquet}
EMB_PATH=${EMB_PATH:-${DATA_ROOT}/derived/catalog/gene_embeddings_960d.parquet}
NG_EMB=${NG_EMB:-${DATA_ROOT}/derived/ng_embeddings/ng_gene_embeddings_960d.parquet}
NG_GBK=${NG_GBK:-${DATA_ROOT}/raw/ng/ClusterFinder_Annotated_Contigs.full.gbk}
NG_LABELS=${NG_LABELS:-${DATA_ROOT}/raw/ng/291BGC.csv}
PT_DATA=${PT_DATA:-${DATA_ROOT}/derived/pretraining/antismash_v2_v1}
PT_CACHE=${PT_CACHE:-${DATA_ROOT}/derived/pretraining/antismash_v2_v1_restart_v2_cache}

# Historical inputs keep their original names and contents; never rename shared data.
if [ ! -f "$TRAIN_DS" ] && [ "$TRAIN_DS" = "${DATA_ROOT}/derived/train_test/dataset_train.parquet" ]; then
    TRAIN_DS=${DATA_ROOT}/derived/train_test/dataset_legacy_train.parquet
fi
if [ ! -f "$VAL_DS" ] && [ "$VAL_DS" = "${DATA_ROOT}/derived/train_test/dataset_val.parquet" ]; then
    VAL_DS=${DATA_ROOT}/derived/train_test/dataset_legacy_val.parquet
fi

for f in "$TRAIN_DS" "$VAL_DS" "$EMB_PATH" "$NG_EMB" "$NG_GBK" "$NG_LABELS"; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done
if [ -z "$PRETRAIN_REUSE" ]; then
    for f in "$PT_DATA/audit.json" "$PT_DATA/windows_train.parquet" "$PT_DATA/windows_val.parquet" \
             "$PT_DATA/proteins.parquet" "$PT_CACHE/metadata.json" \
             "$PT_CACHE/embeddings.npy" "$PT_CACHE/sequence_ids.json"; do
        [ -f "$f" ] || { echo "missing: $f"; exit 1; }
    done
    mkdir -p "$OUTPUT_ROOT"
    uv run python -u -m bgc_detector.verify_inputs \
        --data-dir "$PT_DATA" --cache-dir "$PT_CACHE" \
        --output "$OUTPUT_ROOT/pretraining_input_verification.json"
fi

printf 'Input data root: %s\nOutput root: %s\nTrain: %s\nValidation: %s\n' \
    "$DATA_ROOT" "$OUTPUT_ROOT" "$TRAIN_DS" "$VAL_DS"

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
            uv run python -u -m bgc_detector.pretrain \
                --data-dir "$PT_DATA" --cache-dir "$PT_CACHE" \
                --input-verification "$OUTPUT_ROOT/pretraining_input_verification.json" \
                --output-dir "$pt_dir" --seed "$seed" --device "cuda:${gpu}" \
                > "$log_dir/pretrain.log" 2>&1 \
                || { echo "PRETRAIN FAILED: $log_dir/pretrain.log"; return 1; }
        fi
        pretrained="$pt_dir/pretrained_best.pt"
    fi
    [ -f "$pretrained" ] || { echo "missing: $pretrained"; return 1; }
    uv run python -u -m bgc_detector.train \
        --fixed-dataset "$TRAIN_DS" --fixed-val-dataset "$VAL_DS" \
        --emb-path "$EMB_PATH" --checkpoint-dir "$run_dir" --seed "$seed" \
        --device "cuda:${gpu}" --pretrained-checkpoint "$pretrained" \
        > "$log_dir/train.log" 2>&1 \
        || { echo "TRAIN FAILED: $log_dir/train.log"; return 1; }

    uv run python -u -m scripts.eval.eval_ng \
        --checkpoint "$run_dir/detector_960d_best.pt" --device "cuda:${gpu}" \
        --ng-gbk "$NG_GBK" --labels "$NG_LABELS" --emb-path "$NG_EMB" \
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
