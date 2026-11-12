#!/usr/bin/env bash
# Final-recipe runner: weak pretraining (optional) -> detection training -> NG eval.
#
# One worker per GPU, seeds round-robin. Every seed gets its own run directory.
# Fixed recipe lives in bgc_gnn/train.py; the only model switch here is
# CONTEXT_READOUT. There is no NG-based epoch or seed selection.
#
# Prereqs (once):
#   uv run python scripts/build_catalog.py
#   uv run python scripts/encode.py --source catalog
#   uv run python scripts/encode.py --source ng
#   uv run python scripts/build_dataset.py
#   uv run python scripts/build_pretraining_data.py ...
#   uv run python scripts/encode_pretraining.py ...
#
# Usage:
#   SEEDS="1601 1602 1603" nohup bash scripts/run_final.sh > runner.log 2>&1 &
#   CONTEXT_READOUT=1 SEEDS="1601" GPUS="1" nohup bash scripts/run_final.sh > runner.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(${SEEDS:-1301 1302 1303 1304 1305})
GPUS=(${GPUS:-0})
TRAIN_DIR=${TRAIN_DIR:-runs/final_seed}          # -> ${TRAIN_DIR}${seed}
EMB_DIM=${EMB_DIM:-960}
CONTEXT_READOUT=${CONTEXT_READOUT:-0}             # 0|1, the only model switch
PRETRAIN=${PRETRAIN:-1}                           # 1 = weak pretraining then transfer
PRETRAIN_DIR=${PRETRAIN_DIR:-pretrain/weak_seed}  # -> ${PRETRAIN_DIR}${seed}
PRETRAIN_REUSE=${PRETRAIN_REUSE:-}                # template with {seed}; skips pretraining
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-20}
OVERLAP_STRIDE=${OVERLAP_STRIDE:-64}              # secondary overlap protocol
BGC_PROPHET_POSTPROCESS=${BGC_PROPHET_POSTPROCESS:-0}

CATALOG=data/derived/catalog
TRAIN_DS=${TRAIN_DS:-data/derived/train_test/dataset_train.parquet}
VAL_DS=${VAL_DS:-data/derived/train_test/dataset_val.parquet}
EMB_PATH=${EMB_PATH:-${CATALOG}/gene_embeddings_${EMB_DIM}d.parquet}
NG_EMB=data/derived/ng_embeddings/ng_gene_embeddings_${EMB_DIM}d.parquet
PT_DATA=${PT_DATA:-data/derived/pretraining/antismash_v2_v1}
PT_CACHE=${PT_CACHE:-data/derived/pretraining/as566_v3_cache}

case "$CONTEXT_READOUT" in 0|1) ;; *) echo "CONTEXT_READOUT must be 0 or 1"; exit 2 ;; esac
case "$PRETRAIN" in 0|1) ;; *) echo "PRETRAIN must be 0 or 1"; exit 2 ;; esac
for f in "$TRAIN_DS" "$VAL_DS" "$EMB_PATH" "$NG_EMB"; do
    [ -f "$f" ] || { echo "missing: $f (see prereqs in header)"; exit 1; }
done
if [ -z "$PRETRAIN_REUSE" ] && [ "$PRETRAIN" = "1" ]; then
    for f in "$PT_DATA/audit.json" "$PT_CACHE/metadata.json"; do
        [ -f "$f" ] || { echo "missing: $f (see prereqs in header)"; exit 1; }
    done
fi

eval_ng() {  # <run_dir> <gpu> <extra args...>
    local run_dir=$1 gpu=$2; shift 2
    uv run python -u scripts/eval_ng.py \
        --checkpoint "${run_dir}/detector_${EMB_DIM}d_best.pt" \
        --device "cuda:${gpu}" "$@"
}

run_one() {  # <seed> <gpu>
    local seed=$1 gpu=$2
    local run_dir="${TRAIN_DIR}${seed}"
    local pt_dir="${PRETRAIN_DIR}${seed}"
    local pretrained=""

    echo "[gpu $gpu] seed=$seed context_readout=$CONTEXT_READOUT -> $run_dir"

    if [ -d "$run_dir" ] && [ -n "$(ls -A "$run_dir")" ]; then
        echo "[gpu $gpu] seed=$seed: non-empty run directory $run_dir; use a new TRAIN_DIR (no resume)"
        return 1
    fi

    if [ -n "$PRETRAIN_REUSE" ]; then
        pretrained=${PRETRAIN_REUSE//\{seed\}/$seed}/pretrained_best.pt
        [ -f "$pretrained" ] || { echo "[gpu $gpu] seed=$seed: missing $pretrained"; return 1; }
    elif [ "$PRETRAIN" = "1" ]; then
        if [ ! -f "$pt_dir/status.json" ] || ! grep -q completed "$pt_dir/status.json"; then
            mkdir -p "$pt_dir"
            echo "[gpu $gpu] seed=$seed: weak pretraining -> $pt_dir"
            uv run python -u -m bgc_gnn.pretrain \
                --data-dir "$PT_DATA" --cache-dir "$PT_CACHE" \
                --output-dir "$pt_dir" --seed "$seed" \
                --device "cuda:${gpu}" \
                > "${pt_dir}/pretrain.log" 2>&1 \
                || { echo "[gpu $gpu] seed=$seed: PRETRAIN FAILED (see ${pt_dir}/pretrain.log)"; return 1; }
        fi
        pretrained="$pt_dir/pretrained_best.pt"
    fi

    mkdir -p "$run_dir"
    local extra=(--context-readout)
    [ "$CONTEXT_READOUT" = "1" ] || extra=()
    local transfer=()
    [ -n "$pretrained" ] && transfer=(--pretrained-checkpoint "$pretrained")

    echo "[gpu $gpu] seed=$seed: training"
    uv run python -u -m bgc_gnn.train \
        --fixed-dataset "$TRAIN_DS" --fixed-val-dataset "$VAL_DS" \
        --emb-path "$EMB_PATH" \
        --checkpoint-dir "$run_dir" --seed "$seed" \
        --device "cuda:${gpu}" --epochs "$EPOCHS" --patience "$PATIENCE" \
        "${extra[@]}" "${transfer[@]}" \
        > "${run_dir}/train.log" 2>&1 \
        || { echo "[gpu $gpu] seed=$seed: TRAIN FAILED (see ${run_dir}/train.log)"; return 1; }

    # Primary result: checkpoint selected by validation only.
    eval_ng "$run_dir" "$gpu" \
        --metrics-path "${run_dir}/ng_validation_selected.json" \
        > "${run_dir}/eval_validation_selected.log" 2>&1 \
        || { echo "[gpu $gpu] seed=$seed: EVAL FAILED (see ${run_dir}/eval_validation_selected.log)"; return 1; }

    # Secondary: overlapping windows with mean aggregation.
    eval_ng "$run_dir" "$gpu" \
        --stride "$OVERLAP_STRIDE" --aggregate mean \
        --metrics-path "${run_dir}/ng_validation_selected_overlap_s${OVERLAP_STRIDE}_mean.json" \
        > "${run_dir}/eval_validation_selected_overlap.log" 2>&1 \
        || { echo "[gpu $gpu] seed=$seed: OVERLAP EVAL FAILED (see ${run_dir}/eval_validation_selected_overlap.log)"; return 1; }

    if [ "$BGC_PROPHET_POSTPROCESS" = "1" ]; then
        eval_ng "$run_dir" "$gpu" \
            --stride "$OVERLAP_STRIDE" --aggregate mean \
            --bgc-prophet-postprocess --bgc-prophet-longest-only \
            --metrics-path "${run_dir}/ng_bgc_prophet_longest.json" \
            > "${run_dir}/eval_bgc_prophet.log" 2>&1 \
            || { echo "[gpu $gpu] seed=$seed: POSTPROCESS EVAL FAILED (see ${run_dir}/eval_bgc_prophet.log)"; return 1; }
    fi
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

echo ""
echo "==== validation-selected NG summary ===="
for seed in "${SEEDS[@]}"; do
    for f in "${TRAIN_DIR}${seed}/ng_validation_selected.json" \
             "${TRAIN_DIR}${seed}/ng_validation_selected_overlap_s${OVERLAP_STRIDE}_mean.json"; do
        [ -f "$f" ] || continue
        uv run python -c "
import json
m = json.load(open('$f'))
print('  seed=${seed} %-8s threshold=%.4f AUROC=%.4f AUPRC=%.4f F1=%.4f MCC=%.4f'
      % (m['aggregate'], m['threshold'], m['auroc'], m['auprc'], m['f1'], m['mcc']))
"
    done
done
[ "$failed" -eq 0 ] || exit 1