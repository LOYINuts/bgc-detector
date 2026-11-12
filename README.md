# bgc-detector

Per-gene BGC (biosynthetic gene cluster) detection from frozen ESMC-300M gene
embeddings. This repository holds **only the final model and the pipeline that
produces it**; the ablation codebase and its experiment records stay in
`my-bgc-v2`.

## Final model

One architecture, one training recipe:

```
frozen ESMC-300M 960d gene embeddings
  -> LayerNorm
  -> Channel-Gated Conv-BiGRU encoder (2 layers, d_model 960, kernel 5, gate bias -3.0)
  -> detection MLP 960->128->32->1
  + zero-initialized linear semantic skip (protein-only logits)
  [+ optional zero-initialized context readout MLP 256]
  -> sigmoid -> per-gene BGC probability
```

| Item | Value |
| --- | --- |
| parameters | 10,290,690 (no context readout) / 10,536,963 (with) |
| loss | weighted BCE, pos_weight 2 |
| optimiser | AdamW, lr 5e-4, wd 0.01, encoder lr ×0.2 on transfer |
| batch / dropout | 64 / encoder 0.5, head 0.1 |
| EMA | 0.999 (validated and exported weights) |
| LR schedule | ExponentialLR gamma 0.95, first step at epoch 5, stop at lr ≤ 5e-6 |
| epochs / patience | 100 / 20 |
| checkpoint selection | validation loss |
| decision threshold | selected on validation MCC, saved in the checkpoint |
| pretraining | weak (supervised windows) detector transfer |

`--context-readout` is the only model switch: the final context on/off
comparison had not yet been decided when this repository was split out.

## Layout

```
bgc_gnn/
  model.py       the detector
  dataset.py     embedding store, random and fixed-window datasets
  train.py       detection training (fixed recipe)
  pretrain.py    weak pretraining
  eval.py        detection metrics, validation threshold selection, region postprocess
  checkpoint.py  checkpoint payloads, provenance, weak transfer, settings resolution
scripts/
  build_catalog.py, gbk2glm2.py, gbk_as_to_glm2.py, prune_with_antismash.py   data build
  encode.py                                                                   ESMC embeddings
  build_dataset.py                                                            fixed train/val windows
  build_pretraining_data.py, encode_pretraining.py, plan_pretraining_embeddings.py
  eval_ng.py                                                                  NG test evaluation
  eval_antismash.py, eval_deepbgc.py                                          external comparisons
  check_dataset_split.py                                                      split/leakage audit
  run_final.sh                                                                pretrain + train + eval runner
tests/            unittest suite (uv run python -m unittest discover -s tests)
```

## Pipeline

```bash
uv sync

# 1. catalog (MIBiG BGCs + GTDB genomes, antiSMASH exclusion)
uv run python scripts/build_catalog.py --antismash-dir <antiSMASH results>

# 2. frozen ESMC embeddings for catalog and NG test set
uv run python scripts/encode.py --source catalog
uv run python scripts/encode.py --source ng

# 3. fixed 128-gene train/val windows (grouped entity split)
uv run python scripts/build_dataset.py

# 4. weak pretraining data and its deduplicated embedding cache
uv run python scripts/build_pretraining_data.py \
    --antismash-dir <antiSMASH results> --audit-dir <integrity audit> \
    --output-dir data/derived/pretraining/antismash_v2_v1
uv run python scripts/plan_pretraining_embeddings.py \
    --data-dir data/derived/pretraining/antismash_v2_v1 \
    --reuse-cache data/derived/pretraining/as566_v3_cache \
    --output-dir <inventory>
uv run python scripts/encode_pretraining.py \
    --data-dir data/derived/pretraining/antismash_v2_v1 \
    --output-dir data/derived/pretraining/as566_v3_cache

# 5. seeds: weak pretraining -> training -> NG evaluation
SEEDS="1601 1602 1603" GPUS="0 1 2" bash scripts/run_final.sh
```

Each seed gets `runs/final_seed<seed>/` (training logs, `detector_960d_best.pt`,
`ng_validation_selected.json`, plus the overlap-protocol metrics). Pretraining
reuses existing checkpoints through `PRETRAIN_REUSE="<dir-with-{seed}>"`.

A single training run without the runner:

```bash
uv run python -m bgc_gnn.train \
    --fixed-dataset data/derived/train_test/dataset_train.parquet \
    --fixed-val-dataset data/derived/train_test/dataset_val.parquet \
    --emb-path data/derived/catalog/gene_embeddings_960d.parquet \
    --checkpoint-dir runs/final_seed1301 --seed 1301 \
    --pretrained-checkpoint pretrain/weak_seed1301/pretrained_best.pt
```

## Evaluation rules

- Primary result: NG metrics at the validation-selected threshold, non-overlapping
  128-gene windows. Overlapping/mean-aggregated and postprocessed numbers are
  secondary and reported as such.
- The NG-best threshold scan written into the metrics JSON is diagnostic only: it
  must never pick a checkpoint, a seed or a model variant.
- Checkpoint selection uses validation loss only; no test-set information enters
  training.

## Compatibility

`bgc_gnn/model.py` keeps the original module names and tensor shapes, so
checkpoints written by the ablation codebase (including the
`context_readout` variants) load with `strict=True`, and a given weight set
produces bit-identical predictions.

## Requirements

Installed by `uv sync` (see `pyproject.toml`): Python ≥ 3.12, PyTorch (CUDA build
for training), pyarrow, numpy, scikit-learn, biopython, tqdm.

Installed separately, not in `pyproject.toml`:

- `esm` (biohub) — needed only by `scripts/encode.py` and
  `scripts/encode_pretraining.py` to run ESMC-300M locally
  (`uv pip install esm`).
- antiSMASH 8 — needed to annotate GTDB genomes and for the external comparison.

Raw genomes, antiSMASH output and derived parquet files are not tracked.