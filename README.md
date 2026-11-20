# bgc-detector

Per-gene biosynthetic gene cluster (BGC) detection from frozen ESMC-300M embeddings.
One model, one fixed recipe.

[中文说明](README_ZH.md)

## Install

```bash
git clone https://github.com/LOYINuts/bgc-detecctor.git
cd bgc-detecctor
uv sync --locked          # Python 3.12, all dependencies including esm
```

### ESMC-300M weights (~1.3 GB, not shipped with the repo)

Either download them:

```bash
hf download biohub/ESMC-300M        # into the HF cache ($HF_HOME)
```

or skip the download and point the CLI at existing weights:

```bash
uv run bgc-detector predict ... --esmc-path /path/to/ESMC-300M-snapshot
```

`BGC_DETECTOR_ESMC=/path/to/snapshot` sets the same default. If neither a path nor a
cached copy exists, the first `predict` downloads the weights to `$HF_HOME`.

Offline machines can use the cache with `HF_HUB_OFFLINE=1`; a missing file then fails
instead of silently fetching. When the network is unreachable, `uv run` may stall on
resolution, so use `uv run --no-sync ...` once the environment is synced.

## Usage

Three subcommands; `bgc-detector --help` lists every option.

### Inspect a checkpoint

```text
$ uv run bgc-detector info --checkpoint runs/final_seed1506/detector_960d_best.pt
checkpoint: runs/final_seed1506/detector_960d_best.pt
  kind: best  weights: ema
  epoch: 13  selection: val_loss = 0.005917138763311637
  threshold: 0.690058 (source=validation, metric=mcc)
  trained on: torch 2.14.1+cu126, commit 3039dcf
```

### Predict a genome

```bash
uv run bgc-detector predict \
    --genome genome.gbk \
    --checkpoint runs/final_seed1506/detector_960d_best.pt \
    --threshold checkpoint \
    --outdir output --name genome
```

The input is a GenBank file whose CDS features carry `/translation` (antiSMASH,
ClusterFinder and plain NCBI downloads all work). Missing `locus_tag` values get a
generated name; genes without a translation are skipped with a warning; a file with no
CDS annotation needs gene prediction first.

Two output files share one stem (`--name`, defaulting to the genome file name):

| File | Contents |
| --- | --- |
| `output/genome.genes.tsv` | one row per gene: `contig`, `gene_id`, `locus_tag`, `start`, `end`, `strand`, `score`, `predicted` |
| `output/genome.json` | provenance: sha256 of genome and checkpoint, threshold and where it came from, window 128 / stride 64 / mean / no post-processing, ESMC model and the weights actually loaded, gene and window counts, code commit |

`predicted` means `score >= threshold`. Grouping rows into clusters is a separate step.

Common options:

- `--threshold`: `0.5` by default; `checkpoint` uses the validation-MCC threshold stored
  in the checkpoint; a number sets it explicitly
- `--genome a.gbk b.gbk`: several genomes in one run, model loaded once, separate outputs
- `--emb-cache out/genome.emb.parquet`: reuse embeddings so a repeat run skips ESMC
- `--esmc-path`: weights directory or HF repo id (default: newest local snapshot, then
  `biohub/ESMC-300M`)
- `--device cuda:0` or `cpu` (default: cuda when available); lower `--max-toks` if the
  GPU runs out of memory

### Merge genes into regions

```bash
uv run bgc-detector regions --genes output/genome.genes.tsv --max-gap 3 --min-count 2
```

Writes `output/genome.regions.tsv` with `contig`, `start`, `end`, `n_genes`, `mean_score`
and `max_score`. Consecutive called genes join across up to `--max-gap` uncalled genes;
regions with fewer than `--min-count` genes are dropped (defaults match BGC-Prophet's
`output`/`pipeline`). This is a reporting step: it changes neither the per-gene scores nor
the evaluated protocol.

## Model

```text
ESMC-300M 960d per-gene embedding (frozen)
  -> LayerNorm -> Channel-Gated Conv-BiGRU (2 layers, kernel 5)
  -> 960 -> 128 -> 32 -> 1, plus a zero-initialised semantic skip
  -> sigmoid -> per-gene BGC probability
```

- 10,290,690 parameters; 128-gene windows, stride 64, overlapping windows averaged
- Training: weighted BCE (positive weight 2), AdamW, batch 64, head lr 5e-4, encoder lr
  x0.2, wd 0.01, EMA 0.999, checkpoint selected by validation loss
- Pretraining: weak supervision on antiSMASH-labelled windows, semantic skip included
- The recipe is fixed: paths, seeds, devices and worker counts are runtime inputs, the
  architecture and hyperparameters are not

## Training and reproduction

```bash
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/outputs PRETRAIN_REUSE="" \
SEEDS="1506 1507 1508 1509 1510" GPUS="0 1 2 0 1" \
bash scripts/train/run_final.sh
```

Each seed runs pretraining, training and NG evaluation in turn. An empty
`PRETRAIN_REUSE` trains every seed from scratch; `TRAIN_DS` / `VAL_DS` override the data
paths; input SHA256 is verified once per batch before launch.

- Reproduction record: [docs/reproduction_repro_v6.md](docs/reproduction_repro_v6.md)
- Prediction details and checks: [docs/prediction.md](docs/prediction.md)

## Results

NG test set, 67,148 genes (3,772 positives), 5 seeds. Each seed uses the threshold
picked on its validation split, so thresholds differ between seeds (0.69-0.79 here).

| Metric | Mean ± SD |
| --- | --- |
| AUROC | 0.9280 ± 0.0043 |
| AUPRC | 0.6777 ± 0.0058 |
| F1 | 0.6263 ± 0.0077 |
| MCC | 0.6039 ± 0.0080 |

End-to-end results on two real genomes are in [docs/prediction.md](docs/prediction.md).

## Layout

```text
src/bgc_detector/   model dataset train pretrain eval checkpoint predict esmc
                    runtime verify_inputs cli
scripts/
  data/     antiSMASH audit, catalog/dataset/pretraining data builds
  encode/   ESMC gene embeddings
  eval/     NG nine-genome set, antiSMASH and DeepBGC comparisons
  train/    run_final.sh, run_repro_v7_esm.sh
  tools/    check_esmc.py embedding regression check
tests/              uv run python -m unittest discover -s tests
```

Scripts run as modules, e.g. `uv run python -m scripts.eval.eval_ng --help`.
