# bgc-detector

用冻结的 ESMC-300M 基因嵌入，逐基因判断细菌次生代谢基因簇（BGC）。

## 模型

```text
ESMC-300M 960d per-gene embedding（冻结）
  -> LayerNorm
  -> Channel-Gated Conv-BiGRU（2 层，kernel 5）
  -> 960 -> 128 -> 32 -> 1
  + 零初始化 semantic skip
  -> sigmoid -> 每个基因的 BGC 概率
```

- 参数量 10,290,690
- 输入 128 基因窗口，stride 64，覆盖窗口取 mean
- 损失 weighted BCE（正样本权重 2）；AdamW，batch 64，head lr 5e-4，encoder lr ×0.2
- EMA 0.999，按验证 loss 选点；判定阈值按验证 MCC 选出并写入 checkpoint
- 预训练：antiSMASH 标注窗口上的弱监督，semantic skip 参与

配方固定，没有架构或超参开关。路径、seed、设备和 worker 数量是运行输入。

## 安装

```bash
uv sync --locked          # Python 3.12.x（见 .python-version）
```

CLI 入口是 `bgc-detector`。要现场算嵌入（即不给 `--emb-cache`）时，需要本地 ESMC-300M
权重，把 HF 缓存目录指向它：

```bash
export HF_HOME=/path/to/huggingface
export HF_HUB_OFFLINE=1   # 可选，只用本地缓存，不联网
```

## 用法

### 看 checkpoint

```text
$ uv run bgc-detector info --checkpoint runs/final_seed1506/detector_960d_best.pt
  kind: best  weights: ema
  epoch: 13  selection: val_loss = 0.005917138763311637
  threshold: 0.690058 (source=validation, metric=mcc)
```

### 预测一个基因组

```text
$ uv run bgc-detector predict \
    --genome genome.gbk \
    --checkpoint runs/final_seed1506/detector_960d_best.pt \
    --outdir output --name run1
```

输入是带 `CDS` 和 `translation` 的 GenBank，antiSMASH 与 ClusterFinder 注释都能用。

选项：

- `--threshold`：默认 `0.5`；写 `checkpoint` 则用 checkpoint 里的验证 MCC 阈值
- `--emb-cache`：复用或保存该基因组的嵌入，第二次预测同一基因组时不必重算
- `--device`：`cuda:0` 或 `cpu`，默认自动选
- `--name`：输出文件名后缀

输出两个文件：

- `output/predict.run1.genes.tsv`：contig、gene_id、locus_tag、start、end、strand、score、predicted
- `output/predict.run1.json`：输入与 checkpoint 的 sha256、阈值及来源、窗口参数、基因与窗口计数、代码 commit

`predicted` 就是 `score >= threshold`。没有 translation 或没有嵌入的基因不进结果，会在
stderr 提示条数。

## 训练与复现

```bash
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/outputs PRETRAIN_REUSE="" \
SEEDS="1506 1507 1508 1509 1510" GPUS="0 1 2" \
bash scripts/run_final.sh
```

每个 seed 依次做预训练、正式训练、NG 评估。`PRETRAIN_REUSE` 留空表示从头预训练，
`TRAIN_DS` / `VAL_DS` 可覆盖数据路径。输入 SHA256 在批次启动前统一核验一次。

- 复现记录：[docs/reproduction_repro_v6.md](docs/reproduction_repro_v6.md)
- 预测细节与核对：[docs/prediction.md](docs/prediction.md)

## 结果

NG 测试集 67,148 个基因（3,772 正样本），5 个 seed，阈值取各自验证 MCC 选点：

| 指标 | 均值 ± 标准差 |
| --- | --- |
| AUROC | 0.9281 ± 0.0043 |
| AUPRC | 0.6777 ± 0.0058 |
| F1 | 0.6263 ± 0.0077 |
| MCC | 0.6039 ± 0.0080 |

## 目录

```text
bgc_gnn/   model.py dataset.py train.py pretrain.py eval.py checkpoint.py
           predict.py esmc.py runtime.py verify_inputs.py cli.py
scripts/   数据构建、编码、外部评测、run_final.sh
tests/     uv run python -m unittest discover -s tests
```

## 注意

- 模型靠窗口内的相邻基因，128 个一组；短 contig 用尾部环绕补齐
- 现场算出的嵌入与训练时用的冻结 parquet 平均差约 1e-4（bf16 归约顺序不同），
  预测可用，但不是逐位相同
- 判定阈值随 seed 不同，报告指标时写清用的是哪一个
