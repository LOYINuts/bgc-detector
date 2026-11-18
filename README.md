# bgc-detector

从冻结的 ESMC-300M 基因嵌入逐基因预测细菌次生代谢基因簇（BGC）的检测器。单个模型、固定配方，
仓库只包含最终模型与产出它的流水线。

## 模型

```text
frozen ESMC-300M 960d gene embeddings
  -> LayerNorm
  -> Channel-Gated Conv-BiGRU（2 层，d_model 960，kernel 5，gate bias -3.0）
  -> 检测头 960 -> 128 -> 32 -> 1
  + 零初始化的 semantic skip（单基因线性分支）
  -> sigmoid -> 每个基因的 BGC 概率
```

| 项 | 值 |
| --- | --- |
| 参数量 | 10,290,690 |
| 输入 | 每基因 ESMC-300M 960 维均值池化嵌入（冻结） |
| 上下文 | 128 基因窗口，stride 64，被覆盖窗口取 mean |
| 损失 | weighted BCE，正样本权重 2 |
| 训练 | AdamW，batch 64，head lr 5e-4，encoder lr ×0.2，wd 0.01 |
| EMA | 0.999（验证与导出都用 EMA 权重） |
| 学习率 | 指数衰减 0.95，epoch 5 后首次衰减，无 warmup |
| 选点 | 验证 loss；判定阈值按验证 MCC 选出并写进 checkpoint |
| 预训练 | antiSMASH 标注窗口上的弱监督预训练，semantic skip 参与 |
| 测试口径 | NG 九基因组，128 窗口 / stride 64 / mean，无区域后处理 |

没有架构、超参或分支开关：只有路径、seed、设备、worker 数是运行输入。

## 安装

需要 Python ≥3.12，用 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --locked
```

CLI 入口是 `bgc-detector`。只有需要**现算** embedding 时才额外依赖 `transformers`
（ESMC-300M 是 transformers 原生架构）：

```bash
uv pip install transformers
export HF_HOME=/path/to/huggingface   # 复用已有 ESMC-300M 权重
export HF_HUB_OFFLINE=1              # 可选：只用本地缓存，不联网
```

只要给出已算好的 embedding parquet，打分不需要 `transformers`。

## 用法

### 查看 checkpoint

```bash
uv run bgc-detector info --checkpoint runs/final_seed1506/detector_960d_best.pt
```

```text
  kind: best  weights: ema
  epoch: 13  selection: val_loss = 0.005917138763311637
  threshold: 0.690058 (source=validation, metric=mcc)
```

### 预测一个基因组

```bash
uv run bgc-detector predict \
  --genome genome.gbk \
  --checkpoint runs/final_seed1506/detector_960d_best.pt \
  --device cuda:0 \
  --outdir output --name myrun
```

- 输入：带 `CDS` 与 `translation` 的 GenBank（antiSMASH / ClusterFinder 注释均可）
- 判定阈值**默认 0.5**；要改用 checkpoint 里存的验证 MCC 阈值，加 `--threshold checkpoint`
- 不给 `--emb-cache` 时现场用 ESMC-300M 算嵌入，这一步需要 `transformers` 与本地权重
- `--emb-cache path.parquet`：已存在就复用（不再加载 ESMC），不存在就先算再写入，同一个基因组重复预测不必重算
- 没有 `translation` 或没有 embedding 的基因不进入结果，会在 stderr 警告

输出：

| 文件 | 内容 |
| --- | --- |
| `output/predict.myrun.genes.tsv` | `contig, gene_id, locus_tag, start, end, strand, score, predicted` |
| `output/predict.myrun.json` | 溯源：genome 与 checkpoint 的 sha256、阈值及其来源、窗口/stride/聚合、ESMC 模型、基因与窗口计数、代码 commit |

`predicted` 是 `score >= threshold` 的结果。窗口打分与 GenBank 解析只有一份实现
（`bgc_gnn/predict.py`），`scripts/eval_ng.py` 与 CLI 共用，因此预测与评测口径不会分叉。

## 训练与复现

完整链路（每个 seed：预训练 → 正式训练 → NG 评估）：

```bash
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/outputs PRETRAIN_REUSE="" \
SEEDS="1506 1507 1508 1509 1510" GPUS="0 1 2" \
bash scripts/run_final.sh
```

`TRAIN_DS` / `VAL_DS` 等可覆盖输入位置；空 `PRETRAIN_REUSE` 表示每个 seed 都从头预训练。
输入 SHA256 由 runner 在每个批次启动前统一核验一次，各 seed 只复核文件身份。

复现记录见 [`docs/reproduction_repro_v6.md`](docs/reproduction_repro_v6.md)：5 个 seed 的
预训练曲线、选点与 NG 指标与历史实验逐位一致。预测 CLI 的细节与核对见
[`docs/prediction.md`](docs/prediction.md)。

## 结果（NG 测试集，5 seed）

| 指标 | 均值 ± 标准差 |
| --- | --- |
| AUROC | 0.9281 ± 0.0043 |
| AUPRC | 0.6777 ± 0.0058 |
| F1 | 0.6263 ± 0.0077 |
| MCC | 0.6039 ± 0.0080 |

67,148 个基因（3,772 正样本），阈值取各 seed 的验证 MCC 选点。

## 目录

```text
bgc_gnn/
  model.py      BGCDetector（唯一模型）
  dataset.py    嵌入读取与固定窗口数据集
  train.py      正式训练（固定配方）
  pretrain.py   弱监督预训练
  eval.py       指标与验证阈值选择
  checkpoint.py checkpoint 负载、溯源、预训练迁移校验
  predict.py    GenBank 解析与窗口打分（CLI 与评测共用）
  esmc.py       ESMC-300M 嵌入（transformers，延迟导入）
  runtime.py    CPU 线程数与映射文件预读
  verify_inputs.py  批次输入 SHA256 核验与 receipt
  cli.py        bgc-detector 命令行
scripts/        数据构建、编码、外部对比评测、runner
tests/          unittest（uv run python -m unittest discover -s tests）
```

## 已知限制

- 窗口依赖 128 个相邻基因，短 contig 靠尾部环绕补满；单基因证据不进入模型
- 现算 embedding 需要 transformers + 本地 ESMC-300M 权重；用同一份权重重算与训练时的嵌入
  平均差约 1e-4 量级（最大 3.7e-3），足以复现预测，但不是逐位相同
- 判定阈值随 seed 变化，跨 seed 不可复用；报告指标时务必写明所用阈值
