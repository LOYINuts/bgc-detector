# bgc-detector

用冻结的 ESMC-300M 基因嵌入，逐基因判断细菌次生代谢基因簇（BGC）。单个模型、固定配方。

[English README](README.md)

## 安装

```bash
git clone https://github.com/LOYINuts/bgc-detecctor.git
cd bgc-detecctor
uv sync --locked          # Python 3.12，全部依赖（含 esm）
```

### 还要有 ESMC-300M 权重（约 1.3 GB，不随仓库分发）

两种方式，任选其一：

```bash
hf download biohub/ESMC-300M        # 方式一：先下到 HF 缓存（默认 HF_HOME）
```

或者方式二：**不下载**，直接把已有权重目录交给 CLI：

```bash
uv run bgc-detector predict ... --esmc-path /path/to/ESMC-300M-snapshot
```

也可以设环境变量 `BGC_DETECTOR_ESMC=/path/to/snapshot` 省掉每次传参。既不给路径也没有缓存时，
第一次 `predict` 会自动下载到 `HF_HOME`（需要有网）。

离线机器用本地缓存即可，加 `HF_HUB_OFFLINE=1`，缺失就直接报错、不会偷偷联网。完全断网时
`uv run` 可能卡在解析上，环境已同步过就用 `uv run --no-sync ...` 跳过这一步。

## 用法

三个子命令，`bgc-detector --help` 看全部选项。

### 看 checkpoint

```text
$ uv run bgc-detector info --checkpoint runs/final_seed1506/detector_960d_best.pt
checkpoint: runs/final_seed1506/detector_960d_best.pt
  kind: best  weights: ema
  epoch: 13  selection: val_loss = 0.005917138763311637
  threshold: 0.690058 (source=validation, metric=mcc)
  trained on: torch 2.14.1+cu126, commit 3039dcf
```

### 预测一个基因组

```bash
uv run bgc-detector predict \
    --genome genome.gbk \
    --checkpoint runs/final_seed1506/detector_960d_best.pt \
    --threshold checkpoint \
    --outdir output --name genome
```

输入是带 `CDS` 和 `/translation` 的 GenBank（antiSMASH、ClusterFinder、NCBI 下载的都可）。
缺 `locus_tag` 会自动命名，缺 `translation` 的基因跳过并告警；没有 CDS 注释的文件需要先
做基因预测。

产物两个文件，共用同一个 stem（`--name` 指定，默认取基因组文件名，不含扩展名）：

| 文件 | 内容 |
| --- | --- |
| `output/genome.genes.tsv` | 每个基因一行：`contig`、`gene_id`、`locus_tag`、`start`、`end`、`strand`、`score`、`predicted` |
| `output/genome.json` | 溯源：输入与 checkpoint 的 sha256、阈值及来源、窗口 128 / stride 64 / mean / 无后处理、ESMC 模型与实际加载路径、基因与窗口计数、代码 commit |

`predicted` 就是 `score >= threshold`。合并成区域是单独一步。

常用开关：

- `--threshold`：默认 `0.5`；写 `checkpoint` 用训练时按验证 MCC 选出的阈值；也可给具体数值
- `--genome a.gbk b.gbk`：一次跑多个基因组，模型只加载一次，各写各的产物
- `--emb-cache out/genome.emb.parquet`：复用嵌入，第二次预测同一基因组不必重算 ESMC
- `--esmc-path`：ESMC 权重目录或 HF 仓库名（默认取最新本地快照，其次 `biohub/ESMC-300M`）
- `--device cuda:0` 或 `cpu`，默认自动选；`--max-toks` 在显存不足时调小

### 合并成 BGC 区域

逐基因分数之后，把相邻被判为 1 的基因并成区域（跨过至多 `--max-gap` 个未判基因，丢掉少于
`--min-count` 个基因的区域）：

```bash
uv run bgc-detector regions --genes output/genome.genes.tsv --max-gap 3 --min-count 2
```

写出 `output/genome.regions.tsv`：`contig`、`start`、`end`、`n_genes`、`mean_score`、
`max_score`。默认值与 BGC-Prophet 的 `output`/`pipeline` 一致。这只是展示层，不改逐基因
预测，也不参与评测口径。

## 模型

```text
ESMC-300M 960d 每基因嵌入（冻结）
  -> LayerNorm -> Channel-Gated Conv-BiGRU（2 层，kernel 5）
  -> 960 -> 128 -> 32 -> 1，加零初始化 semantic skip
  -> sigmoid -> 每基因 BGC 概率
```

- 参数量 10,290,690；输入 128 基因窗口，stride 64，重叠窗口取 mean
- 训练：weighted BCE（正样本权重 2），AdamW，batch 64，head lr 5e-4，encoder lr ×0.2，
  wd 0.01，EMA 0.999，按验证 loss 选点
- 预训练：antiSMASH 标注窗口的弱监督训练，semantic skip 参与
- 配方固定，没有架构或超参开关；路径、seed、设备、worker 数是运行输入

## 训练与复现

```bash
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/outputs PRETRAIN_REUSE="" \
SEEDS="1506 1507 1508 1509 1510" GPUS="0 1 2 0 1" \
bash scripts/train/run_final.sh
```

每个 seed 依次预训练、正式训练、NG 评估。`PRETRAIN_REUSE` 留空表示从头预训练，
`TRAIN_DS` / `VAL_DS` 可覆盖数据路径，输入 SHA256 在批次启动前统一核验一次。

- 复现记录：[docs/reproduction_repro_v6.md](docs/reproduction_repro_v6.md)
- 预测细节与核对：[docs/prediction.md](docs/prediction.md)

## 结果

NG 测试集 67,148 个基因（3,772 正样本），5 个 seed，阈值取各自验证 MCC 选点：

| 指标 | 均值 ± 标准差 |
| --- | --- |
| AUROC | 0.9280 ± 0.0043 |
| AUPRC | 0.6777 ± 0.0058 |
| F1 | 0.6263 ± 0.0077 |
| MCC | 0.6039 ± 0.0080 |

两个真实基因组的端到端结果见 [docs/prediction.md](docs/prediction.md)。

## 目录

```text
src/bgc_detector/   model dataset train pretrain eval checkpoint predict esmc
                    runtime verify_inputs cli
scripts/
  data/     antiSMASH 审计、catalog/数据集/预训练数据构建
  encode/   ESMC 基因嵌入
  eval/     NG 九基因组、antiSMASH 与 DeepBGC 对比
  train/    run_final.sh、run_repro_v7_esm.sh
  tools/    check_esmc.py 嵌入回归检查
tests/              uv run python -m unittest discover -s tests
```

脚本按模块运行，例如 `uv run python -m scripts.eval.eval_ng --help`。

## 注意

- 模型靠窗口内的相邻基因；短 contig 尾部环绕补齐，单基因证据不进入模型
- 现算嵌入是 bf16：在与产出 parquet 相同的机器/GPU 上逐位相同（8 个 NG 基因 max|Δ| = 0），
  换 GPU 可能有 4e-3 量级差异；`uv run python -m scripts.tools.check_esmc` 可复核
- 判定阈值随 seed 变化（本例 0.69–0.79）；报告指标时写清用的是哪一个。同一 checkpoint 在
  NG 上默认 0.5 得 MCC 0.5855，验证阈值得 0.5988
