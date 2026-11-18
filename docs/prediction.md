# 用模型做预测（`bgc-detector` CLI）

`bgc-detector` 是这个仓库唯一的对外入口，模型配方固定，CLI 只接受运行输入（路径、设备、
embedding 缓存），不提供架构或超参数开关。

## 安装

```bash
uv sync --locked                     # 仓库环境
export HF_HOME=/home/gonghang/hfhome # 需要新算 embedding 时复用已有 ESMC-300M 权重
```

`predict` 只有在需要**现算** embedding 时才依赖 `transformers`：ESMC-300M 是 transformers
原生架构（`ESMCModel`），训练时的冻结嵌入也是用这份权重算的。只给现成 parquet 打分不需要它。

```bash
uv pip install transformers    # 或装进运行 CLI 的那个环境
export HF_HOME=/path/to/huggingface
export HF_HUB_OFFLINE=1        # 只用本地缓存
```

注意不要用 Biohub 的 `esm` 包加载这份权重：它的本地注册表指向另一套布局
（`esmc-300m-2024-12` 的 `data/`），硬套 safetensors 会出现 308 个缺失、398 个多余张量，
模型大部分随机初始化——用它算出的嵌入与训练用的完全不同（实测 max |Δ| 4.45，而参考量级
只有 0.0155）。

## 子命令

### 查看 checkpoint

```bash
uv run bgc-detector info --checkpoint runs/final_seed1506/detector_960d_best.pt
```

```text
checkpoint: runs/final_seed1506/detector_960d_best.pt
  kind: best  weights: ema
  epoch: 13  selection: val_loss = 0.005917138763311637
  threshold: 0.690058 (source=validation, metric=mcc)
```

阈值本身也存在 checkpoint 里。`predict` 默认用 0.5；加 `--threshold checkpoint` 才改用它，
读取时强制校验该阈值存在、来自验证集、按 MCC 选出，缺失则报错而不是静默回退。

### 预测一个基因组

```bash
uv run bgc-detector predict \
  --genome genome.gbk \
  --checkpoint runs/final_seed1506/detector_960d_best.pt \
  --device cuda:0 \
  --outdir output --name myrun \
  --threshold checkpoint \
  --emb-cache output/myrun.emb.parquet
```

- 输入：带 `CDS` 与 `translation` 的 GenBank（antiSMASH / ClusterFinder 注释均可）
- 每条 contig 按基因起点排序，128 基因窗口、stride 64、尾部环绕补满，被覆盖窗口取 mean
- 判定阈值：默认 0.5（`threshold_source=fixed`）；`--threshold checkpoint` 用 checkpoint 的验证
  MCC 阈值（`threshold_source=validation`），也可以直接给一个 0–1 之间的数
- 输出两件事：
  - `output/predict.myrun.genes.tsv`：`contig, gene_id, locus_tag, start, end, strand, score, predicted`
  - `output/predict.myrun.json`：溯源（genome/checkpoint 的 sha256、阈值及其来源、窗口/聚合参数、
    ESMC 模型名、基因与窗口计数、代码 commit）
- `--emb-cache` 给出时：存在就复用，不存在就先算再写入，方便同一个基因组反复预测
- 没有 `translation` 的基因、以及没有 embedding 的基因不进结果，并在 stderr 警告

## 与评测口径一致

窗口打分与 GenBank 解析已抽到 `bgc_gnn/predict.py`，`scripts/eval_ng.py` 与 CLI 调用同一个
函数，因此不存在"预测一套、评测另一套"的风险。

实测核对（seed 1506 checkpoint、NG 九基因组、同一份 NG embedding、`--threshold checkpoint`）：

| 项 | 结果 |
| --- | --- |
| 基因数 / 窗口数 | 67,148 / 1,055（与评测一致） |
| 阈值 | 0.6900579333305359（两侧相同） |
| 判为 BGC 的基因数 | 4,410（两侧相同） |
| 逐基因分数 | 与 `eval_ng --dump-gene-scores` 一致到该 dump 的 float32 精度（0 个不同） |
| F1 / MCC | 0.62112 / 0.598764（两侧相同） |
| 同一命令跑两次 | 逐位相同（max diff 0.0） |

注意 `eval_ng` 的 dump 存的是 float32，而 TSV 写的是 float64 的窗口均值，所以原始数字可能
差半个 float32 ULP；判定与指标不受影响。

## 现算 embedding 的实测

用 transformers + 本地 ESMC-300M 权重（bf16）重算 NG 里 40 个基因的嵌入，与训练用的冻结
parquet 比对：

| 项 | 值 |
| --- | --- |
| max abs diff | 0.0037 |
| mean abs diff | 1.5e-4 |
| 参考量级（分量绝对值均值） | 0.0155 |
| 模型 | `ESMCModel`，332M 参数 |

差异来自 bf16 注意力归约顺序（未装 xformers/flash-attn 时走 PyTorch SDPA），量级约为分量
均值 1% 的平均误差、最大 24%。用于预测足够，但不能声称逐位相同。

## 端到端行为检查（两个 NCBI 下载的基因组）

| 基因组 | 基因数 | 判为 BGC | 比例 | 预测出的区域数 |
| --- | --- | --- | --- | --- |
| *Mycoplasmoides genitalium* G37（无已知 BGC） | 504 | 0 | 0.00% | 0 |
| *Streptomyces coelicolor* A3(2)（BGC 丰富） | 7,593 | 284 | 3.74% | 26（最大 41/33/24/23/23/21 个基因） |

coelicolor 预测出的 26 个连续区域与文献报道的约 25–30 个 BGC 数量级相符，被判阳性的基因
中位分数 0.9966；阴性对照一个都没判。
