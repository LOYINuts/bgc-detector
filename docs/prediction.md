# 用模型做预测（`bgc-detector` CLI）

`bgc-detector` 是这个仓库唯一的对外入口，模型配方固定，CLI 只接受运行输入（路径、设备、
embedding 缓存），不提供架构或超参数开关。

## 安装

```bash
uv sync --locked                     # 仓库环境
export HF_HOME=/home/gonghang/hfhome # 需要新算 embedding 时复用已有 ESMC-300M 权重
```

`predict` 只有在需要**现算** embedding 时才依赖 `esm` 包（Biohub 的 ESMC 实现，不在
transformers 里）。只给现成 parquet 打分不需要它。要装就是同一份来源：

```bash
uv add "esm @ git+https://github.com/Biohub/esm.git@26b0bc2b771e3e419ea74f445a5f35cc094a1509"
```

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

阈值就存在 checkpoint 里，读取时会强制校验它存在、来自验证集、按 MCC 选出；缺失直接报错，
不会回退到 0.5。

### 预测一个基因组

```bash
uv run bgc-detector predict \
  --genome genome.gbk \
  --checkpoint runs/final_seed1506/detector_960d_best.pt \
  --device cuda:0 \
  --outdir output --name myrun \
  --emb-cache output/myrun.emb.parquet
```

- 输入：带 `CDS` 与 `translation` 的 GenBank（antiSMASH / ClusterFinder 注释均可）
- 每条 contig 按基因起点排序，128 基因窗口、stride 64、尾部环绕补满，被覆盖窗口取 mean
- 输出两件事：
  - `output/predict.myrun.genes.tsv`：`contig, gene_id, locus_tag, start, end, strand, score, predicted`
  - `output/predict.myrun.json`：溯源（genome/checkpoint 的 sha256、阈值及其来源、窗口/聚合参数、
    ESMC 模型名、基因与窗口计数、代码 commit）
- `--emb-cache` 给出时：存在就复用，不存在就先算再写入，方便同一个基因组反复预测
- 没有 `translation` 的基因、以及没有 embedding 的基因不进结果，并在 stderr 警告

## 与评测口径一致

窗口打分与 GenBank 解析已抽到 `bgc_gnn/predict.py`，`scripts/eval_ng.py` 与 CLI 调用同一个
函数，因此不存在"预测一套、评测另一套"的风险。

实测核对（seed 1506 checkpoint、NG 九基因组、同一份 NG embedding）：

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

## 尚未验证

用真实 ESMC 现算 embedding 这条路只做了单元测试（用桩替换 `embed_sequences`），没有在服务器
上跑过真模型。启用前应先做一次一致性检查：取一小批已知基因现算 embedding，与冻结 parquet
比对最大绝对差。ESMC 缺少 `xformers`/`flash-attn` 时会退回 PyTorch SDPA，bf16 注意力归约
顺序与融合内核不同，理论上会有微小差异，需要实测确认后再声称"与训练时同分布"。
