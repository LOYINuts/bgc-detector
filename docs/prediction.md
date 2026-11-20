# 预测接口与核对记录

用法见 [README](../README.md)。这里记两件事：阈值怎么取，以及预测结果与评测、与训练嵌入
对得上到什么程度。

## 阈值

训练时按验证 MCC 扫描选出判定阈值，和权重一起存进 checkpoint（字段
`validation_threshold`，另附 `source=validation`、`metric=mcc`）。

`predict` 默认用 0.5。传 `--threshold checkpoint` 才改用 checkpoint 里的值。读取时会校验
该字段存在且来自验证集；缺失就报错，不会静默退回 0.5。输出 JSON 的 `threshold_source`
写明用的是哪一种（`fixed` 或 `validation`）。

同一个 checkpoint 在 NG 上的差别（seed 1506）：

| 阈值 | F1 | MCC |
| --- | --- | --- |
| 0.5（默认） | 0.6053 | 0.5855 |
| 0.6901（验证 MCC 选点） | 0.6211 | 0.5988 |

## 区域合并

`predict` 只给逐基因分数。`regions` 负责把相邻的 `predicted=1` 并成区域，规则和 BGC-Prophet
的 `output` / `pipeline` 相同：跨过 `--max-gap` 个未判基因（默认 3），丢掉少于 `--min-count`
个基因的区域（默认 2）。输出 `<stem>.regions.tsv`，列为 `contig`、`start`、`end`、`n_genes`、
`mean_score`、`max_score`。

这一步是展示层，不参与评测：NG 指标、阈值选择、与 `eval_ng` 的核对都只用逐基因分数。

## 与评测一致

GenBank 解析和窗口打分只有一份实现（`bgc_gnn/predict.py`），`scripts/eval_ng.py` 与 CLI
都调用它。核对条件：seed 1506 checkpoint、NG 九基因组、同一份 NG embedding、`--threshold checkpoint`。

| 项 | 结果 |
| --- | --- |
| 基因 / 窗口 | 67,148 / 1,055，与评测一致 |
| 阈值 | 0.6900579333305359，两侧相同 |
| 判为 BGC | 4,410 个，两侧相同 |
| 逐基因分数 | 与 `eval_ng --dump-gene-scores` 相同（按该 dump 的 float32 精度，0 个不同） |
| F1 / MCC | 0.62112 / 0.598764，两侧相同 |
| 同一命令跑两次 | 逐位相同 |

`eval_ng` 的 dump 存 float32，TSV 写 float64 的窗口均值，原始数字可能差半个 float32 ULP，
判定与指标不受影响。

## 现算嵌入

ESMC-300M 用 `esm` 包的 `EsmcModel` 加载（`my-bgc-v2/scripts/encode.py` 里的原版写法），
bf16，与训练时那份嵌入同源。注意不要用 transformers 的 ESMC：对这份 checkpoint 它会按自己
的默认几何建出 2560 宽、80 层模型，权重全部 MISSING（`scripts/check_esmc.py` 会直接判定
不可用）。

用 `scripts/check_esmc.py` 复核（8 个 NG 基因，`esm` 3.5.0 + 本地权重）：

| 项 | 值 |
| --- | --- |
| max abs diff | 0.0039 |
| 平均 abs diff | 1.8e-3 |
| 训练期参考（fp32 路线） | max 0.0037 / mean 1.5e-4 |

差值都是 bf16 的量化台阶（0.00098 的整数倍），判定为可用。

差值都是 bf16 的量化台阶（0.00098 的整数倍），预测够用，但不是逐位相同。

走过一次回头路记在这里：一开始用 transformers 的 `AutoModel` 加载同一份权重，它按自己的默认
几何建出 2560 宽 / 80 层，checkpoint 的键全部 `UNEXPECTED`、模型侧全部 `MISSING`；命令行
`--threshold` 之外的输出看起来都正常，嵌入却是随机的。`scripts/check_esmc.py` 就是防这个的
回归检查。

## 端到端检查

两个从 NCBI 下载的基因组，阈值取 checkpoint 的 0.6901：

| 基因组 | 基因数 | 判为 BGC | 比例 | 连续区域数 |
| --- | --- | --- | --- | --- |
| *Mycoplasmoides genitalium* G37（无已知 BGC） | 504 | 0 | 0.00% | 0 |
| *Streptomyces coelicolor* A3(2)（BGC 丰富） | 7,593 | 284 | 3.74% | 26 |

coelicolor 的 26 个区域，最大几个是 41、33、24、23、23、21 个基因，和文献里约 25–30 个
BGC 的数量级对得上；被判阳性的基因中位分数 0.9966。阴性对照一个都没判。
