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

ESMC-300M 用 transformers 加载（`ESMCModel`），bf16。用训练时那份本地权重重算 NG 里 40 个
基因，与冻结 parquet 比：

| 项 | 值 |
| --- | --- |
| max abs diff | 0.0037 |
| mean abs diff | 1.5e-4 |
| 参考量级（分量绝对值均值） | 0.0155 |

差值来自 bf16 注意力归约顺序（没装 xformers/flash-attn 时走 PyTorch SDPA）。预测够用，
但不能说逐位相同。

另外一条弯路记在这里：不要用 Biohub 的 `esm` 包加载这份权重。它的本地注册表指向另一套布局
（`esmc-300m-2024-12` 的 `data/`），硬套 safetensors 会缺 308 个、多 398 个张量，模型基本
是随机初始化；用它算出的嵌入与训练用的差到 max |Δ| 4.45。

## 端到端检查

两个从 NCBI 下载的基因组，阈值取 checkpoint 的 0.6901：

| 基因组 | 基因数 | 判为 BGC | 比例 | 连续区域数 |
| --- | --- | --- | --- | --- |
| *Mycoplasmoides genitalium* G37（无已知 BGC） | 504 | 0 | 0.00% | 0 |
| *Streptomyces coelicolor* A3(2)（BGC 丰富） | 7,593 | 284 | 3.74% | 26 |

coelicolor 的 26 个区域，最大几个是 41、33、24、23、23、21 个基因，和文献里约 25–30 个
BGC 的数量级对得上；被判阳性的基因中位分数 0.9966。阴性对照一个都没判。
