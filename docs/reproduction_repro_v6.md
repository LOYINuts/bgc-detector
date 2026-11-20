# 复现记录：repro_v6（2026-11-16）

本仓库固定配方在服务器上的完整复现：5 个 seed 从头预训练 → 正式训练 → NG 重叠窗口评估。
**结论：预训练曲线、正式训练选点与 NG 指标与历史实验逐位一致（bit-identical）。**

## 运行信息

| 项 | 值 |
| --- | --- |
| 源码 commit | `a1d22c3`（`outputs/repro_v6/source_commit.txt`） |
| 节点 | `node002`，Linux 4.18，2TB RAM |
| GPU | RTX A6000 ×2 + A100X ×1，5 个 seed 全部并排（映射 `0 1 2 0 1`） |
| PyTorch | 2.14.1+cu126（CUDA 12.6） |
| CPU 线程 | `OMP_NUM_THREADS=2`，MKL/OpenBLAS 同值 |
| 数据根 | `/home/gonghang/codes/my-bgc-v2/data`（共享 NFS，只读，未复制到本机盘） |
| 输出根 | `outputs/repro_v6` |
| 时间 | 启动 2026-11-16 08:46:35 EST，结束 11:07:14 EST，总墙钟 **2 小时 21 分** |

运行命令（服务器仓库根目录）：

```bash
DATA_ROOT=/home/gonghang/codes/my-bgc-v2/data \
OUTPUT_ROOT=/home/gonghang/codes/bgc-detector/outputs/repro_v6 \
PRETRAIN_REUSE="" \
TRAIN_DS=/home/gonghang/codes/my-bgc-v2/data/derived/train_test/dataset_legacy_train.parquet \
VAL_DS=/home/gonghang/codes/my-bgc-v2/data/derived/train_test/dataset_legacy_val.parquet \
SEEDS="1506 1507 1508 1509 1510" GPUS="0 1 2 0 1" \
bash scripts/run_final.sh
```

`TRAIN_DS`/`VAL_DS` 必须显式指向历史文件：数据根下同时存在同名新版
`dataset_train/val.parquet`，runner 默认会优先取它，从而偏离可复现输入。

## 输入校验

runner 在启动任何 seed 之前完整复算一次 sha256，写入
`outputs/repro_v6/pretraining_input_verification.json`；各 seed 只复核文件身份
（size、mtime、ctime、device、inode），不再重读 82.6GB 缓存。全部通过：

| 输入 | SHA256 |
| --- | --- |
| `dataset_legacy_train.parquet` | `53663a571ce2b995333a54e50aa098e1db135ebdeddc5a4f05c2604f5b137eae` |
| `dataset_legacy_val.parquet` | `ff8a0c6fc9a892dc89e19368b8f42565fd6b216a7f58058c2d2348ce2b4d926d` |
| `gene_embeddings_960d.parquet` | `eed6d18a9eead4ade0035e6c3ac0aae909e1b72b020d323b64e95917b8888629` |
| `windows_train.parquet` | `854a485ab411e3ee7d64b0b6585bc70cd4ca7613f58fdab59fe109170743dd11` |
| `windows_val.parquet` | `55962549f0b7a9783b04190347db7e5f895ce3d681601062a32218366e6fd90c` |
| `proteins.parquet` | `a07c8af963af34237e59622797548c3d40e8445969e0fa7a9802a53088ccfca5` |
| 预训练 `embeddings.npy` | `635df0fc70bbd01f88940a58f4c5ad782be0bd9ea59ae2cdc5c74c53235c8e11` |

`audit.json` 与缓存 `metadata.json` 状态均为 `completed`，模型 `biohub/ESMC-300M`、维度 960。

## 预训练

配方固定：batch 32、AdamW lr 1e-4 wd 0.01、clip 1、最多 30 epoch、patience 5，按预训练验证
loss 选点，semantic skip 参与训练。

| seed | 选点 epoch | best val loss | 实际 epoch 数 |
| --- | --- | --- | --- |
| 1506 | 4 | 0.054972 | 9 |
| 1507 | 4 | 0.057144 | 9 |
| 1508 | 3 | 0.051750 | 8 |
| 1509 | 4 | 0.053544 | 9 |
| 1510 | 3 | 0.051609 | 8 |

逐 epoch 的 `train_loss`/`val_loss` 与历史 `runs/pretrain_weak_semantic1506_v1_seed<seed>/history.json`
完全相等（9/9、9/9、8/8、9/9、8/8 条记录逐字段相同）。

## 正式训练

配方固定：batch 64、EMA 0.999、head lr 5e-4、encoder lr ×0.2、wd 0.01、无 warmup、
指数衰减 0.95（epoch 5 后首次衰减）、最多 100 epoch、patience 20，按验证 loss 选 EMA
checkpoint，验证 MCC 选阈值。

| seed | 选点 epoch | best val loss | 实际 epoch 数 | 阈值（验证 MCC） |
| --- | --- | --- | --- | --- |
| 1506 | 13 | 0.0059 | 33 | 0.6901 |
| 1507 | 17 | 0.0053 | 37 | 0.7266 |
| 1508 | 13 | 0.0051 | 33 | 0.7684 |
| 1509 | 12 | 0.0051 | 32 | 0.7036 |
| 1510 | 12 | 0.0049 | 32 | 0.7946 |

## NG 测试集结果

口径：128 基因窗口、stride 64、mean 聚合、无区域后处理，阈值来自验证 MCC。
共 67,148 个基因（3,772 正样本），无缺失 embedding。

| seed | AUROC | AUPRC | accuracy | precision | recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1506 | 0.9279 | 0.6712 | 0.9538 | 0.5762 | 0.6736 | 0.6211 | 0.5988 |
| 1507 | 0.9321 | 0.6776 | 0.9545 | 0.5816 | 0.6758 | 0.6251 | 0.6030 |
| 1508 | 0.9253 | 0.6729 | 0.9544 | 0.5839 | 0.6540 | 0.6170 | 0.5939 |
| 1509 | 0.9225 | 0.6819 | 0.9568 | 0.6042 | 0.6689 | 0.6349 | 0.6129 |
| 1510 | 0.9325 | 0.6850 | 0.9567 | 0.6039 | 0.6654 | 0.6332 | 0.6111 |

5 seed 汇总：

| 指标 | 均值 | 标准差 | 最小 | 最大 |
| --- | --- | --- | --- | --- |
| AUROC | 0.9281 | 0.0043 | 0.9225 | 0.9325 |
| AUPRC | 0.6777 | 0.0058 | 0.6712 | 0.6850 |
| accuracy | 0.9552 | 0.0014 | 0.9538 | 0.9568 |
| precision | 0.5900 | 0.0132 | 0.5762 | 0.6042 |
| recall | 0.6675 | 0.0086 | 0.6540 | 0.6758 |
| F1 | 0.6263 | 0.0077 | 0.6170 | 0.6349 |
| MCC | 0.6039 | 0.0080 | 0.5939 | 0.6129 |

## 与历史结果的比对

对照 `my-bgc-v2/runs/semantic_ema_weak_nocontext_presemantic1506_v1_seed<seed>/ng_validation_selected_overlap_s64_mean.json`：

- 5 个 seed 的 AUROC、AUPRC、accuracy、precision、recall、F1、MCC、阈值**全部按浮点逐位相等**
- 预训练 `history.json` 的每个 epoch 双 loss 也逐位相等

也就是说，这不是"数值接近"，而是同 seed、同输入、同配方下的确定性重跑。能成立的前提是
本记录顶部的 sha256 全部未变、且 `TRAIN_DS`/`VAL_DS` 指向历史文件。

## 运行观察（非配方问题）

- 预训练每 epoch 11,004 个 batch。本次 5 个进程全并排、共用 3 张卡，单 epoch 训练段
  668–752 秒（约 11–12.5 分钟），加验证约 2 分钟；正式训练每 epoch 21–42 秒。
  单条链路（预训练 + 正式训练 + NG 评估）合计 2 小时 21 分。
- 预训练缓存 82.6GB 走 `mmap`，速度完全取决于 page cache 命中：命中时约 0.07ms/窗口，
  未命中回到网络/磁盘随机读时约 166ms/窗口（约 2000 倍），因此日志每 100 个 batch 打印
  `batches/s`，正常 5–7，掉到 0.1 以下即说明页被回收。批次 SHA 校验时会把整个文件顺序读
  一遍，正好完成预热；`POSIX_FADV_WILLNEED` 只是补充提示，不保证驻留。
- 本次未使用任何本机 NVMe 中转副本，全部直接读共享 NFS。

## 产物位置

| 内容 | 路径（服务器） |
| --- | --- |
| 预训练 | `outputs/repro_v6/pretrain/seed<seed>/{pretrained_best.pt,history.json,status.json}` |
| 正式训练 | `outputs/repro_v6/runs/final_seed<seed>/detector_960d_best.pt`、`detector_960d_epoch*.pt` |
| NG 指标 | `outputs/repro_v6/runs/final_seed<seed>/ng_metrics.json` |
| 日志 | `outputs/repro_v6/runs/final_seed<seed>.logs/{pretrain,train,eval_overlap}.log` |
| 批次校验记录 | `outputs/repro_v6/pretraining_input_verification.json` |
| 启动与收尾日志 | `/home/gonghang/restart_from_nfs.log` |

## 复跑注意

- 输出目录必须全新；`pretrain.py`/`train.py` 都拒绝已存在的输出目录，不支持断点续训。
- 不要省略 `TRAIN_DS`/`VAL_DS`，否则 runner 可能选到新版 `dataset_train/val.parquet`。
- 5 seed 的指标需整体报告，不要只挑单 seed；阈值随 seed 变化，不可跨 seed 复用。

## 第二次独立复现：repro_v7_esm（2026-11-19）

同一条 `scripts/run_final.sh` 链路、同一份输入，在新输出根 `outputs/repro_v7_esm/retrain`
重跑一遍（commit `5f753b6`，启动 2026-11-19 05:45:29 EST，结束 11-20 03:0x EST）。

结论：5 个 seed 的 NG 指标、阈值、预训练选点与 repro_v6 **逐位相同**（`abs diff < 1e-12`）。

| seed | 预训练 best epoch / val loss | 正式选点 epoch | 阈值 | MCC | F1 | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1506 | 4 / 0.054972 | 13 | 0.6901 | 0.5988 | 0.6211 | 0.9279 | 0.6712 |
| 1507 | 4 / 0.057144 | 17 | 0.7266 | 0.6030 | 0.6251 | 0.9321 | 0.6776 |
| 1508 | 3 / 0.051750 | 13 | 0.7684 | 0.5939 | 0.6170 | 0.9253 | 0.6729 |
| 1509 | 4 / 0.053544 | 12 | 0.7036 | 0.6129 | 0.6349 | 0.9225 | 0.6819 |
| 1510 | 3 / 0.051609 | 12 | 0.7946 | 0.6111 | 0.6332 | 0.9325 | 0.6850 |
| 均值 | — | — | — | 0.6039 ± 0.0080 | 0.6263 ± 0.0077 | 0.9280 ± 0.0043 | 0.6777 ± 0.0058 |

同一次运行还验证了预测链路：`bgc-detector predict` 用仓库自带的 `esm` 路径现算嵌入
（`embeddings_from: computed`），两个基因组的逐基因判定与之前用另一套加载器得到的结果完全一致
（*M. genitalium* 0/504；*S. coelicolor* 284/7,593，26 个连续区域）。
