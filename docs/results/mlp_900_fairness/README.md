# 当前有限轨迹 MLP 基线：900 epochs

本目录发布 Experiment 04 中预先锁定的 **A 配置**，替代当前实验入口的欠拟合 MLP 基线。360 个 seed–map–budget 各有独立权重，并不是一个跨地图通用模型。

- 架构不变：Flatten + 8 维动作嵌入 → Linear(152,64) → GELU → Linear(64,|S|)。
- AdamW，lr=0.001，weight decay=0.0001，900 次全批更新，从头训练，固定最后一轮。
- 原 24 张地图、3 seeds、预算 1/2/4/8/16；每条轨迹 16 步。训练只用原已观察转移，未重新采样。
- JEPA 保持原架构、lr=0.0003、300 epochs 及历史结果。
- [原 300 轮配置](../../../configs/finite_trace_mlp_vs_jepa.json)和[原结果](../finite_trace_mlp_vs_jepa/)完整保留。论文未修改。

## 已完成实验的事实

全部 360 例的 observed unique / repeated-step accuracy 都是 **100%**；最大最终训练 CE 为 0.019311。参数量为 11,774–11,904；总更新 324,000 次，总训练样本呈现 32,140,800 次。本机优化循环耗时合计 593.126 秒；逐例实际耗时见 runs.jsonl 和 csv/training_costs.csv。历史 JEPA 计时来自不同机器，不代表相同 FLOPs 或严格等计算成本。

| 宏平均，360 例 | 原 MLP 300 | 当前 MLP 900 | 原 JEPA 300 |
|---|---:|---:|---:|
| 已观察 unique accuracy | 35.77% | 100% | 99.55% |
| 未观察 moving accuracy | 0.262% | 0.277% | 1.709% |
| 未观察 self-loop accuracy | 0.418% | 1.532% | 96.327% |
| 全部 unseen accuracy | 0.307% | 0.634% | 28.424% |
| Held-out trace acceptance | 0.456% | 23.893% | 27.743% |

训练不足已得到纠正，但未知移动转移的恢复仍很弱；不能将本次替换解释为良好泛化。JEPA 的全部 unseen 优势约 96.4% 来自 self-loop 正确数差，且这不是 JEPA 机制的因果识别。剔除九例 Pilot 后结论仍成立，完整分组见 CSV。两种动作语义下初态双模拟仍为 120/360、identity 双模拟为 0/360；部分 CTL/LTL 一致率提高不等于行为等价性改善。

配对分析按同一 seed–map–budget 对齐；报告地图 family/layout 分块、剔除 Pilot 和留一组敏感性结果。旋转、嵌套预算和共享 seed 存在依赖性，未把 360 例或所有预测边作为独立样本做显著性推断。900 轮是预先固定的 A 配置，本次发布没有调参、重训全部模型或重新选择 checkpoint。

## 权重与使用

[weights.zip](weights.zip) 包含全部 360 个原始 PyTorch state_dict 文件，名称为 `SEED_MAP_bBUDGET_mlp_A_900.pt`。文件哈希和参数哈希在 runs.jsonl；状态顺序、动作顺序及原训练轨迹在 training_inputs.json。不要在其他地图上直接解释该分类头的输出索引。

在项目根目录安装依赖并设置 `PYTHONPATH=src` 后：

```bash
# 加载所有权重并核对保存预测及已观察样本拟合，不调用环境或评价器
python -m experiments.mlp900_checkpoint

# 加载一个案例并打印预测
python -m experiments.mlp900_checkpoint --key 20260804_sealed_region_layout0_rot0_b1

# 以当前默认 MLP 900 / 原 JEPA 300 配置从头运行；NEW_DIR 必须尚不存在
python -m experiments.finite_trace_mlp_vs_jepa --output-dir NEW_DIR

# 复现历史 MLP 300 / JEPA 300 协议时显式使用旧配置
python -m experiments.finite_trace_mlp_vs_jepa --config configs/finite_trace_mlp_vs_jepa.json --output-dir ANOTHER_NEW_DIR
```

完整评价需设置 `NUXMV_BINARY`。旧 complete_finite_trace_run 工具仅用于原 300/300 协议，不用于当前 900/300 结果。本次上传不触发全量 benchmark 自动重跑。

## 文件

- `cases.jsonl`：360 个 MLP 900 逐例预测和完整评价。
- `runs.jsonl`：360 次实际训练的初始化、数据/权重哈希、样本数、参数量、更新数、耗时与学习曲线。
- `csv/`：原 Experiment 04 的 MLP 900 与两个历史对照的逐例、汇总、配对、分块、学习曲线及成本表；去除了涉及 1800 轮模型的行，数值未改。
- `experiment04_provenance/`：完整 A/B 实验的原始协议、验证记录、环境及中文研究报告。该处保留 1800 轮背景结果；本次没有发布 B 权重，也没有将 B 设为默认。
- `publication.json`、`manifest.json`：发布范围、来源及文件完整性记录。
- `release_validation.json`、`release_test_log.txt`：本次集成与发布前验证。

原实验 Windows/PyTorch 运行环境见 provenance/runtime.json。跨环境浮点差异可能影响权重逐位复现；提供的是已冻结权重，而非跨硬件逐位一致保证。
