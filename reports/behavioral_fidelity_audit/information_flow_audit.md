# 信息流与领域独立性审计

研究分支继承被动负轨迹试验。核查时 GitHub 共五个原有分支。下列为当时的 HEAD，当前报告不随分支后续移动而改变。用于本地运行和审查的 29 个源文件与配置逐一匹配父提交 Git blob ID，见 `pinned_sources.json`。

|原分支|审计时提交|本轮用途|
|---|---|---|
|main|539410dec1d5c4e4433bd9a57b10a309cd9a850d|模型、检查器、历史汇总|
|research/passive-negative-traces-20261117|7fdabb95501f9cc78cad119d7dea1f5517c1bc8d|分支起点，12 设置、96 张预测表、采集和训练协议|
|docs/archive-bisimulation-failure-analysis-20261109|670e0a003ce7bb1d6b7b270bad43b09fd47d4fec|Ranking 图证书、反事实、哈希清单|
|experiment/finite-trace-mlp-vs-jepa|73a40322b2c55113a8fa146c182272555b95eb59|MLP-900 与旧 300 轮协议的沿革，360 设置未计入本轮分母|
|experiment/good-bad-trace-sanity-20261005|b6231f702ddc1f4098b086696e90722afcf50d91|原始 paired negatives、部分观测转移与补自环检查的区别|

## 两种观测协议

|环节|R · Cross-Map Ranking Round 2|F · 逐图有限轨迹|
|---|---|---|
|合法输入|训练图的可见全图观测、动作、真实正例后继。40 个训练图共 4,736 条有标签边；每轮枚举这些边。目标 stress 图提供状态、动作、状态图像、标签和初态|已知有限状态目录、动作目录、初态、各状态标签、图尺寸。逐图正轨迹给出源、动作、后继；重复访问保留。B/C 另收整条负轨迹|
|观察边的身份|键必须包含地图身份。两个地图坐标相同并不表示同一状态。24 个 stress 图没有本图训练正例。代码亦按地图指纹检查训练旋转轨道和评估集合|来自 `goods` 的所有相邻三元组去重，包括 oracle 接受的候选。预算是自然正轨迹数，不能直接当作有标签样本数|
|负例采集|以已标记训练图的正确后继为正类，在同图已知候选库上计算 ranking CE|每条自然轨迹采八个预设候选，以一位有效性 oracle 标记，A/B/C 均得到被接受候选。B 用同数量旧式负例，C 用被拒候选。B 的构造/后缀重放访问环境；C 的标签也需要整轨迹有效性接口|
|模型及选择|epoch-100 权重热启动，重启优化器；三个候选、三个种子。用 20 个 validation 图的有标签准确率早停/选配置；选出 ranking。三个 validation-best 的额外 epoch 依次为 20、30、20。stress 是已看过的诊断集|每模型从头训练。MLP 900 轮、lr 0.001；JEPA 300 轮、lr 0.0003。固定最终 epoch，无本设置 heldout/真图/形式指标选优。不同结构、目标、步数使 MLP–JEPA 比较无法单独归因给架构|
|预测|context encoder + 动作嵌入预测 delta，context + delta 与该图所有合法状态的 EMA target 向量作欧氏距离，float32 排序|同类 JEPA 梯度目标及全目录最近邻。MLP 使用分类分数 argmax。符号比较器按字典序选一个可行补全|
|评价|冻结 stress 表、真实图、完整最大关系、失配 DAG、CTL/LTL 对；oracle 改边仅供存档反事实|完整环境只在采集和离线对照中有权使用。heldout 的整轨迹与训练去重，可能共享边。真实转移、形式 verdict、距离真目标的排名均只供事后评价|

**可定位的代码依据**

- `experiments/improve_top1_ranking.py` 的 `training_maps`、`train_epoch`、`run_candidate`、`select_model`；`configs/top1_ranking_protocol.json`；`experiments/top1_quality_diagnosis.py` 的 `split_audit`、`inspect_map`。恢复包内 `selection.json` 和各 seed 的 `validation.json` 记录实际选型。归档报告中的原始行为导出代码 revision 为 `21fab50923c6d88eee9a8ef92b7be66f9deb4ff2`，与后续主分支及归档提交分别记账。更早的权重训练没有重跑，代码路径和记录支持协议核查，不能据此重建研究者此前看过的所有结果。
- `experiments/passive_negative_traces.py` 的 `prepare`、`ValidityOracle.accepts`、`train`；`reports/passive_negative_traces_20261117/run/executed_runner.py` 保存实际 runner；`src/jepa_lmc/evaluation/trace_sanity.py` 的 `KnownProblem`、`make_hard_negatives`；`src/jepa_lmc/learning/finite_trace.py` 的 `from_traces`、`known_observations`、`fit_finite_model`、`predict_successors`。学习器不获得被篡改步索引。含真值的 runner 和学习器处于同一进程，隔离通过显式函数入参实现。
- `src/jepa_lmc/learning/passive_negative.py` 的 `encode_trace`、`analyze_version_space`、`trace_negative_loss`、`fit_with_negative_traces`。完整负轨迹排除一组边同时成立；重复的同一查询共用一个函数值。逐边 marginal support 无法表达所有联合约束。训练时 JEPA 在负平方距离上构造 logits，推理审计报告普通欧氏距离差。二者量纲不同。
- `src/jepa_lmc/learning/model.py` 的 `GridStateEncoder.forward`、`ActionConditionedPredictor.forward`；`learning/data.py:gridworld_observation` 和 `learning/ranking.py:same_map_ranking_loss`；`evaluation/dynamics.py` 中 `torch.cdist`、`argsort` 和 `action_override` 路径。四个输入通道含墙、危险、目标和 agent。编码器拼接卷积特征和显式行列/三角位置特征；预测器回加当前 context。零 delta 在 context/target 接近时可能偏向当前状态。该结构提供机制假设，本轮未操纵残差或重训编码器。未发现以自环回退覆盖 JEPA Top-1 的推理分支。
- `src/jepa_lmc/evaluation/structural.py:behavioral_metrics` 同时审计最大关系及独立 partition refinement；`verification/behavioral_relations.py` 实现 simulation 与 bisimulation。`benchmarks/ctl_suite.py`、`benchmarks/ltl_suite.py` 定义公式及 safety 标志。`verification/nuxmv.py` 导出、调用和读取后端结果。

## 本轮重建接口及审计边界

`audit_core.py:observations` 仅从正轨迹构造部分函数，冲突即报错。`project(pred, observed)` 只覆盖已见查询；`scores` 只接收模型、已知状态目录/标签、图尺寸。本轮未以 E2 投影训练模型，也未用负轨迹中的中间状态当作正边。self-loop 和循环置换动作基线只访问当前状态或同源其它动作的既有输出。循环置换使用目录中的固定顺序，改变动作身份，不依赖动作的几何含义。

`run_audit.py:record` 先生成原始、投影和基线表，再生成 evaluator 的真值。R 的原始文件同时保存真图和预测图；代码提取 `graphs.top1` 作为 E1，另行检验 `graphs.real`。历史集没有物理权限隔离。E3 的 score1、runner-up 和 gap 可在部署时计算；true rank、是否真实移动、带符号的“真后继至最近错误候选间隔”、错误 AUROC、false-safe 和与真实图的双模拟均需要额外答案。把后者用于解码或阈值搜索会泄漏。用模型自身 margin 给一张图颁发可靠证书同样缺乏依据。

**推广接口。** 投影、动作置换、版本空间、双向性质证书适用于可枚举且身份稳定的有限状态和离散动作。这里还假设已知标签/初态、确定性、总转移、可靠且语法完整的轨迹。学习器所见网格尺寸、完整合法位置、可见墙及真实 AP 是强先验；若实际传感器只看到局部图像或标签需估计，须重新解决状态识别和标注。它们属于当前 benchmark 的建模前提。通用算法不读取墙、相邻性、移动方向或终点布局。未做跨领域实验。
