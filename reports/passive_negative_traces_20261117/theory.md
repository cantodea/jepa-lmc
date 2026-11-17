# Good/Bad Traces：形式化信息分析与理论边界

日期：2026-11-17。此文为理论研究代理的独立分析；仓库级事实应以代码审计为准。本稿中的定理是针对下述假设类直接证明的结果，不冒充文献已发表的新定理。

## 1. 明确学习对象

固定有限状态集合 S、动作集合 A 及精确、跨轨迹一致的状态身份。令 n=|S|，候选模型为所有总确定性转移函数 f:S×A→S，记该假设类为 H。初始集合 I 若受限，必须固定且已知；若数据允许任意状态重置，轨迹有效性只检查转移。状态标签 L 可以用于事后行为评价，但不参与以下转移约束。本文不是只观察动作字符串、同时推断隐藏状态的 DFA 识别问题。

轨迹 τ=(s₀,a₀,s₁,…,aₗ₋₁,sₗ) 的接受公式为

Aτ(f) = [s₀∈I] ∧ ∧ᵢ[f(sᵢ,aᵢ)=sᵢ₊₁]。

任意状态重置时去掉初始项。正例要求 Aτ，负例要求 ¬Aτ。若起点已保证合法，则负例约束为

Cτ(f) = ∨ᵢ[f(sᵢ,aᵢ)≠sᵢ₊₁]。

负轨迹只说明至少一条边不成立，不能把每条边标成非法。令

ΦG = ∧g∈G Ag，H(G)={f∈H:ΦG(f)}；
ΦG,B = ΦG ∧ ∧b∈B ¬Ab，H(G,B)={f∈H:ΦG,B(f)}。

始终 H(G,B)⊆H(G)。以下非冗余及计数结论假设标签无噪声且 H(G,B) 非空；空空间应报告为数据/假设类不一致，不能报告为识别成功。

## 2. 配对状态替换负例的冗余定理

**定理 1。** 假设：(i) 状态和动作身份精确且一致；(ii) 候选模型是确定性的；(iii) 候选模型必须接受全部正例；(iv) 对每个 b∈B，存在正例中的边 (s,a,t) 及 b 中的边 (s,a,u)，且 u≠t。那么 H(G,B)=H(G)。

**证明。** 任取 f∈H(G)。因 f 接受含边 (s,a,t) 的正例，f(s,a)=t。确定性与 u≠t 蕴含 f(s,a)≠u，故 b 至少一条边不成立，f 拒绝 b。因此所有 f∈H(G) 都拒绝全部 B，即 H(G)⊆H(G,B)。结合反向包含得等式。证毕。

**配对扰动推论。** 若 b 由 g∈G 仅替换一个非初始状态 sⱼ（1≤j≤ℓ）为 u≠sⱼ得到，其他状态与动作不变，则原入边 (sⱼ₋₁,aⱼ₋₁,sⱼ) 固定正确后继，替换后的入边与其冲突。即使改变内点同时影响出边，入边一条就已证明 b 冗余。若 g 仅用于生成、却未加入 G，则不能套用此结论。

该结论不依赖于总性：部分确定性候选若接受正例也具有相同唯一后继。总性只用于后续计数。该结论同样适用于确定性候选的任意结构化子类。

不能推广的情形：

- **非确定性候选：** R 可同时包含 (s,a,t) 和 (s,a,u)。正例只保证前者存在，负例可以排除后者，因而提供信息。
- **初始状态替换：** 没有被固定的入边；合法性取决于起点协议，不能机械套用推论。
- **动作替换：** 改为未观测动作时没有固定后继；可能提供信息。
- **状态混叠或潜在状态：** 同一观测不必代表同一状态，直接函数约束不再适用。
- **软拟合网络：** 网络未必接受全部 G；B 可改变优化、正则化和误差分配，但这种改变不证明增加独立转移信息。
- **噪声数据或不一致 G：** H(G)=∅ 时等式可能平凡成立；应先检测一致性。

定理的强度是“配对标签没有增加该假设类中的硬约束”，并不是“训练中加入这些样本绝不可能改变神经网络表现”。

## 3. 一般负轨迹何时真正提供信息

先将 G 展开为已知部分函数 p:O→S，O⊆S×A 为观测过的不同状态—动作对。若 G 内同一 pair 出现冲突后继，则不一致。对 b 的接受要求按 pair 去重；同一 pair 要求两个不同后继，接受公式为假。

**定理 2。** 在全部总确定性函数构成的 H 中，假设 G 一致且 b 起点合法。b 相对 G 冗余，当且仅当以下至少一项成立：

1. b 含已观测 pair x∈O，但要求后继不同于 p(x)；
2. b 自身对某个 pair 要求两个不同后继。

**证明。** 任一条件都使 ΦG∧Ab 不可满足，故 G 蕴含 ¬Ab。反过来，若二者都不成立，G 与 b 的合并要求仍是部分函数，可以把剩余 pair 任意补全为总函数；这个模型接受 b，因此加入 ¬Ab 会排除至少一个 H(G) 中的模型。对于可信 oracle 判定的负例，真实模型拒绝 b 且属于 H(G)，故缩小是严格且非空的。证毕。

已知非法初始状态也使 b 冗余，但它约束的是起点协议，不能计为转移信息。若加入几何、连通性等结构先验，以上两条件仍充分，但不再必要；应针对实际假设类做 SAT 查询。

**例 1：负例可完全识别未观测转移。** S={0,1}，A={a}，G={(0,a,1)}。H(G) 有 2 个模型，区别仅为 f(1,a)。若 b=(0,a,1,a,1) 被判负，则 f(1,a)≠1，结合已知候选状态只有 0 和 1，推出 f(1,a)=0。H(G,{b}) 仅剩 1 个模型。这里两步轨迹的第一条边合法；把两条边都标负会直接破坏正例约束。

**例 2：全局空间缩小不等于边际候选数减少。** 同样 S={0,1}, A={a}，G=∅，b=(0,a,1,a,1)。原有 4 个函数，负例只排除 f(0,a)=1 且 f(1,a)=1 的函数，剩 3 个。每个 pair 的后继候选仍都是 {0,1}。因此仅报告每对候选数会遗漏负例产生的变量间关联。

**推论。** 在无额外结构先验的函数类中，可信且相对 G 非冗余的 b 必须涉及至少一个未观测 pair。反过来，包含未观测 pair 并不保证非冗余：轨迹可能已被另一条观测冲突边或自身确定性冲突否定。这正是为什么不能把“含 unseen pair”作为信息标签，更不应将其设为采样过滤条件。

## 4. 精确计数与信息量

设 m=|S||A|−|O|。在无结构约束的总确定性类中 |H(G)|=n^m。对一个相对 G 非冗余负例 b，设其接受要求涉及 k 个不同的未观测 pair，则恰有 n^(m−k) 个正例一致模型接受 b。因此

|H(G,{b})| = n^m−n^(m−k)，
Δ(b|G) = log₂|H(G)|−log₂|H(G,{b})| = −log₂(1−n^(−k))。

这里必须 k≥1，否则若 b 的全部要求都已由 G 固定，b 应被真实系统接受而不可能是可信负例。n≥2 时公式正常；n=1 没有合法起点上的可信转移负例。

Δ 是 **均匀版本空间下的对数体积缩减**。不能未经概率假设就称其为真实系统分布的 Shannon 信息量。一个实现会给所有未知 pair 独立均匀先验时，这个量等于观察到该负标签的自信息；它不是查询前期望信息增益。实际地图生成先验通常远非均匀函数先验。

较长随机坏轨迹虽然可能逻辑非冗余，但 k 大时每条排除比例只有 n^(−k)，信息极弱。这给出自然负例无明显恢复收益的一种理论解释，无需推断“模型没有能力”。轨迹长度与 k 不同，重复 pair 必须去重。

多个负例的体积缩减通常不能相加。应精确计算 |H(G,B)|，或采用顺序增量

Δⱼ=log₂|H(G,B<ⱼ)|−log₂|H(G,B≤ⱼ)|。

总和与顺序无关，单条增量与顺序有关；重复负例和彼此蕴含会导致后续增量为 0。单独相对 G 非冗余条数不等于独立约束条数。

**SAT/SMT 编码。** 为每个 (s,a) 建有限域变量 Xsa∈S。正轨迹给等式，负轨迹给析取不等式。对当前 Φ，判断新 b 是否有增量，应检查 SAT(Φ∧Ab)：不可满足即冗余；可满足即存在被该负例排除的模型（另检查 Φ∧¬Ab 保持非空）。通过 SAT(Φ∧Xsa=t) 对每个 t 求边际候选域。具体抽取一个满足模型的任意解不是识别证明，必须同时报告候选域和空间大小。

**精确与近似分界。** 小系统可枚举 n^m 个补全，或使用带模型阻断的 AllSAT 精确计数；SAT 可满足性及边际域查询是精确结论，即使没有总模型计数。大系统如只做 Monte Carlo 估计，零命中不是不可满足证据，边际未采到不是候选不存在，必须报告采样量、置信界与偏差。对于重叠负轨迹，用独立排除近似乘积会忽略相关性，不能当成精确 |H|。

## 5. 正确的神经轨迹级负例目标

令 qθ(t|s,a) 是后继的分类概率。引入一个**关于未知确定性函数的因子化分布**：每个 pair x 的 Fx 独立服从 qθ(·|x)，但同一 Fx 在整条轨迹中重复出现时是同一个随机变量。

对无自相矛盾的 b，令 U(b) 为其不同 pair→successor 要求，则

Pθ(Ab)=∏(x,t)∈U(b) qθ(t|x)，
ℓbad(b)=−log[1−Pθ(Ab)]。

若 b 对同一 pair 要求不同后继，Pθ(Ab)=0，ℓbad(b)=0。若初始协议不合法也同理。这个目标对“不接受整条轨迹”的逻辑语义正确，不把每条边当负例。

**不能逐 occurrence 直接相乘：** b=(0,a,0,a,0) 的接受事件仅为 F0a=0，概率是 qθ(0|0,a)，不是其平方。若使用每次访问都独立重新抽样的概率，描述的是随机转移过程而非未知确定性函数；需要明确更换理论对象。

训练可用 L=Lpositive+λ mean_b ℓbad(b)，λ、温度、训练步数在看测试指标前冻结。MLP 的 q 通常由分类 logits 得到；JEPA 可用与全体候选目标嵌入的距离经 softmax 得到 q，但此时新增分类接口、温度和原 JEPA 目标的组合均须明确报告。若改变原训练机制，不应笼统宣称“原 JEPA 仅增加负例”。

若以硬方式把 q 在 G 已观测 pair 上条件化为 p(x) 的点质量，定理 1 的冗余负例具有严格零损失；普通软网络未被如此条件化时仍可能有非零损失。这解释了信息冗余与优化影响可以同时成立。

对多条轨迹求和是 composite objective，并不等于 −log Pθ(∧b ¬Ab)；不同轨迹共享 pair 时不独立。应把单轨迹目标的语义正确性与联合概率模型区分。长轨迹的 Pθ(Ab) 可能极小，使负损失及梯度几乎消失；这是该目标的局限，而不是事后筛选短轨迹的理由。

数值实现应先求 log P，再用稳定的 log1mexp 计算 −log(1−exp(log P))；阈值裁剪须记录。测试应覆盖重复 pair、冲突 pair、两条边中仅一条不成立、全零概率、接近 1 的概率及有限梯度。

## 6. 被动学习与 oracle 数据采集的边界

固定种子、固定候选分布、模型训练前完成所有查询，排除了按模型表现选例的适应性。但候选轨迹仍由验证 oracle 返回成员资格标签，因此应精确称为：**非自适应 oracle 标注数据上的离线学习**，或“被动学习器 + 预先固定的 oracle 标注数据采集”。它不是无需 membership oracle 的纯日志被动学习，也不是以候选模型和反例驱动的主动学习。

建议把数据的两种成本分别记录：自然 rollout 的执行步数/重置次数，以及候选轨迹 validity 查询数与总候选长度。一次整轨迹查询返回一比特，不应把验证器内部扫描了几条边视为学习者收到的边级标签。记录 oracle 的计算工作量可以，但不得将错误位置、正确后继、搜索停止位置或内部缓存暴露到训练输入。

候选采样分布可以依赖冻结的自然观测集合（例如统一扰动），但不依赖模型或正确性筛选；所有候选、重复项、正/负标签均保存。所有通过验证的候选应加入共同正例集，保证 Good-only 与 Good+Bad 的差异只在负约束；同时单独报告自然 G 和 oracle 增补 G 的成本与覆盖变化。不能把额外正例的收益记为负标签收益。

纯“修改正例后继”在此假设类下按定理 1 完全冗余。为了允许非冗余，固定通用采样可包括改变动作、随机轨迹、拼接片段等；这些机制未必有效，但可以在不看模型表现的情况下评估。不得保证生成含未知转移，不得只保留 informative negatives。信息分组只能在生成后作诊断。

## 7. 可识别性与行为评价的边界

版本空间是单点时，数据在指定假设类内唯一识别整个转移函数。某个 pair 的可行域为单点时，仅该 pair 被识别。候选域大于 1 时，无论采用神经还是符号方法，都需归纳偏置选择一个后继；不存在只凭这些样本就对所有一致真模型保证正确的算法。

若一个未知 pair 不出现在任何非平凡约束中，则在无结构先验的函数类中其后继仍可任取 S；仅增加与它无关的样本不能识别它。结构化神经模型可通过参数共享泛化，但应与数据逻辑蕴含分开分析。

更高 unseen edge accuracy 不蕴含更高初始双模拟成功率：一个对关键可达分支的错误就可能破坏等价，而多个不可达边错误对初始行为无影响。全状态 identity 检查与初始状态检查必须分开。动作被忽略的关系可能容忍动作语义错误；AP 标签过粗还可能让不同状态行为等价。固定有限 CTL/LTL 测试集上的 agreement 不是全部性质等价的证明。

统计应以独立生成的基础地图/结构为聚类单位；旋转、嵌套预算、相同数据上的模型与种子对比都依赖。小型先导实验优先报告逐例配对差值、基础地图聚合和范围，不以数千条边作为独立样本计算虚假精确的显著性。信息量与性能的关系首先是描述性关联，不能由相关性直接推断因果。

## 8. 可用于英文论文的理论总结

### Information content of negative execution traces

We consider deterministic finite-state systems whose state and action identities are known. A positive execution trace fixes the successor of every state–action pair occurring in that trace. A negative execution trace imposes a disjunction: at least one of its proposed transitions must be absent. It does not provide an invalidity label for each individual transition.

This distinction yields a simple redundancy result. Suppose that a negative trace contains a transition (s,a,u), while a positive trace fixes (s,a,t) with u≠t. Every deterministic model consistent with the positive data must already reject the negative trace. Consequently, negatives obtained by replacing a non-initial state of an included positive trace do not reduce the version space: H(G,B)=H(G). The conclusion relies on exact state identities, determinism, and hard consistency with all positive traces. It does not extend to arbitrary negative traces, nondeterministic candidates, or partially observed state identities. Redundant negatives may nevertheless affect a neural learner through optimization, because the learner need not satisfy the positive constraints exactly.

For the unrestricted class of total deterministic transition functions, a negative trace adds a non-redundant constraint precisely when its acceptance requirements neither contradict the positive observations nor contradict themselves through repeated state–action pairs with different successors. Such a trace necessarily involves at least one previously unobserved pair. If it constrains k distinct unobserved pairs over n possible successor states, it excludes a fraction n^(−k) of the positive-consistent models. Thus, non-redundancy alone need not imply substantial information: long candidate traces can remove an extremely small portion of the version space. Moreover, several negative traces can constrain combinations of transitions without reducing any individual successor domain.

To learn from trace-level labels, we use a factorized distribution over unknown deterministic transition functions. The acceptance probability of a trace is the product of the successor probabilities for its distinct, mutually consistent state–action requirements. A negative trace contributes −log(1−P(acceptance)). Repeated requirements are counted once; inconsistent repeated requirements have acceptance probability zero. Summing these losses is a composite objective rather than the exact joint likelihood of all constraints. This formulation preserves trace-level semantics while exposing potential optimization difficulties, including weak gradients for long, low-probability candidates.

The empirical question is therefore separated into three stages: whether negative data reduce the specified version space, whether a learning algorithm exploits that reduction to improve unobserved transitions, and whether these improvements preserve formally evaluated behavior. None of these implications is automatic. In particular, sample consistency, transition accuracy, initial-state bisimulation, and finite-suite temporal-property agreement are distinct outcomes. A fixed, model-independent query protocol supports an unbiased comparison of learners, but its oracle-labeling cost must be disclosed separately from learning on unqueried execution logs.

## 9. 文献定位与可核实来源

1. Mitchell, T. M. (1977). *Version Spaces: A Candidate Elimination Approach to Rule Learning*. IJCAI-77, 305–309. https://ijcai.org/proceedings/1977/68 ; original PDF https://www.ijcai.org/Proceedings/77-1/Papers/048.pdf . 依据：以全部样本一致假设集合表达归纳不确定性；本稿把这一视角应用到已知状态的转移函数，不声称版本空间概念为新贡献。
2. Angluin, D. (1987). *Learning Regular Sets from Queries and Counterexamples*. Information and Computation 75(2), 87–106. https://doi.org/10.1016/0890-5401(87)90052-6 ; accessible paper https://homepages.math.uic.edu/~lreyzin/papers/angluin87.pdf . 依据：membership/equivalence queries 的经典主动学习设定；这里没有等价查询，也没有 L* 的辨识保证。
3. Angluin, D. (1988). *Queries and Concept Learning*. Machine Learning 2, 319–342. https://doi.org/10.1007/BF00116828 . 出版商记录日期为 April 1988，部分索引记 1987；本文使用出版商年份。依据：区分查询类型及随机采样学习模型。
4. Yogananda Jeppu, N., Melham, T., and Kroening, D. (2023). *Enhancing active model learning with equivalence checking using simulation relations*. Formal Methods in System Design 61, 164–197. https://doi.org/10.1007/s10703-023-00433-y ; author-hosted full text https://www.cs.ox.ac.uk/tom.melham/pub/Jeppu-2023-EAM.pdf . 依据：该工作利用系统实现上的检查与反例迭代建立充分的完整性保证；本研究的有限、冻结数据协议并不享有该保证。作者定义的观测标签 trace 与本稿显式状态 trace 也不可混用。
5. Xu, J., Zhang, Z., Friedman, T., Liang, Y., and Van den Broeck, G. (2018). *A Semantic Loss Function for Deep Learning with Symbolic Knowledge*. ICML, PMLR 80, 5502–5511. https://proceedings.mlr.press/v80/xu18h.html ; https://proceedings.mlr.press/v80/xu18h/xu18h.pdf . 依据：以逻辑约束的满足概率构造可微损失的语义视角；本稿单负轨迹闭式公式是相应确定性函数分布下的直接推导，尚不证明任何网络的性能收益。

文献均经 2026-11-17 网页/原始论文检索核实。这里的具体冗余定理、单负例计数与反例为自包含推导；引用提供背景，不替代这些证明。是否构成新的可发表理论贡献仍需更广泛文献审查。

## 10. 理论小系统穷尽验证记录

2026-11-17 在 JavaScript V8 工具运行时实际执行独立枚举：2 状态、1 动作、全部 4 个总函数、全部 9 个已知部分函数及长度 1–4 的所有 60 条显式状态轨迹。540 个冗余判别与精确计数检查、60 个带权模型枚举与去重概率比较、80 个配对后继替换检查全部通过，0 失败。该验证覆盖有限特例，不替代定理证明，也不属于神经网络训练结果。

可复现脚本：`node reports/passive_negative_traces_20261117/theory_exact_check.js`；原始摘要见 `theory_exact_results.json`。脚本不需要额外依赖，不读取真实实验地图或模型预测。
