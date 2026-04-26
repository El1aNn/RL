# Dataset Plan

## Target Model

- 模型：`DeepSeek-R1-Distill-Qwen-1.5B`
- 定位：小模型、可训得动，但很容易被单一模式数据带偏
- 结论：优先追求高质量、强约束、多模式覆盖，而不是盲目堆超大规模
- 训练主线：SFT 只负责把协议和基本谈判行为教稳，主要策略提升留给 RL

## SFT Recommendation

### Recommended Scale

- 烟雾测试版：`1,500 ~ 2,000` 条完整对话
  - 用于检查格式稳定性、结局分布、模式覆盖
  - 预期得到约 `9,000 ~ 16,000` 条 turn-level SFT 样本
- 课程项目主力版：`4,000 ~ 6,000` 条完整对话
  - 推荐作为 `1.5B` 模型的主力 SFT 数据规模
  - 预期得到约 `25,000 ~ 50,000` 条 turn-level SFT 样本
- 谨慎上限版：`8,000` 条完整对话
  - 只有在验证集和 hard-case 指标还明显提升时才扩到这一档
  - 超过这一档通常不如把算力和时间投入到 RL

### Why Keep SFT Smaller

- 任务 domain 很窄，结构化协议强，SFT 的收益会较早饱和
- 当前训练样本是 turn-level 展开，单条完整对话会扩成多条样本，实际 token 量并不小
- 如果 SFT 过多，模型容易把模板谈判学得太死，反而压缩 RL 的优化空间
- 课程项目里，后续自博弈带来的策略提升通常比继续堆同分布 SFT 更值

### Scenario Mix

- 谈判空间分层：`4` 种
  - `near_zero_space`
  - `narrow_space`
  - `balanced_space`
  - `wide_space`
- 推荐占比：
  - `near_zero_space`: `18%`
  - `narrow_space`: `27%`
  - `balanced_space`: `35%`
  - `wide_space`: `20%`

### Dialogue Modes

- 对话模式：`8` 种
  - `early_deal_anchor`
  - `patient_balanced_deal`
  - `hard_bargain_late_deal`
  - `near_limit_deal`
  - `buyer_walkaway_with_zone`
  - `seller_walkaway_protect_margin`
  - `timeout_deadlock`
  - `bluff_and_reversal`

### Outcome Mix

- 推荐目标分布：
  - `deal`: `54% ~ 62%`
  - `walkaway`: `20% ~ 26%`
  - `timeout`: `14% ~ 20%`
- 原则：
  - 不能把“存在 bargaining zone”直接学成“必成交”
  - `walkaway` 必须包含“明明有空间但还是谈崩”的样本
  - `timeout` 要保留一部分，让模型学会拖延与僵持的真实形态

### Coverage Rule

- 关键组合至少保证 `120 ~ 200` 条完整对话
- 最重要的 hard case 组合至少保证 `250 ~ 350` 条完整对话
- 优先保证这些组合：
  - `near_zero_space + near_limit_deal`
  - `near_zero_space + timeout_deadlock`
  - `balanced_space + buyer_walkaway_with_zone`
  - `balanced_space + seller_walkaway_protect_margin`
  - `wide_space + early_deal_anchor`
  - `wide_space + bluff_and_reversal`

## Data Quality Rules

- 卖家报价整体单调不升，买家报价整体单调不降
- 不允许出现机械等差让步，例如连续每轮都刚好 `100`
- `deal` 价格必须落在 `[seller_cost, buyer_budget]`
- `walkaway` 和 `timeout` 要与最后一条动作一致
- 不能直接泄露“底线”“最高预算”“最低售价”等私密信息
- 训练样本中的 `messages` 结构必须和未来 rollout 时看到的结构一致

## RL Data Production

### Core Principle

- RL 阶段的主数据不是“标注答案”，而是“场景池 + 在线 self-play rollout”
- SFT 提供的是行为先验
- RL 再通过奖励把模型往更优策略推

### RL Scenario Pool

- 推荐固定场景池：
  - 训练集：`20,000 ~ 30,000` 个场景
  - 验证集：`2,000 ~ 3,000` 个场景
  - 测试集：`2,000 ~ 3,000` 个场景
- 场景池只包含环境状态，不包含标准答案对话
- 和 SFT 一样保留 `4` 种谈判空间分层，但 RL 中应略微提高 hard case 比例

### RL Curriculum

- Phase 1：稳定协议与基本成交
  - 只用 `balanced_space` 和 `wide_space`
  - 让模型先学会格式稳定、基本收敛、少犯非法成交错误
- Phase 2：加入狭窄空间与放弃样本
  - 引入 `narrow_space`
  - 强化“有空间但不一定成交”的认识
- Phase 3：加入 hardest cases
  - 引入 `near_zero_space`
  - 提高 `walkaway` 和 `timeout` 比例
  - 重点优化僵持、误判、极限成交等难例

### RL Rollout Mix

- 每轮训练推荐混合三类对手：
  - `60%` 当前策略 self-play
  - `25%` 冻结的 SFT / 旧 checkpoint 对手
  - `15%` 规则型启发式对手
- 启发式对手建议至少有这些类型：
  - 高开慢降卖家
  - 低开慢加买家
  - 固执型卖家
  - 易妥协型买家
  - 临界点放弃型买家

### RL Batch Suggestion

- 每次 rollout 采样 `256 ~ 512` 个场景
- 每个场景生成 `4 ~ 8` 条轨迹
- 每轮得到约 `1,000 ~ 4,000` 条在线对话
- 保留一个 hard-case replay buffer
  - 存放格式错误、非法成交、超时僵局、过早放弃、极限成交等轨迹
  - 后续每轮混入 `10% ~ 20%` 的 hard-case 场景做再训练

## Suggested Execution Order

1. 先生成 `1,500 ~ 2,000` 条 SFT 对话做 smoke test
2. 检查模式分布、结局分布、非法样本率
3. 扩到 `4,000 ~ 6,000` 条完整对话做正式 SFT
4. 用固定 `2,000 ~ 3,000` 个验证场景评估 SFT checkpoint
5. 只要格式和基本策略已稳定，就尽早进入 RL curriculum，先 easy，再 mixed，最后 hard

## Recommended Budget Tiers

- 低预算版
  - SFT：`2,000` 条完整对话
  - RL 训练场景池：`20,000`
  - 适合先验证方法能否跑通
- 课程项目标准版
  - SFT：`4,000 ~ 5,000` 条完整对话
  - RL 训练场景池：`25,000`
  - 推荐作为主线方案
- 冲效果版
  - SFT：`6,000 ~ 8,000` 条完整对话
  - RL 训练场景池：`30,000`
  - 只在已经确认 SFT 仍有收益时采用
