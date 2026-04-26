# Negotiation SFT and GRPO Training Plan

## 目标

本项目训练一个二手商品谈判模型。模型通过不同 system prompt 扮演 buyer 或 seller：

- buyer 目标：在不超过最高预算的前提下尽量低价成交。
- seller 目标：在不低于最低售价的前提下尽量高价成交。
- 双方都不能泄露自己的私密底线。
- 对话必须遵守 `[报价：XXX]`、`<deal>价格</deal>`、`<walkaway>` 格式。

整体路线：

1. 用 SFT 学会基本格式、角色扮演、谈判语言和成交/放弃行为。
2. 用 GRPO 让模型在自博弈中优化谈判收益。
3. buyer 和 seller 的目标冲突，所以训练时需要分角色、分 reward、分 adapter 控制更新。

## 数据分层

场景数据由 `data/generate_scenarios.py` 生成。这个脚本是纯规则采样器，不调用 API。

它读取：

- `configs/scenario_templates.yaml`：商品模板、描述、成本区间、市场价区间。
- `configs/generation_profiles.yaml` 或 `configs/rl_generation_profiles.yaml`：谈判空间难度分布。

每条 scenario 包含：

```json
{
  "scenario_id": "train_00000",
  "item_name": "二手实木书桌",
  "item_description": "白橡木，120x60cm，使用半年，保养良好",
  "buyer_budget": 1331,
  "seller_cost": 1141,
  "market_ref_price": 2336,
  "max_rounds": 10,
  "metadata": {
    "zone_profile": "narrow_space",
    "zone_profile_name": "狭窄谈判空间",
    "gap_ratio": 0.1665
  }
}
```

难度来自 bargaining zone：

```text
gap_ratio = (buyer_budget - seller_cost) / seller_cost
```

默认 RL scenario pool 使用更偏难的比例：

| 难度 | 含义 | 目标权重 |
|---|---|---:|
| `near_zero_space` | 近乎无谈判空间 | 25% |
| `narrow_space` | 狭窄谈判空间 | 35% |
| `balanced_space` | 中等谈判空间 | 25% |
| `wide_space` | 宽松谈判空间 | 15% |

默认使用的 RL 场景路径：

```text
data/scenarios_rl_5k/train.jsonl  # 4000
data/scenarios_rl_5k/val.jsonl    # 500
data/scenarios_rl_5k/test.jsonl   # 500
```

生成命令如下。这个阶段只做规则采样，不调用 API：

```bash
python3 data/generate_scenarios.py \
  --profile-config configs/rl_generation_profiles.yaml \
  --output-dir data/scenarios_rl_5k \
  --num-train 4000 \
  --num-val 500 \
  --num-test 500 \
  --seed 20260419
```

实际生成后的分布：

| Split | Total | near_zero | narrow | balanced | wide |
|---|---:|---:|---:|---:|---:|
| train | 4000 | 1008 | 1426 | 980 | 586 |
| val | 500 | 114 | 163 | 139 | 84 |
| test | 500 | 118 | 193 | 124 | 65 |

`data/scenarios_rl_1k/` 可以保留为快速 smoke test 数据，但正式 GRPO 默认使用 `data/scenarios_rl_5k/`。

## SFT 阶段

SFT 的目的不是追求最优谈判收益，而是让模型先学会：

- buyer/seller 两种身份。
- 议价格式。
- 多轮对话节奏。
- 成交、放弃、超时三类结局。
- 不泄露私密底线。
- 报价随轮次合理收敛。

SFT 数据生产流程：

```text
scenario_templates.yaml
  -> generate_scenarios.py
  -> scenarios_sft_1k/*.jsonl
  -> generate_sft_data.py 调用 API 生成完整谈判对话
  -> PromptBuilder 按每个 turn 拆成 SFT messages
  -> data/sft_1k/*.jsonl
```

如果使用 thinking 版本：

```text
data/sft_1k/*.jsonl
  -> add_thinking_to_sft.py
  -> data/sft_1k_think_content/*.jsonl
```

给 LLaMA-Factory 训练时，推荐使用 `content_prefix` 形式：

```text
<think>策略思路...</think>
【第X轮-买家/卖家】...
```

不要依赖单独的 `thinking` 字段，因为大多数 ShareGPT/OpenAI 格式训练只读取 `messages[].content`。

## GRPO 阶段

GRPO 不应该直接消费 SFT 那种固定 dialogue。更合理的是：

```text
给定 scenario
  -> 当前 policy 在线 rollout
  -> 对每个 prompt/state 采样 K 个候选动作或 K 条完整 rollout
  -> 根据最终成交/流局计算 reward
  -> 在同组内做 reward normalization / advantage
  -> 只更新当前 active role 的 adapter
```

scenario 是 RL 的输入数据；rollout 是训练时动态生成的，不提前写死。

## 模型与 Adapter

从 SFT checkpoint 初始化两个 adapter：

```text
base_model + sft_adapter
  -> buyer_adapter
  -> seller_adapter
```

运行时：

```text
buyer policy  = base_model + buyer_adapter
seller policy = base_model + seller_adapter
```

buyer 和 seller 可以是同一个 base model，但 adapter 分开更新。

## Reward 设计

成交时：

```text
seller_reward = (deal_price - seller_cost) / (buyer_budget - seller_cost)
buyer_reward  = (buyer_budget - deal_price) / (buyer_budget - seller_cost)
```

如果成交价越接近 seller_cost，buyer 越好；越接近 buyer_budget，seller 越好。

非法成交：

```text
deal_price < seller_cost  -> seller 强负分
deal_price > buyer_budget -> buyer 强负分
```

放弃或超时：

- 如果对方报价已经进入自己可接受范围却放弃，给负分。
- 如果继续谈会违反自己底线，放弃可以给小正分或轻微负分。
- 超时通常给小负分，鼓励模型有效收敛。

通用惩罚：

- 格式错误。
- 没有报价。
- 过早输出 `<deal>` 或 `<walkaway>`。
- 泄露“最高预算”“最低售价”“底线”等私密信息。
- seller 报价不降反升。
- buyer 报价不升反降。

## 三阶段 GRPO

### 阶段 1：训练 buyer，seller 冻结

目标：先让 buyer 学会压价、守预算、判断何时成交或退出。

流程：

```text
for scenario in data/scenarios_rl_5k/train.jsonl:
    seller_fixed 先开价
    buyer_trainable 生成回复
    seller_fixed 继续回复
    buyer_trainable 继续回复
    ...
    根据最终 outcome 计算 buyer_reward
    GRPO 只更新 buyer_adapter
```

此阶段 seller 不参与 loss，不更新参数，只作为 frozen opponent。

### 阶段 2：训练 seller，buyer 冻结

目标：让 seller 学会高价锚定、逐步让步、保护最低售价。

流程：

```text
for scenario in data/scenarios_rl_5k/train.jsonl:
    seller_trainable 先开价
    buyer_fixed 回复
    seller_trainable 继续回复
    buyer_fixed 继续回复
    ...
    根据最终 outcome 计算 seller_reward
    GRPO 只更新 seller_adapter
```

此阶段 buyer 不参与 loss，不更新参数，只作为 frozen opponent。

### 阶段 3：混合 self-play，少量更新双方

目标：让 buyer 和 seller 适应彼此，但避免双方同时剧烈漂移。

推荐交替更新：

```text
batch 1: 更新 buyer_adapter，冻结 seller_adapter
batch 2: 更新 seller_adapter，冻结 buyer_adapter
batch 3: 更新 buyer_adapter，冻结 seller_adapter
batch 4: 更新 seller_adapter，冻结 buyer_adapter
```

学习率应低于前两个阶段，例如：

```text
stage1 buyer lr:  5e-6
stage2 seller lr: 5e-6
stage3 mixed lr:  1e-6
```

阶段 3 必须保留 KL 约束，避免策略语言和格式退化。

## 同一条 Trajectory 的两个视角

一条在线 self-play trajectory 可以拆成两个视角，但训练时不要把双方目标混在一起。

原始 trajectory：

```text
S1: [报价：4500] ...
B1: [报价：3800] ...
S2: [报价：4200] ...
B2: <deal>4200</deal>
```

seller 视角：

```text
history: 空
completion: S1
reward: seller_reward

history: S1, B1
completion: S2
reward: seller_reward
```

buyer 视角：

```text
history: S1
completion: B1
reward: buyer_reward

history: S1, B1, S2
completion: B2
reward: buyer_reward
```

关键规则：

- history 可以包含双方发言。
- completion 只包含当前 active role 的发言。
- loss 只打在当前 active role 的 token 上。
- reward 使用当前 active role 的收益。
- 更新 buyer 时冻结 seller；更新 seller 时冻结 buyer。

## 推荐执行顺序

1. 先用 `data/sft_1k_think_content/train_800.jsonl` 做 SFT。
2. 保存 SFT checkpoint，并拆出 `buyer_adapter` 与 `seller_adapter`。
3. 用 `data/scenarios_rl_5k/train.jsonl` 做阶段 1 buyer GRPO，seller 冻结。
4. 固定 buyer，用同一批 5k train scenario 做阶段 2 seller GRPO。
5. 用更小学习率做阶段 3 alternating self-play。
6. 用 `data/scenarios_rl_5k/val.jsonl` 做 reward、格式、成交率、泄密率验证。
7. 最后只在 `data/scenarios_rl_5k/test.jsonl` 上做一次报告。

## 当前结论

SFT 数据负责教模型“怎么像一个谈判者说话”。

RL scenario 数据负责给 GRPO 提供不同难度的谈判环境。

真正的 GRPO 数据不是提前固定好的对话，而是在训练时由 buyer/seller policy 根据 scenario 在线 rollout 产生。
