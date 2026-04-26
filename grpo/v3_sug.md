# V3 Reward 优化方案

## 一、V1 / V2 问题回顾

### V1 问题
- **成交太快**：平均 1.6~2.1 轮即成交
- **严重偏向 buyer**：buyer reward 常 40+，seller 常为负
- 典型案例：price=7700, seller_cost=7014, buyer_budget=9580
  - buyer surplus = 9580 - 7700 = 1880 (73%)
  - seller surplus = 7700 - 7014 = 686 (27%)

### V2 问题
- 成交轮数增加到 3.6~4.6（改善），但 deal_rate 从 100% 降到 87~97%（过度）
- **仍然偏向 buyer**（方向对但量变不够）
- 更高的 KL (0.12) 和更低的 LR (6e-7) 使学习变慢

### V1 vs V2 超参对比

| 参数 | V1 | V2 |
|------|----|----|
| `shared_balance_alpha` | 0.35 | 0.40 |
| `balance_gap_threshold` | 0.25 | 0.22 |
| `balance_gap_penalty` | -40 | -45 |
| `early_deal_penalty` | -4 | -5 |
| `learning_rate` | 8e-7 | 6e-7 |
| `beta_kl` | 0.10 | 0.12 |

---

## 二、根因分析（5 个核心问题）

### 问题 1：Terminal reward 的线性 surplus 结构天然鼓励"占便宜"

```python
buyer_r = 100 * (budget - price) / zone   # price 越低 buyer reward 越高
seller_r = 100 * (price - cost) / zone     # price 越高 seller reward 越高
```

这是零和博弈结构。Nash 项只占 35~40%，剩下 60~65% 仍是零和激励。
buyer 在前几个 stage 已学会压价策略，这个比例不够纠正。

### 问题 2：Seller 缺乏"拒绝坏 deal"的激励

- `walkaway_right = +5`，远不如一个勉强 deal（即使 seller_u 只有 0.1 也有 10 分）
- Seller 理性地接受任何正收益 deal，哪怕收益极低
- 没有 seller 侧"底线保护"（buyer 有 `buyer_budget_pressure_penalty`，seller 无对称机制）

### 问题 3：Deal balance penalty 力度不够且形式欠佳

```python
excess = max(0.0, gap - threshold)
penalty = deal_balance_gap_penalty * excess  # 线性
```

- threshold=0.22 意味着 buyer_u=0.61, seller_u=0.39 时才开始罚（61:39 的分配！）
- 只罚多拿方，不奖少拿方 —— seller 无额外激励去推高价格

### 问题 4：Round cost 太低，早期成交太"便宜"

- `round_cost = -0.3/turn`，谈 2 轮（4 turns）只扣 1.2 分
- `early_deal_penalty = -4~-5`，而 deal reward 通常是 30~50+
- 模型发现快速成交的期望收益远高于多谈几轮

### 问题 5：Shared balance 的 sqrt 函数在不平衡区域梯度太平

- `sqrt(0.7 * 0.3) = 0.458` vs `sqrt(0.5 * 0.5) = 0.5`，差距只有 8%
- 当 alpha=0.4 时，这 8% 只贡献约 3.2 分 reward 差异 —— 信号太弱

---

## 三、V3 改动方案（7 项）

### 改动 1（P0）：大幅提升 Nash 权重

```yaml
shared_balance_alpha: 0.65   # 从 0.40 → 0.65
```

让公平信号成为主导（65%），个人 surplus 只占 35%。
Nash 乘积 `sqrt(bu * su)` 在 50:50 分配时达到最大值，自然引导向中间价。

### 改动 2（P0）：新增"公平成交奖金"（fairness bonus）

**双方共享**的正和激励，deal 接近 midpoint 时 buyer 和 seller 同时获奖。

```python
# reward/shaping.py 新增
def compute_fairness_bonus(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """Reward BOTH sides when the deal lands near the midpoint."""
    if not cfg.enable_fairness_bonus or state.deal_price is None:
        return 0.0
    if getattr(state.outcome, "value", None) != "deal":
        return 0.0

    buyer_u, seller_u = _deal_utilities(state, cfg)
    # fairness = 1 - |bu - su|, 范围 [0, 1], 在 50:50 时为 1
    fairness = 1.0 - abs(buyer_u - seller_u)
    return cfg.fairness_bonus_scale * fairness
```

```yaml
enable_fairness_bonus: true
fairness_bonus_scale: 15.0    # 完美平衡时双方各 +15
```

**关键**：创造正和博弈信号，双方合作达成公平交易比单方面占便宜更好。

### 改动 3（P1）：提高 seller walkaway 收益

```yaml
walkaway_right: 12.0    # 从 5 → 12
walkaway_wrong: -25.0   # 从 -30 → -25（减少误判惩罚）
```

当 seller surplus 只有 zone 的 10%（即 10 分），经 Nash 混合后约 6-7 分，
walkaway 的 12 分就成为更优选择 —— seller 学会拒绝不公平 deal。

### 改动 4（P1）：新增 seller 成本保护惩罚

对称 `buyer_budget_pressure_penalty`，新增 seller 侧保护。

```python
# reward/shaping.py 新增
def compute_seller_cost_pressure_penalty(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """Discourage the seller from accepting deals too close to private cost."""
    if role != "seller" or not cfg.enable_seller_cost_pressure_penalty:
        return 0.0

    seller_cost = float(state.scenario.seller_cost)
    if seller_cost <= 0:
        return 0.0

    total = 0.0
    seller_offers = [
        t.parsed.price
        for t in state.history
        if t.role == "seller"
        and t.parsed.action_type == "offer"
        and t.parsed.price is not None
    ]

    # 惩罚 seller 报价太接近成本
    near_cost_threshold = seller_cost * cfg.seller_near_cost_offer_ratio
    total += cfg.seller_near_cost_offer_penalty * sum(
        1 for price in seller_offers if price <= near_cost_threshold
    )

    # 惩罚成交价太接近成本
    if (state.deal_price is not None
            and float(state.deal_price) <= seller_cost * cfg.seller_near_cost_deal_ratio):
        total += cfg.seller_near_cost_deal_penalty

    return total
```

```yaml
enable_seller_cost_pressure_penalty: true
seller_near_cost_offer_ratio: 1.08    # seller 报价低于 cost*1.08 时罚
seller_near_cost_offer_penalty: -8.0
seller_near_cost_deal_ratio: 1.10     # 成交价低于 cost*1.10 时罚
seller_near_cost_deal_penalty: -15.0
```

### 改动 5（P2）：Deal balance penalty 改二次 + 降阈值

```python
# reward/shaping.py 修改
def compute_deal_balance_penalty(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """Penalize the side that captures too much — quadratic for stronger gradient."""
    if not cfg.enable_deal_balance_penalty or state.deal_price is None:
        return 0.0
    if getattr(state.outcome, "value", None) != "deal":
        return 0.0

    buyer_u, seller_u = _deal_utilities(state, cfg)
    gap = abs(buyer_u - seller_u)
    excess = max(0.0, gap - cfg.deal_balance_gap_threshold)
    if excess <= 0:
        return 0.0

    # 二次惩罚：gap 越大罚得越狠
    penalty = cfg.deal_balance_gap_penalty * (excess ** 2)

    # 多拿方全额惩罚；少拿方也受部分惩罚（鼓励 push back）
    if (buyer_u > seller_u and role == "buyer") or (seller_u > buyer_u and role == "seller"):
        return penalty
    else:
        return penalty * cfg.deal_balance_victim_ratio
```

```yaml
deal_balance_gap_threshold: 0.12     # 从 0.22 → 0.12（56:44 就开始罚）
deal_balance_gap_penalty: -120.0     # 提升（配合二次公式）
deal_balance_victim_ratio: 0.3       # 新增：少拿方也受 30% 惩罚
```

**效果举例**：

| buyer_u | seller_u | gap | excess | buyer 惩罚 | seller 惩罚 |
|---------|----------|-----|--------|-----------|------------|
| 0.56 | 0.44 | 0.12 | 0.00 | 0 | 0 |
| 0.70 | 0.30 | 0.40 | 0.28 | **-9.4** | **-2.8** |
| 0.85 | 0.15 | 0.70 | 0.58 | **-40.4** | **-12.1** |

### 改动 6（P2）：加强早期成交惩罚

```python
# reward/shaping.py 修改
def compute_early_deal_penalty(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """Discourage accepting very early unless the split is already fair."""
    if not cfg.enable_early_deal_penalty or state.deal_price is None:
        return 0.0
    if getattr(state.outcome, "value", None) != "deal":
        return 0.0
    if state.current_round >= int(cfg.early_deal_min_rounds):
        return 0.0

    buyer_u, seller_u = _deal_utilities(state, cfg)
    if abs(buyer_u - seller_u) <= cfg.deal_balance_gap_threshold:
        return 0.0  # 已经公平则不罚

    # 双方都罚（不只是提出 deal 的一方）
    return cfg.early_deal_penalty
```

```yaml
early_deal_min_rounds: 3     # 从 2 → 3
early_deal_penalty: -10.0    # 从 -5 → -10
```

### 改动 7（P3）：开启 buyer budget pressure

```yaml
enable_buyer_budget_pressure_penalty: true   # 完善对称性
```

---

## 四、V3 完整配置汇总

### reward 段
```yaml
reward:
  deal_scale: 100.0
  violation_penalty: -100.0
  violation_counterparty_reward: -10.0
  walkaway_wrong: -25.0                        # 放宽 (-30 → -25)
  walkaway_right: 12.0                         # 大幅提升 (5 → 12)
  timeout: -15.0
  format_error: -50.0

  # --- Shaping 开关 ---
  enable_format_bonus: true
  enable_monotone: true
  enable_round_cost: true
  enable_leak_penalty: true
  enable_extreme_offer_penalty: true
  enable_buyer_budget_pressure_penalty: true   # 新开启
  enable_seller_cost_pressure_penalty: true    # 新增
  enable_shared_balance_reward: true
  enable_deal_balance_penalty: true
  enable_early_deal_penalty: true
  enable_fairness_bonus: true                  # 新增

  # --- Shared balance (Nash) ---
  shared_balance_alpha: 0.65                   # 核心 (0.40 → 0.65)
  shared_balance_scale: 1.0
  shared_balance_eps: 1.0e-3

  # --- Deal balance penalty (改二次) ---
  deal_balance_gap_threshold: 0.12             # 收紧 (0.22 → 0.12)
  deal_balance_gap_penalty: -120.0             # 配合二次公式
  deal_balance_victim_ratio: 0.3               # 新增

  # --- Fairness bonus (新增) ---
  fairness_bonus_scale: 15.0

  # --- Early deal ---
  early_deal_min_rounds: 3                     # 提升 (2 → 3)
  early_deal_penalty: -10.0                    # 提升 (-5 → -10)

  # --- Seller cost protection (新增) ---
  seller_near_cost_offer_ratio: 1.08
  seller_near_cost_offer_penalty: -8.0
  seller_near_cost_deal_ratio: 1.10
  seller_near_cost_deal_penalty: -15.0

  # --- 其他不变 ---
  format_bonus: 1.0
  monotone_bonus: 2.0
  monotone_penalty: -3.0
  round_cost: -0.3
  leak_penalty: -20.0
  extreme_offer_penalty: -8.0
  buyer_min_market_ratio: 0.35
  seller_max_market_ratio: 1.65
  zone_floor: 1.0
```

### stage3 训练超参
```yaml
stage3:
  learning_rate: 7.0e-7      # 比 V2 稍高，新 reward 信号更强
  beta_kl: 0.10              # 比 V2 稍低，给模型更多探索空间
  total_steps: 150           # 多训一些
```

---

## 五、改动优先级

| 优先级 | 改动 | 涉及文件 | 预期效果 |
|--------|------|----------|----------|
| **P0** | `shared_balance_alpha` 0.40→0.65 | config.py, 启动脚本 | 公平成为主信号 |
| **P0** | 新增 fairness_bonus | shaping.py, config.py | 正和激励，双方追求公平 |
| **P1** | `walkaway_right` 5→12 | config.py, 启动脚本 | seller 有底气拒绝坏 deal |
| **P1** | 新增 seller_cost_pressure_penalty | shaping.py, config.py | 对称保护，seller 不再贱卖 |
| **P2** | deal_balance_penalty 改二次+降阈值 | shaping.py, config.py | 极端不平衡惩罚指数增长 |
| **P2** | early_deal_penalty 加强+双方罚 | shaping.py, config.py | 抑制快速不公平成交 |
| **P3** | 开启 buyer_budget_pressure | 启动脚本 | 完善对称性 |

**时间紧迫时至少做 P0 + P1（4 项改动）**，应能显著改善价格平衡。

---

## 六、需要修改的文件清单

1. `grpo/reward/config.py` — 新增字段
2. `grpo/reward/shaping.py` — 新增 2 个函数 + 修改 2 个函数 + 注册到 compute_shaping_for_role
3. `grpo/configs/default.yaml` — 新增配置项默认值
4. `grpo/scripts/start_stage3_balanced_v3_from_s12_best_100.sh` — 新建 V3 启动脚本
