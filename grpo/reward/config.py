"""
Reward 配置

从 configs/default.yaml 的 `reward:` 字段加载；也可以直接手动构造。
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class RewardConfig:
    # ---- 终止态主奖励 ----
    deal_scale: float = 100.0           # 合法成交的分摊总量（0~deal_scale）
    violation_penalty: float = -100.0   # 违反底线成交
    violation_counterparty_reward: float = -10.0  # 对手违规成交时，另一方也不应因诱导违规获利
    walkaway_wrong: float = -30.0       # 对方报价已进入自己接受范围却放弃
    walkaway_right: float = 5.0         # 继续谈会违反底线，放弃合理
    timeout: float = -15.0              # 超时未成交
    format_error: float = -50.0         # 格式违规超预算

    # ---- Shaping 开关 ----
    enable_format_bonus: bool = True
    enable_monotone: bool = True
    enable_round_cost: bool = True
    enable_leak_penalty: bool = True
    enable_extreme_offer_penalty: bool = True
    enable_buyer_budget_pressure_penalty: bool = False
    enable_shared_balance_reward: bool = False
    enable_deal_balance_penalty: bool = False
    enable_early_deal_penalty: bool = False

    # ---- Shaping 数值 ----
    format_bonus: float = 1.0           # 每轮格式合规 +1
    monotone_bonus: float = 2.0         # 报价朝有利方向变化
    monotone_penalty: float = -3.0      # 报价朝不利方向变化
    round_cost: float = -0.3            # 每轮成本（总和 = -0.3 * rounds）
    leak_penalty: float = -20.0         # 每次字面泄密
    extreme_offer_penalty: float = -8.0 # 明显脱离市场价的报价
    buyer_min_market_ratio: float = 0.35
    seller_max_market_ratio: float = 1.65
    buyer_near_budget_offer_ratio: float = 0.98
    buyer_near_budget_offer_penalty: float = -8.0
    buyer_first_offer_budget_ratio: float = 0.95
    buyer_first_offer_budget_penalty: float = -15.0
    buyer_near_budget_deal_ratio: float = 0.98
    buyer_near_budget_deal_penalty: float = -20.0
    shared_balance_alpha: float = 0.35
    shared_balance_scale: float = 1.0
    shared_balance_eps: float = 1.0e-3
    deal_balance_gap_threshold: float = 0.25
    deal_balance_gap_penalty: float = -40.0
    early_deal_min_rounds: int = 2
    early_deal_penalty: float = -4.0

    # ---- 零谈判空间的防方差爆炸 ----
    zone_floor: float = 1.0             # 分母最小值

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})
