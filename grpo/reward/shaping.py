"""
Reward shaping：格式 / 单调性 / 轮数 / 泄密

所有 shaping 项以 "per-role delta" 的形式返回，由 reward_fn 最后相加。
"""
import re
from typing import Dict, Tuple

from Final_project.grpo.env.negotiation_env import EnvState, Turn
from Final_project.grpo.reward.config import RewardConfig


# 泄密关键词（字面命中才罚，战术性「只有 XXX 块」不罚）
LEAK_PATTERNS_SELLER = [
    re.compile(r"我的?\s*最低售价"),
    re.compile(r"我的?\s*最低\s*(成本|成交|价)"),
    re.compile(r"(最低价|最低售价|底价)\s*(了|啦|是|为|就是)?\s*\d*"),
    re.compile(r"\d+\s*(已经|就|是)?\s*(最低价|最低售价|底价)"),
    re.compile(r"(不能|没法|无法)\s*再\s*(低|降|让)"),
    re.compile(r"再\s*(低|降|让)\s*(就)?\s*(亏|赔)"),
    re.compile(r"(最后|最终)\s*的?\s*(让步|价格|报价)"),
    re.compile(r"我的?\s*底价"),
    re.compile(r"我的?\s*底线"),
    re.compile(r"成本\s*(就是|是|为|价)\s*\d+"),   # 直接说成本数字
]

LEAK_PATTERNS_BUYER = [
    re.compile(r"我的?\s*最高\s*(预算|出价|价)"),
    re.compile(r"我的?\s*预算\s*(上限|最高|顶)"),
    re.compile(r"(最高预算|预算上限|最高出价|最多出)\s*(是|为|就)?\s*\d*"),
    re.compile(r"\d+\s*(是|就是|已经是)?\s*(最高预算|预算上限|最高出价)"),
    re.compile(r"预算\s*(真的|确实|实在)?\s*(有限|就这么多|只有这么多)"),
    re.compile(r"(不能|没法|无法)\s*再\s*(加|涨|出了?)"),
    re.compile(r"(最多|顶多|只能)\s*(出|给)\s*\d+"),
    re.compile(r"我的?\s*底线"),
    re.compile(r"我的?\s*底价"),
]


def compute_format_bonus(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """该角色所有发言中 is_format_valid 的轮数 × format_bonus"""
    if not cfg.enable_format_bonus:
        return 0.0
    return cfg.format_bonus * sum(
        1 for t in state.history if t.role == role and t.parsed.is_format_valid
    )


def compute_monotone(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """
    单调性奖励：
    - seller 的 offer 应单调不增（让步）
    - buyer  的 offer 应单调不减（加价）
    每次符合 +monotone_bonus；反向 +monotone_penalty（负值）。
    """
    if not cfg.enable_monotone:
        return 0.0

    offers = [
        t.parsed.price
        for t in state.history
        if t.role == role and t.parsed.action_type == "offer" and t.parsed.price is not None
    ]
    if len(offers) < 2:
        return 0.0

    total = 0.0
    for prev, cur in zip(offers, offers[1:]):
        if role == "seller":
            # 期望 cur <= prev
            if cur <= prev:
                total += cfg.monotone_bonus
            else:
                total += cfg.monotone_penalty
        else:
            # buyer 期望 cur >= prev
            if cur >= prev:
                total += cfg.monotone_bonus
            else:
                total += cfg.monotone_penalty
    return total


def compute_round_cost(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """每个完整轮数 -round_cost（双方共担），这里按 role 的 turn 数分摊"""
    if not cfg.enable_round_cost:
        return 0.0
    n_turns = sum(1 for t in state.history if t.role == role)
    return cfg.round_cost * n_turns


def compute_leak_penalty(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """字面命中泄密关键词 × leak_penalty"""
    if not cfg.enable_leak_penalty:
        return 0.0

    patterns = LEAK_PATTERNS_SELLER if role == "seller" else LEAK_PATTERNS_BUYER
    count = 0
    for t in state.history:
        if t.role != role:
            continue
        for pat in patterns:
            if pat.search(t.utterance or ""):
                count += 1
                break   # 一条发言最多算一次
    return cfg.leak_penalty * count


def compute_extreme_offer_penalty(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """Penalize obviously unrealistic anchors that tend to exploit weak opponents."""
    if not cfg.enable_extreme_offer_penalty:
        return 0.0

    market_ref = float(getattr(state.scenario, "market_ref_price", 0.0) or 0.0)
    if market_ref <= 0:
        return 0.0

    if role == "buyer":
        threshold = market_ref * cfg.buyer_min_market_ratio
        count = sum(
            1
            for t in state.history
            if t.role == role
            and t.parsed.action_type == "offer"
            and t.parsed.price is not None
            and t.parsed.price < threshold
        )
    else:
        threshold = market_ref * cfg.seller_max_market_ratio
        count = sum(
            1
            for t in state.history
            if t.role == role
            and t.parsed.action_type == "offer"
            and t.parsed.price is not None
            and t.parsed.price > threshold
        )

    return cfg.extreme_offer_penalty * count


def compute_buyer_budget_pressure_penalty(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """Discourage the buyer from revealing or settling at its full private budget."""
    if role != "buyer" or not cfg.enable_buyer_budget_pressure_penalty:
        return 0.0

    buyer_budget = float(state.scenario.buyer_budget)
    if buyer_budget <= 0:
        return 0.0

    total = 0.0
    buyer_offers = [
        t.parsed.price
        for t in state.history
        if t.role == "buyer"
        and t.parsed.action_type == "offer"
        and t.parsed.price is not None
    ]

    near_offer_threshold = buyer_budget * cfg.buyer_near_budget_offer_ratio
    total += cfg.buyer_near_budget_offer_penalty * sum(
        1 for price in buyer_offers if price >= near_offer_threshold
    )

    if buyer_offers and buyer_offers[0] >= buyer_budget * cfg.buyer_first_offer_budget_ratio:
        total += cfg.buyer_first_offer_budget_penalty

    if state.deal_price is not None and float(state.deal_price) >= buyer_budget * cfg.buyer_near_budget_deal_ratio:
        total += cfg.buyer_near_budget_deal_penalty

    return total


def _deal_utilities(state: EnvState, cfg: RewardConfig) -> Tuple[float, float]:
    if state.deal_price is None:
        return 0.0, 0.0
    buyer_budget = float(state.scenario.buyer_budget)
    seller_cost = float(state.scenario.seller_cost)
    zone = max(buyer_budget - seller_cost, cfg.zone_floor)
    price = float(state.deal_price)
    buyer_u = max(0.0, min(1.0, (buyer_budget - price) / zone))
    seller_u = max(0.0, min(1.0, (price - seller_cost) / zone))
    return buyer_u, seller_u


def compute_deal_balance_penalty(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """Penalize the side that captures too much of the bargaining zone."""
    if not cfg.enable_deal_balance_penalty or state.deal_price is None:
        return 0.0
    if getattr(state.outcome, "value", None) != "deal":
        return 0.0

    buyer_u, seller_u = _deal_utilities(state, cfg)
    gap = abs(buyer_u - seller_u)
    excess = max(0.0, gap - cfg.deal_balance_gap_threshold)
    if excess <= 0:
        return 0.0
    if buyer_u > seller_u and role == "buyer":
        return cfg.deal_balance_gap_penalty * excess
    if seller_u > buyer_u and role == "seller":
        return cfg.deal_balance_gap_penalty * excess
    return 0.0


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
        return 0.0

    last = state.history[-1] if state.history else None
    if last is not None and last.role == role and last.parsed.action_type == "deal":
        return cfg.early_deal_penalty
    return 0.0


def compute_low_utility_deal_penalty(state: EnvState, role: str, cfg: RewardConfig) -> float:
    """Penalize accepting a deal that leaves this role too little surplus."""
    if not cfg.enable_low_utility_deal_penalty or state.deal_price is None:
        return 0.0
    if getattr(state.outcome, "value", None) != "deal":
        return 0.0

    last = state.history[-1] if state.history else None
    if last is None or last.role != role or last.parsed.action_type != "deal":
        return 0.0

    buyer_u, seller_u = _deal_utilities(state, cfg)
    if role == "seller":
        threshold = max(float(cfg.seller_min_deal_util), 1e-6)
        if seller_u >= threshold:
            return 0.0
        return cfg.seller_low_util_deal_penalty * (threshold - seller_u) / threshold

    threshold = max(float(cfg.buyer_min_deal_util), 1e-6)
    if buyer_u >= threshold:
        return 0.0
    return cfg.buyer_low_util_deal_penalty * (threshold - buyer_u) / threshold


def compute_shaping_for_role(
    state: EnvState, role: str, cfg: RewardConfig,
) -> Dict[str, float]:
    """
    返回各项 shaping 的明细。由 reward_fn 汇总。
    """
    return {
        "format_bonus": compute_format_bonus(state, role, cfg),
        "monotone": compute_monotone(state, role, cfg),
        "round_cost": compute_round_cost(state, role, cfg),
        "leak_penalty": compute_leak_penalty(state, role, cfg),
        "extreme_offer": compute_extreme_offer_penalty(state, role, cfg),
        "buyer_budget_pressure": compute_buyer_budget_pressure_penalty(state, role, cfg),
        "deal_balance": compute_deal_balance_penalty(state, role, cfg),
        "early_deal": compute_early_deal_penalty(state, role, cfg),
        "low_utility_deal": compute_low_utility_deal_penalty(state, role, cfg),
    }
