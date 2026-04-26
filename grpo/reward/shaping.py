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
    re.compile(r"我的?\s*底价"),
    re.compile(r"我的?\s*底线"),
    re.compile(r"成本\s*(就是|是|为|价)\s*\d+"),   # 直接说成本数字
]

LEAK_PATTERNS_BUYER = [
    re.compile(r"我的?\s*最高\s*(预算|出价|价)"),
    re.compile(r"我的?\s*预算\s*(上限|最高|顶)"),
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
    }
