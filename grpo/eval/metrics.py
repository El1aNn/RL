"""
谈判评估指标

输入：一组 RolloutTrajectory（来自 rollout_batch 的输出）
输出：dict 形式的指标
"""
from dataclasses import dataclass
from typing import List, Dict, Any
from collections import Counter

from Final_project.grpo.rollout.selfplay import RolloutTrajectory
from Final_project.grpo.env.outcome import Outcome


# ============================================================
# 单指标函数
# ============================================================

def agreement_rate(trajs: List[RolloutTrajectory]) -> float:
    """合法成交率（排除违规成交）"""
    if not trajs:
        return 0.0
    return sum(1 for t in trajs if t.final_state.outcome == Outcome.DEAL) / len(trajs)


def deal_including_violation_rate(trajs: List[RolloutTrajectory]) -> float:
    """包括违规在内的成交率"""
    if not trajs:
        return 0.0
    return sum(1 for t in trajs if t.final_state.outcome.is_deal) / len(trajs)


def violation_rate(trajs: List[RolloutTrajectory]) -> float:
    """违反底线成交占总成交的比例"""
    dealt = [t for t in trajs if t.final_state.outcome.is_deal]
    if not dealt:
        return 0.0
    viol = sum(
        1 for t in dealt
        if t.final_state.outcome in (Outcome.VIOLATION_BUYER, Outcome.VIOLATION_SELLER)
    )
    return viol / len(dealt)


def format_error_rate(trajs: List[RolloutTrajectory]) -> float:
    if not trajs:
        return 0.0
    return sum(1 for t in trajs if t.final_state.outcome == Outcome.FORMAT_ERROR) / len(trajs)


def timeout_rate(trajs: List[RolloutTrajectory]) -> float:
    if not trajs:
        return 0.0
    return sum(1 for t in trajs if t.final_state.outcome == Outcome.TIMEOUT) / len(trajs)


def walkaway_rate(trajs: List[RolloutTrajectory]) -> float:
    if not trajs:
        return 0.0
    return sum(1 for t in trajs if t.final_state.outcome == Outcome.WALKAWAY) / len(trajs)


def pareto_efficiency(trajs: List[RolloutTrajectory]) -> float:
    """
    帕累托效率 = Σ(buyer_surplus + seller_surplus) / Σ(bargaining_zone)

    对于未成交 traj：surplus = 0
    """
    total_surplus = 0.0
    total_max = 0.0
    for t in trajs:
        sc = t.final_state.scenario
        zone = max(float(sc.buyer_budget) - float(sc.seller_cost), 0.0)
        total_max += zone
        if t.final_state.outcome == Outcome.DEAL and t.final_state.deal_price is not None:
            p = float(t.final_state.deal_price)
            buyer_surplus = float(sc.buyer_budget) - p
            seller_surplus = p - float(sc.seller_cost)
            total_surplus += buyer_surplus + seller_surplus
    if total_max <= 0:
        return 0.0
    return total_surplus / total_max


def avg_rounds(trajs: List[RolloutTrajectory]) -> float:
    """平均谈判轮数（一轮 = 双方各说一次 = 2 turn）"""
    if not trajs:
        return 0.0
    total_turns = sum(len(t.final_state.history) for t in trajs)
    return total_turns / len(trajs) / 2


def avg_surplus_by_role(trajs: List[RolloutTrajectory]) -> Dict[str, float]:
    """买家 / 卖家的平均盈余（仅成交）"""
    dealt = [t for t in trajs if t.final_state.outcome == Outcome.DEAL]
    if not dealt:
        return {"buyer_surplus": 0.0, "seller_surplus": 0.0}
    bs = sum(
        float(t.final_state.scenario.buyer_budget) - float(t.final_state.deal_price)
        for t in dealt
    ) / len(dealt)
    ss = sum(
        float(t.final_state.deal_price) - float(t.final_state.scenario.seller_cost)
        for t in dealt
    ) / len(dealt)
    return {"buyer_surplus": bs, "seller_surplus": ss}


def price_fairness(trajs: List[RolloutTrajectory]) -> float:
    """
    价格公平性 = 平均 |deal_price - midpoint| / (bargaining_zone / 2)
    越小越公平（0 表示刚好中间）
    """
    dealt = [t for t in trajs if t.final_state.outcome == Outcome.DEAL]
    if not dealt:
        return float("nan")
    total = 0.0
    for t in dealt:
        sc = t.final_state.scenario
        mid = (float(sc.buyer_budget) + float(sc.seller_cost)) / 2
        half_zone = max((float(sc.buyer_budget) - float(sc.seller_cost)) / 2, 1e-6)
        total += abs(float(t.final_state.deal_price) - mid) / half_zone
    return total / len(dealt)


def outcome_distribution(trajs: List[RolloutTrajectory]) -> Dict[str, int]:
    """各 Outcome 的计数分布"""
    c = Counter(t.final_state.outcome.value for t in trajs)
    return dict(c)


# ============================================================
# 聚合入口
# ============================================================

def compute_all_metrics(trajs: List[RolloutTrajectory]) -> Dict[str, Any]:
    """
    一次性计算所有评估指标。
    """
    surplus = avg_surplus_by_role(trajs)
    return {
        "n_trajectories": len(trajs),
        "agreement_rate": agreement_rate(trajs),
        "deal_with_violation_rate": deal_including_violation_rate(trajs),
        "violation_rate": violation_rate(trajs),
        "format_error_rate": format_error_rate(trajs),
        "timeout_rate": timeout_rate(trajs),
        "walkaway_rate": walkaway_rate(trajs),
        "pareto_efficiency": pareto_efficiency(trajs),
        "avg_rounds": avg_rounds(trajs),
        "buyer_surplus": surplus["buyer_surplus"],
        "seller_surplus": surplus["seller_surplus"],
        "price_fairness": price_fairness(trajs),
        "outcome_distribution": outcome_distribution(trajs),
    }
