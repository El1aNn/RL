"""
reward_batch.py

把 SelfPlayRollout 产出的 RolloutGroup 列表转换为
TRL/自定义 trainer 可消费的批量 reward 格式。

主要用途：
- 在 NegotiationGRPOTrainer._build_flat_samples 之前做额外的 reward 后处理
- 提供 normalize / clip / logging-friendly 的辅助函数
- 如果以后切换回 trl.GRPOTrainer，也可以直接把这里的函数包成 reward_funcs
"""
from typing import List, Dict, Any, Optional
import math

from Final_project.grpo.rollout.selfplay import RolloutGroup, RolloutTrajectory


# ============================================================
# 辅助：组内 reward 统计
# ============================================================

def group_reward_stats(group: RolloutGroup, role: str) -> Dict[str, float]:
    """
    计算单个 group 内 active_role reward 的统计量。
    role: "buyer" | "seller"
    """
    rewards = [
        t.buyer_reward if role == "buyer" else t.seller_reward
        for t in group.trajectories
    ]
    n = len(rewards)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mu = sum(rewards) / n
    var = sum((r - mu) ** 2 for r in rewards) / n
    return {
        "mean": mu,
        "std": math.sqrt(var),
        "min": min(rewards),
        "max": max(rewards),
    }


# ============================================================
# 批量展平：groups → per-trajectory reward dicts
# ============================================================

def flatten_rewards(
    groups: List[RolloutGroup],
    active_role: str,
    normalize_within_group: bool = True,
    advantage_eps: float = 1e-4,
    clip_range: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    把 RolloutGroup 列表展平为每条 trajectory 的 reward 记录。

    返回 list of dict，每个 dict 包含：
        scenario_id:    str
        trajectory_idx: int（组内序号）
        buyer_reward:   float
        seller_reward:  float
        advantage:      float（active_role 的 z-score reward；未启用标准化时 = raw reward）
        outcome:        str
        deal_price:     float | None
        n_turns:        int
        reward_breakdown: dict

    Args:
        groups: rollout_batch 的返回值
        active_role: "buyer" | "seller"
        normalize_within_group: 是否做组内 z-score
        advantage_eps: z-score 分母的 epsilon
        clip_range: 若非 None，对 advantage 做 ±clip_range 截断
    """
    records: List[Dict[str, Any]] = []

    for group in groups:
        trajs = group.trajectories
        if not trajs:
            continue

        raw_rewards = [
            t.buyer_reward if active_role == "buyer" else t.seller_reward
            for t in trajs
        ]

        if normalize_within_group:
            mu = sum(raw_rewards) / len(raw_rewards)
            var = sum((r - mu) ** 2 for r in raw_rewards) / len(raw_rewards)
            sigma = math.sqrt(var) + advantage_eps
            advantages = [(r - mu) / sigma for r in raw_rewards]
        else:
            advantages = list(raw_rewards)

        if clip_range is not None:
            advantages = [max(-clip_range, min(clip_range, a)) for a in advantages]

        for idx, (traj, adv) in enumerate(zip(trajs, advantages)):
            sc = group.scenario
            records.append({
                "scenario_id": getattr(sc, "scenario_id", str(id(sc))),
                "trajectory_idx": idx,
                "buyer_reward": traj.buyer_reward,
                "seller_reward": traj.seller_reward,
                "advantage": adv,
                "outcome": traj.final_state.outcome.value,
                "deal_price": traj.final_state.deal_price,
                "n_turns": len(traj.final_state.history),
                "reward_breakdown": traj.reward_breakdown,
            })

    return records


# ============================================================
# 聚合统计（用于 logging）
# ============================================================

def aggregate_batch_stats(
    groups: List[RolloutGroup],
    active_role: str,
) -> Dict[str, float]:
    """
    对整个 batch（所有 group）的 reward / 成交率等做快速聚合统计，
    方便在 trainer 里一行打日志。
    """
    all_trajs: List[RolloutTrajectory] = [t for g in groups for t in g.trajectories]
    n = len(all_trajs) or 1

    raw_rewards = [
        t.buyer_reward if active_role == "buyer" else t.seller_reward
        for t in all_trajs
    ]
    mu = sum(raw_rewards) / n
    var = sum((r - mu) ** 2 for r in raw_rewards) / n

    from Final_project.grpo.env.outcome import Outcome
    deal_n = sum(1 for t in all_trajs if t.final_state.outcome == Outcome.DEAL)
    timeout_n = sum(1 for t in all_trajs if t.final_state.outcome == Outcome.TIMEOUT)
    walkaway_n = sum(1 for t in all_trajs if t.final_state.outcome == Outcome.WALKAWAY)
    fmt_err_n = sum(1 for t in all_trajs if t.final_state.outcome == Outcome.FORMAT_ERROR)

    return {
        "n_trajectories": n,
        "reward_mean": mu,
        "reward_std": math.sqrt(var),
        "reward_min": min(raw_rewards) if raw_rewards else 0.0,
        "reward_max": max(raw_rewards) if raw_rewards else 0.0,
        "deal_rate": deal_n / n,
        "timeout_rate": timeout_n / n,
        "walkaway_rate": walkaway_n / n,
        "format_error_rate": fmt_err_n / n,
        "avg_turns": sum(len(t.final_state.history) for t in all_trajs) / n,
    }
