"""
主 Reward 函数

根据 EnvState 的 Outcome + 数值，计算 buyer / seller 各自的 reward。
返回结构化的明细，方便日志记录。
"""
from typing import Dict, Any

from Final_project.grpo.env.negotiation_env import EnvState
from Final_project.grpo.env.outcome import Outcome
from Final_project.grpo.reward.config import RewardConfig
from Final_project.grpo.reward.shaping import compute_shaping_for_role


def _safe_zone(state: EnvState, cfg: RewardConfig) -> float:
    """bargaining zone，最小 zone_floor 防方差爆炸"""
    zone = float(state.scenario.buyer_budget) - float(state.scenario.seller_cost)
    return max(zone, cfg.zone_floor)


def _judge_walkaway(
    state: EnvState, who_walked_away: str, cfg: RewardConfig,
) -> Dict[str, float]:
    """
    判断 walkaway 是合理还是错误。

    合理：若继续谈，对方最近报价已经违反自己的底线。
    错误：对方报价已经在自己可接受范围内，却放弃。
    """
    buyer_budget = float(state.scenario.buyer_budget)
    seller_cost = float(state.scenario.seller_cost)

    buyer_reward = 0.0
    seller_reward = 0.0

    if who_walked_away == "buyer":
        # 对方（seller）最近报价是否 <= budget？
        last_offer = state.last_seller_offer
        if last_offer is None:
            # 还没看到对方报价就 walk，按「错误」处理（缺乏诚意）
            buyer_reward = cfg.walkaway_wrong
        elif last_offer <= buyer_budget:
            buyer_reward = cfg.walkaway_wrong
        else:
            buyer_reward = cfg.walkaway_right
        # seller 被放弃，也视为错误（没谈成）
        seller_reward = cfg.walkaway_wrong
    else:
        # seller 放弃
        last_offer = state.last_buyer_offer
        if last_offer is None:
            seller_reward = cfg.walkaway_wrong
        elif last_offer >= seller_cost:
            seller_reward = cfg.walkaway_wrong
        else:
            seller_reward = cfg.walkaway_right
        buyer_reward = cfg.walkaway_wrong

    return {"buyer": buyer_reward, "seller": seller_reward}


def _terminal_rewards(state: EnvState, cfg: RewardConfig) -> Dict[str, float]:
    """根据 Outcome 计算终止态的主 reward（不含 shaping）"""
    buyer_budget = float(state.scenario.buyer_budget)
    seller_cost = float(state.scenario.seller_cost)
    zone = _safe_zone(state, cfg)

    outcome = state.outcome

    if outcome == Outcome.DEAL:
        p = float(state.deal_price)
        buyer_r = cfg.deal_scale * (buyer_budget - p) / zone
        seller_r = cfg.deal_scale * (p - seller_cost) / zone
        return {"buyer": buyer_r, "seller": seller_r}

    if outcome == Outcome.VIOLATION_BUYER:
        # 买家违规（price > budget）：买家重罚；卖家仍按合法计算（上限为 deal_scale）
        p = float(state.deal_price)
        seller_r = cfg.deal_scale * (p - seller_cost) / zone
        return {"buyer": cfg.violation_penalty, "seller": seller_r}

    if outcome == Outcome.VIOLATION_SELLER:
        # 卖家违规（price < cost）
        p = float(state.deal_price)
        buyer_r = cfg.deal_scale * (buyer_budget - p) / zone
        return {"buyer": buyer_r, "seller": cfg.violation_penalty}

    if outcome == Outcome.WALKAWAY:
        # 区分主动放弃者
        reason = state.terminated_reason or ""
        who = "buyer" if reason.startswith("buyer") else "seller"
        return _judge_walkaway(state, who, cfg)

    if outcome == Outcome.TIMEOUT:
        return {"buyer": cfg.timeout, "seller": cfg.timeout}

    if outcome == Outcome.FORMAT_ERROR:
        # 格式错误只罚犯错那一方；另一方不奖不罚（避免"躺赢"）
        reason = state.terminated_reason or ""
        who = "buyer" if reason.startswith("buyer") else "seller"
        rewards = {"buyer": 0.0, "seller": 0.0}
        rewards[who] = cfg.format_error
        return rewards

    # ONGOING 不应该进来
    return {"buyer": 0.0, "seller": 0.0}


def compute_rewards(
    state: EnvState,
    cfg: RewardConfig,
) -> Dict[str, Any]:
    """
    计算一条 trajectory 的 buyer / seller reward。

    Returns:
        {
            "buyer_reward": float,
            "seller_reward": float,
            "breakdown": {
                "buyer": {"terminal": ..., "format_bonus": ..., ...},
                "seller": {...}
            }
        }
    """
    assert state.is_done, "compute_rewards should be called only after env terminates"

    terminal = _terminal_rewards(state, cfg)

    buyer_shaping = compute_shaping_for_role(state, "buyer", cfg)
    seller_shaping = compute_shaping_for_role(state, "seller", cfg)

    buyer_total = terminal["buyer"] + sum(buyer_shaping.values())
    seller_total = terminal["seller"] + sum(seller_shaping.values())

    return {
        "buyer_reward": buyer_total,
        "seller_reward": seller_total,
        "breakdown": {
            "buyer": {"terminal": terminal["buyer"], **buyer_shaping},
            "seller": {"terminal": terminal["seller"], **seller_shaping},
        },
    }
