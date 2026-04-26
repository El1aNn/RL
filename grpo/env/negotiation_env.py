"""
NegotiationEnv：管理单场买卖谈判对话的状态机

职责：
- 接收当前角色的发言 → 解析 → 更新状态
- 检查终止条件（deal / walkaway / timeout / format_error）
- 对外提供 next_role、is_done、get_dialogue_history_for

协议：seller 先开口（current_turn_role 初始值为 "seller"）。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from Final_project.grpo.env.parser import DialogueParser, ParseResult
from Final_project.grpo.env.outcome import Outcome


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Turn:
    round_num: int
    role: str               # "buyer" | "seller"
    utterance: str          # 原始文本
    parsed: ParseResult     # 解析结果


@dataclass
class EnvState:
    scenario: Any                          # NegotiationScenario（避免循环 import）
    history: List[Turn] = field(default_factory=list)
    current_round: int = 0                 # 已完成的完整轮数（seller+buyer 各说一次 = 1轮）
    current_turn_role: str = "seller"      # 下一个要发话的角色（seller 先开口）
    outcome: Outcome = Outcome.ONGOING
    deal_price: Optional[float] = None
    terminated_reason: Optional[str] = None
    format_violations: int = 0
    # 最近一次各方 offer（用于 walkaway 判断）
    last_seller_offer: Optional[float] = None
    last_buyer_offer: Optional[float] = None

    @property
    def is_done(self) -> bool:
        return self.outcome != Outcome.ONGOING


# ============================================================
# NegotiationEnv
# ============================================================

class NegotiationEnv:
    """
    纯文本谈判环境状态机。不负责生成，只接收发言并推进状态。

    终止条件优先级：
    1. <walkaway> → WALKAWAY
    2. <deal>X</deal> → DEAL 或 VIOLATION_BUYER / VIOLATION_SELLER
    3. 格式缺失 → format_violations++，若超预算 → FORMAT_ERROR；否则用 fallback 继续
    4. 双方均发完 max_rounds 轮 → TIMEOUT
    """

    DEFAULT_CONFIG = {
        "format_error_budget": 2,
    }

    def __init__(self, scenario, config: Optional[Dict[str, Any]] = None, parser=None):
        self.scenario = scenario
        self.cfg = dict(self.DEFAULT_CONFIG)
        if config:
            self.cfg.update(config)
        self.parser = parser or DialogueParser()
        self.state = EnvState(scenario=scenario)

    # ------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------

    def step(self, utterance: str) -> EnvState:
        """
        当前 current_turn_role 发出 utterance，推进状态。
        如果 env 已经终止则忽略（直接返回当前 state）。
        """
        if self.state.is_done:
            return self.state

        role = self.state.current_turn_role
        parsed = self.parser.parse(utterance)
        turn = Turn(
            round_num=self.state.current_round,
            role=role,
            utterance=utterance,
            parsed=parsed,
        )
        self.state.history.append(turn)

        # 更新最近报价记录
        if parsed.action_type in ("offer", "deal") and parsed.price is not None:
            if role == "seller":
                self.state.last_seller_offer = parsed.price
            else:
                self.state.last_buyer_offer = parsed.price

        # 检查终止
        if parsed.action_type == "walkaway":
            self.state.outcome = Outcome.WALKAWAY
            self.state.terminated_reason = f"{role}_walkaway"
            return self.state

        if parsed.action_type == "deal":
            self._resolve_deal(parsed.price, role)
            return self.state

        if parsed.action_type == "invalid":
            # 格式错误：fallback 继续
            self.state.format_violations += 1
            fallback_price = self._fallback_price(role)
            # 用 fallback 覆盖更新最近报价
            if role == "seller":
                self.state.last_seller_offer = fallback_price
            else:
                self.state.last_buyer_offer = fallback_price

            if self.state.format_violations > int(self.cfg["format_error_budget"]):
                self.state.outcome = Outcome.FORMAT_ERROR
                self.state.terminated_reason = f"{role}_format_error"
                return self.state

        # 推进轮数与角色
        self._advance_turn(role)
        return self.state

    def is_done(self) -> bool:
        return self.state.is_done

    def next_role(self) -> str:
        return self.state.current_turn_role

    def get_dialogue_history_for(self, role: str) -> List[Dict[str, str]]:
        """
        返回适合 PromptBuilder 使用的原始对话历史。
        PromptBuilder 会再根据当前角色映射成 chat template 的 user/assistant。
        """
        history = []
        for turn in self.state.history:
            history.append({
                "round": turn.round_num,
                "role": turn.role,
                "utterance": turn.utterance,
            })
        return history

    # ------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------

    def _resolve_deal(self, price: float, who_proposed: str) -> None:
        """判断成交是否违规，设置 Outcome"""
        buyer_budget = float(self.scenario.buyer_budget)
        seller_cost = float(self.scenario.seller_cost)
        self.state.deal_price = price

        if price > buyer_budget:
            self.state.outcome = Outcome.VIOLATION_BUYER
            self.state.terminated_reason = f"{who_proposed}_deal_violation_buyer"
        elif price < seller_cost:
            self.state.outcome = Outcome.VIOLATION_SELLER
            self.state.terminated_reason = f"{who_proposed}_deal_violation_seller"
        else:
            self.state.outcome = Outcome.DEAL
            self.state.terminated_reason = f"{who_proposed}_deal"

    def _fallback_price(self, role: str) -> float:
        """格式错误时的占位报价：取保守中间值"""
        buyer_budget = float(self.scenario.buyer_budget)
        seller_cost = float(self.scenario.seller_cost)
        mid = (buyer_budget + seller_cost) / 2
        if role == "buyer":
            return min(mid, buyer_budget)
        else:
            return max(mid, seller_cost)

    def _advance_turn(self, just_spoke: str) -> None:
        """切换到对方发言；如果双方都完成了 max_rounds 则 TIMEOUT"""
        next_role = "buyer" if just_spoke == "seller" else "seller"
        self.state.current_turn_role = next_role

        # 每当 buyer 说完即完成一整轮
        if just_spoke == "buyer":
            self.state.current_round += 1
            max_rounds = int(getattr(self.scenario, "max_rounds", 10))
            if self.state.current_round >= max_rounds:
                self.state.outcome = Outcome.TIMEOUT
                self.state.terminated_reason = "timeout"
