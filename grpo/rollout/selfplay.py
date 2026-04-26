"""
Self-Play Rollout 引擎

核心流程：
    对每个 scenario 并行开 group_size 个 NegotiationEnv。
    按"seller 说话" / "buyer 说话" 两阶段轮流批量推理，
    直到所有 env 终止为止。

输出：按 scenario 聚合的 RolloutGroup，包含所有 trajectory 与 reward。
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

from Final_project.src.environment.scenario import NegotiationScenario
from Final_project.src.agent.prompt_builder import PromptBuilder
from Final_project.grpo.env.negotiation_env import NegotiationEnv, EnvState
from Final_project.grpo.reward.reward_fn import compute_rewards
from Final_project.grpo.reward.config import RewardConfig


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ActiveTurnRecord:
    """
    active_role 在某一 turn 的记录。
    用于构造 GRPO 的训练样本：(prompt_ids, completion_ids, advantage)
    """
    round_num: int
    prompt_text: str                   # apply_chat_template 后的完整 prompt
    completion_text: str               # 该 turn 生成的文本
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    prompt_dialogue_history: List[Dict[str, str]]


@dataclass
class RolloutTrajectory:
    scenario: NegotiationScenario
    final_state: EnvState              # env 终止时的状态
    active_role: str                   # "buyer" | "seller"（该 rollout 谁是 active）

    # 只记录 active_role 每个 turn 的生成信息
    active_turns: List[ActiveTurnRecord] = field(default_factory=list)

    # reward
    buyer_reward: float = 0.0
    seller_reward: float = 0.0
    raw_buyer_reward: float = 0.0
    raw_seller_reward: float = 0.0
    reward_breakdown: Dict[str, Any] = field(default_factory=dict)

    @property
    def advantage_reward(self) -> float:
        """供 GRPO 组内标准化使用的 role-specific reward"""
        return self.buyer_reward if self.active_role == "buyer" else self.seller_reward


@dataclass
class RolloutGroup:
    """同一 scenario 下的一组 (group_size) trajectory"""
    scenario: NegotiationScenario
    trajectories: List[RolloutTrajectory]


# ============================================================
# SelfPlayRollout
# ============================================================

class SelfPlayRollout:
    """
    self-play 对话生成。

    用法：
        rollout = SelfPlayRollout(vllm_client, tokenizer, prompt_builder, reward_cfg)
        groups = rollout.rollout_batch(
            scenarios=[s1, s2, s3],
            group_size=16,
            active_role="buyer",
            opponent_adapter="seller",
        )
    """

    def __init__(
        self,
        vllm_client,
        tokenizer,
        prompt_builder: Optional[PromptBuilder] = None,
        reward_cfg: Optional[RewardConfig] = None,
        env_config: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 128,
        max_prompt_length: int = 1536,
        temperature_active: float = 0.9,
        temperature_opponent: float = 0.7,
        top_p: float = 0.9,
        seller_cold_guard: Optional[Dict[str, Any]] = None,
    ):
        self.client = vllm_client
        self.tokenizer = tokenizer
        self.pb = prompt_builder or PromptBuilder()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.env_config = env_config or {}

        self.max_new_tokens = max_new_tokens
        self.max_prompt_length = max_prompt_length
        self.temperature_active = temperature_active
        self.temperature_opponent = temperature_opponent
        self.top_p = top_p
        self.seller_cold_guard = dict(seller_cold_guard or {})

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        """移除 <think>...</think>，避免推理链暴露给对话对手。"""
        if not text:
            return ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # 容错：若模型输出了残缺标签，也直接去掉标签文本。
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        # 容错：SFT 样本里的历史标签有时会被模型复读到回复开头。
        # 这些标签不是谈判协议的一部分，进入环境前统一剥离。
        cleaned = re.sub(r"^\s*【第\d+轮-(?:你|买家|卖家|buyer|seller)】\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(?:第\d+轮[-：:])\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _recent_buyer_offers_below_cost(self, env: NegotiationEnv, seller_cost: float) -> int:
        """Count consecutive latest buyer offers below seller cost."""
        count = 0
        for turn in reversed(env.state.history):
            if turn.role != "buyer":
                continue
            price = turn.parsed.price
            if price is not None and price < seller_cost:
                count += 1
                continue
            break
        return count

    def _apply_seller_cold_guard(
        self,
        env: NegotiationEnv,
        utterance: str,
        is_active: bool,
    ) -> str:
        """
        Stage-1-only safety patch for cold-start seller.

        This is intentionally a rollout guard, not an environment rule: it prevents
        buyer training from exploiting a weak frozen seller, while later seller
        training still has to learn the behavior itself.
        """
        cfg = self.seller_cold_guard
        if is_active or not bool(cfg.get("enabled", False)):
            return utterance

        seller_cost = float(env.scenario.seller_cost)
        last_buyer_offer = env.state.last_buyer_offer
        if last_buyer_offer is None:
            return utterance

        min_cost_ratio = float(cfg.get("min_cost_ratio", 0.8))
        consecutive_threshold = int(cfg.get("consecutive_below_cost", 2))
        consecutive_below = self._recent_buyer_offers_below_cost(env, seller_cost)

        parsed = env.parser.parse(utterance)
        bad_deal = (
            bool(cfg.get("walkaway_on_bad_deal", True))
            and parsed.action_type == "deal"
            and parsed.price is not None
            and parsed.price < seller_cost
        )
        invalid_after_low = (
            bool(cfg.get("walkaway_on_invalid_after_low", True))
            and parsed.action_type == "invalid"
            and last_buyer_offer < seller_cost
        )
        extreme_low_offer = last_buyer_offer < seller_cost * min_cost_ratio
        repeated_low_offer = consecutive_below >= consecutive_threshold

        if bad_deal or invalid_after_low or extreme_low_offer or repeated_low_offer:
            return "<walkaway>"
        return utterance

    # ------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------

    def rollout_batch(
        self,
        scenarios: List[NegotiationScenario],
        group_size: int,
        active_role: str,
        active_adapter: str,
        opponent_adapter: str,
    ) -> List[RolloutGroup]:
        """
        为每个 scenario 并行生成 group_size 条 trajectory。

        Args:
            scenarios: 本 batch 的 scenario
            group_size: 每个 scenario 的并行对话数（GRPO group）
            active_role: "buyer" 或 "seller"，表示当前训练/记录梯度的角色
            active_adapter: vLLM 里 active_role 使用的 adapter 名
            opponent_adapter: 对方使用的 adapter 名（frozen）
        """
        assert active_role in ("buyer", "seller")

        # 展平为 flat envs：每个 (scenario_idx, group_idx) 一个 env
        envs: List[NegotiationEnv] = []
        env_meta: List[Tuple[int, int]] = []    # (scenario_idx, group_idx)
        for si, sc in enumerate(scenarios):
            for gi in range(group_size):
                envs.append(NegotiationEnv(sc, config=self.env_config))
                env_meta.append((si, gi))

        # 每个 env 维护 active_role 的 turn records
        active_records: List[List[ActiveTurnRecord]] = [[] for _ in envs]

        # 循环直到所有 env 终止
        while True:
            # 收集 "下一步该 seller 说" 和 "下一步该 buyer 说" 的 env
            seller_idx = [i for i, e in enumerate(envs) if (not e.is_done()) and e.next_role() == "seller"]
            buyer_idx = [i for i, e in enumerate(envs) if (not e.is_done()) and e.next_role() == "buyer"]

            if not seller_idx and not buyer_idx:
                break  # 全部终止

            # 分别批量生成
            if seller_idx:
                self._run_role(
                    envs, seller_idx, active_records,
                    role="seller",
                    active_role=active_role,
                    active_adapter=active_adapter,
                    opponent_adapter=opponent_adapter,
                )
            if buyer_idx:
                self._run_role(
                    envs, buyer_idx, active_records,
                    role="buyer",
                    active_role=active_role,
                    active_adapter=active_adapter,
                    opponent_adapter=opponent_adapter,
                )

        # 构造输出
        groups: List[RolloutGroup] = [
            RolloutGroup(scenario=sc, trajectories=[]) for sc in scenarios
        ]
        for i, env in enumerate(envs):
            si, gi = env_meta[i]
            r = compute_rewards(env.state, self.reward_cfg)
            traj = RolloutTrajectory(
                scenario=scenarios[si],
                final_state=env.state,
                active_role=active_role,
                active_turns=active_records[i],
                buyer_reward=r["buyer_reward"],
                seller_reward=r["seller_reward"],
                raw_buyer_reward=r.get("raw_buyer_reward", r["buyer_reward"]),
                raw_seller_reward=r.get("raw_seller_reward", r["seller_reward"]),
                reward_breakdown=r["breakdown"],
            )
            groups[si].trajectories.append(traj)

        return groups

    # ------------------------------------------------------------
    # 单 role 的批量生成
    # ------------------------------------------------------------

    def _run_role(
        self,
        envs: List[NegotiationEnv],
        indices: List[int],
        active_records: List[List[ActiveTurnRecord]],
        role: str,
        active_role: str,
        active_adapter: str,
        opponent_adapter: str,
    ) -> None:
        """
        让 indices 列表里的 envs 按当前 role 生成一轮发言。
        """
        # 1. 构造 prompts
        messages_list: List[List[Dict[str, str]]] = []
        prompts: List[str] = []
        prompt_histories: List[List[Dict[str, str]]] = []
        for idx in indices:
            env = envs[idx]
            dialogue_history = env.get_dialogue_history_for(role)
            trimmed_history = self.pb.trim_dialogue_history_to_budget(
                tokenizer=self.tokenizer,
                role=role,
                scenario=env.scenario,
                dialogue_history=dialogue_history,
                max_prompt_tokens=self.max_prompt_length,
            )
            msgs = self.pb.build_messages(
                role=role,
                scenario=env.scenario,
                dialogue_history=trimmed_history,
            )
            messages_list.append(msgs)
            prompt_histories.append([dict(turn) for turn in trimmed_history])
            prompts.append(self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            ))

        # 2. 决定用哪个 adapter 和 temperature
        is_active = (role == active_role)
        adapter_name = active_adapter if is_active else opponent_adapter
        temperature = self.temperature_active if is_active else self.temperature_opponent

        # 3. 批量生成
        outputs = self.client.generate(
            prompts=prompts,
            adapter_name=adapter_name,
            temperature=temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
        )

        # 4. step env + 记录 active turn
        for idx, out, prompt_history in zip(indices, outputs, prompt_histories):
            env = envs[idx]
            raw_utterance = out.text.strip()
            utterance = self._strip_think_blocks(raw_utterance)
            if role == "seller":
                utterance = self._apply_seller_cold_guard(env, utterance, is_active)

            # 记录 active role 的 turn（用于训练）
            if is_active:
                active_records[idx].append(ActiveTurnRecord(
                    round_num=env.state.current_round,
                    prompt_text=out.prompt,
                    completion_text=raw_utterance,
                    prompt_token_ids=out.prompt_token_ids,
                    completion_token_ids=out.completion_token_ids,
                    prompt_dialogue_history=prompt_history,
                ))

            env.step(utterance)
