"""
Prompt 构造器

核心设计原则（来自 plan.md 2.4 节）：
- 买家和卖家使用同一个模型，通过不同的 system prompt 切换角色
- 当前角色的发言始终在 assistant 位置，对方发言在 user 位置
- 角色切换时，对话历史中 user/assistant 的映射要相应翻转
"""
from typing import List, Dict, Optional
from Final_project.src.environment.scenario import NegotiationScenario


# ============================================================
# 系统提示词模板
# ============================================================

BUYER_SYSTEM_TEMPLATE = """\
你是一个精明的买家，正在和卖家谈判购买"{item_name}"。

## 你的私密信息（绝对不要透露给卖家！）
- 你的最高预算：{buyer_budget} 元
- 你的目标：以尽可能低的价格买到这件商品

## 公共信息
- 商品：{item_name}
- 商品描述：{item_description}
- 市场参考价：{market_ref_price} 元

## 对话规则
1. 每次发言必须包含你的报价，格式为 [报价：XXX]
2. 如果你接受对方的报价，输出 <deal>价格</deal> 来达成交易
3. 如果你决定放弃谈判，输出 <walkaway>
4. 绝对不要以高于 {buyer_budget} 元的价格达成交易
5. 不要透露或暗示你的最高预算、预算上限、底线、底价等私密信息
6. 不要在回复里输出"第X轮"、"买家"、"卖家"、"你"等对话历史标签
7. 不要过早接受明显偏向某一方的价格；优先争取双方都合理的成交
8. 你可以使用各种谈判技巧，比如强调商品的缺点、货比三家等"""

SELLER_SYSTEM_TEMPLATE = """\
你是一个精明的卖家，正在和买家谈判出售"{item_name}"。

## 你的私密信息（绝对不要透露给买家！）
- 你的最低售价：{seller_cost} 元
- 你的目标：以尽可能高的价格卖出这件商品

## 公共信息
- 商品：{item_name}
- 商品描述：{item_description}
- 市场参考价：{market_ref_price} 元

## 对话规则
1. 每次发言必须包含你的报价，格式为 [报价：XXX]
2. 如果你接受对方的报价，输出 <deal>价格</deal> 来达成交易
3. 如果你决定放弃谈判，输出 <walkaway>
4. 绝对不要以低于 {seller_cost} 元的价格达成交易
5. 不要透露或暗示你的最低售价、成本、底线、底价等私密信息
6. 不要在回复里输出"第X轮"、"买家"、"卖家"、"你"等对话历史标签
7. 不要过早接受明显偏向某一方的价格；优先争取双方都合理的成交
8. 你可以使用各种谈判技巧，比如强调商品的优点、提及其他买家有意等"""


# ============================================================
# Prompt Builder
# ============================================================

class PromptBuilder:
    """
    为买家/卖家构造 chat-style 的 messages 列表。

    输出格式（适用于 SFT 和 Rollout）：
    [
        {"role": "system",    "content": "你是一个精明的买家..."},
        {"role": "user",      "content": "【第1轮-卖家】[报价：5000] ..."},
        {"role": "assistant", "content": "【第1轮-买家】[报价：3200] ..."},
        {"role": "user",      "content": "【第2轮-卖家】[报价：4200] ..."},
    ]
    最后一个 user 消息之后，模型在 assistant 位置生成回复。
    """

    def __init__(
        self,
        buyer_template: str = BUYER_SYSTEM_TEMPLATE,
        seller_template: str = SELLER_SYSTEM_TEMPLATE,
    ):
        self.buyer_template = buyer_template
        self.seller_template = seller_template

    # ----------------------------------------------------------
    # 系统提示词
    # ----------------------------------------------------------

    def build_system_prompt(
        self,
        role: str,
        scenario: NegotiationScenario,
    ) -> str:
        """
        构造指定角色的 system prompt。

        Args:
            role: "buyer" 或 "seller"
            scenario: 谈判场景
        Returns:
            填充后的 system prompt 字符串
        """
        template = self.buyer_template if role == "buyer" else self.seller_template
        return template.format(
            item_name=scenario.item_name,
            item_description=scenario.item_description,
            buyer_budget=scenario.buyer_budget,
            seller_cost=scenario.seller_cost,
            market_ref_price=scenario.market_ref_price,
        )

    # ----------------------------------------------------------
    # 对话历史 → messages 列表
    # ----------------------------------------------------------

    @staticmethod
    def _format_turn_text(round_num: int, role: str, utterance: str) -> str:
        """格式化单轮发言文本：【第X轮-角色】内容"""
        role_cn = "买家" if role == "buyer" else "卖家"
        return f"【第{round_num}轮-{role_cn}】{utterance}"

    def build_messages(
        self,
        role: str,
        scenario: NegotiationScenario,
        dialogue_history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        为指定角色构造完整的 messages 列表（用于模型推理）。

        核心逻辑：
        - 当前角色的发言 → assistant
        - 对方角色的发言 → user
        - 这样 LLM 自然地在 assistant 位置生成下一轮回复

        Args:
            role: 当前角色，"buyer" 或 "seller"
            scenario: 谈判场景
            dialogue_history: 按时间顺序排列的对话历史，每项为
                {"round": int, "role": str, "utterance": str}
        Returns:
            messages 列表，可直接传给 tokenizer.apply_chat_template
        """
        messages = [
            {"role": "system", "content": self.build_system_prompt(role, scenario)}
        ]

        for turn in dialogue_history:
            turn_text = self._format_turn_text(
                round_num=turn["round"],
                role=turn["role"],
                utterance=turn["utterance"],
            )
            # 关键：当前角色 → assistant，对方 → user
            if turn["role"] == role:
                messages.append({"role": "assistant", "content": turn_text})
            else:
                messages.append({"role": "user", "content": turn_text})

        return messages

    def trim_dialogue_history_to_budget(
        self,
        tokenizer,
        role: str,
        scenario: NegotiationScenario,
        dialogue_history: List[Dict[str, str]],
        max_prompt_tokens: int,
    ) -> List[Dict[str, str]]:
        """
        保留 system prompt，并尽量保留最近完整轮次的对话历史。

        如果完整历史超出 token 预算，就按“轮”从最早开始裁掉，
        直到 system + 最近 K 轮能够放进 max_prompt_tokens。
        """
        history = [dict(turn) for turn in dialogue_history]
        if not history:
            return history

        rounds = []
        for turn in history:
            if turn["round"] not in rounds:
                rounds.append(turn["round"])

        for start_idx in range(len(rounds) + 1):
            if start_idx == len(rounds):
                trimmed_history = []
            else:
                min_round = rounds[start_idx]
                trimmed_history = [turn for turn in history if turn["round"] >= min_round]

            messages = self.build_messages(role, scenario, trimmed_history)
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(prompt_ids) <= max_prompt_tokens:
                return trimmed_history

        return []

    # ----------------------------------------------------------
    # 构造 SFT 训练样本
    # ----------------------------------------------------------

    def build_sft_sample(
        self,
        role: str,
        scenario: NegotiationScenario,
        dialogue_history: List[Dict[str, str]],
        current_response: Dict[str, str],
    ) -> Dict:
        """
        构造一条 SFT 训练样本。

        对话历史作为 context（system + 交替的 user/assistant），
        当前角色本轮的回复作为最终的 assistant（训练目标）。

        Args:
            role: 当前角色
            scenario: 谈判场景
            dialogue_history: 到当前轮之前的所有历史
            current_response: 当前轮回复 {"round": int, "role": str, "utterance": str}
        Returns:
            {"messages": [...]}，直接可序列化为 SFT jsonl
        """
        messages = self.build_messages(role, scenario, dialogue_history)
        # 加上当前轮的回复作为 assistant 训练目标
        response_text = self._format_turn_text(
            round_num=current_response["round"],
            role=current_response["role"],
            utterance=current_response["utterance"],
        )
        messages.append({"role": "assistant", "content": response_text})
        return {"messages": messages}

    # ----------------------------------------------------------
    # 从完整对话中提取所有 SFT 样本
    # ----------------------------------------------------------

    def extract_all_sft_samples(
        self,
        scenario: NegotiationScenario,
        full_dialogue: List[Dict[str, str]],
    ) -> List[Dict]:
        """
        从一条完整对话中提取所有可能的 SFT 训练样本。

        对于每一轮发言，都以该角色的视角生成一条训练样本：
        - context = 该角色视角下的对话历史
        - target  = 该轮的回复

        这样一条 N 轮对话可以产出 N 条 SFT 样本。

        Args:
            scenario: 谈判场景
            full_dialogue: 完整对话列表
        Returns:
            SFT 样本列表
        """
        samples = []
        for i, turn in enumerate(full_dialogue):
            history_before = full_dialogue[:i]
            sample = self.build_sft_sample(
                role=turn["role"],
                scenario=scenario,
                dialogue_history=history_before,
                current_response=turn,
            )
            # 附加元信息，方便调试
            sample["metadata"] = {
                "scenario_id": scenario.scenario_id,
                "role": turn["role"],
                "round": turn["round"],
                "turn_index": i,
            }
            samples.append(sample)
        return samples
