"""
SFT 数据构造脚本 —— 调用 LLM API 生成训练对话

流程：
1. 读取场景 JSONL
2. 为每个场景构造一个「对话生成 prompt」发给 LLM
3. 解析 LLM 返回的对话 JSON
4. 用 PromptBuilder 转换为标准 SFT messages 格式
5. 输出 sft_dialogues.jsonl

用法：
    # 先设置环境变量
    export ARK_API_KEY="your-api-key"

    python3 data/generate_sft_data.py \
        --scenarios data/scenarios/train.jsonl \
        --output data/sft/sft_dialogues.jsonl \
        --num-dialogues 1000 \
        --model ep-20260309141351-27swm
"""
import json
import os
import random
import argparse
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from volcenginesdkarkruntime import Ark
from tqdm import tqdm
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment.scenario import NegotiationScenario
from src.agent.prompt_builder import PromptBuilder

ROOT = Path(__file__).resolve().parent.parent


def log_progress(message: str) -> None:
    """配合 tqdm 输出日志，避免打乱进度条显示。"""
    tqdm.write(message)


# ============================================================
# 用于调用 LLM 生成对话的 Meta-Prompt
# ============================================================

DIALOGUE_GENERATION_PROMPT = """\
你是一个数据生成助手。请根据以下谈判场景信息，生成一段真实自然的买家-卖家谈判对话。

## 场景信息
- 商品名称：{item_name}
- 商品描述：{item_description}
- 市场参考价：{market_ref_price} 元
- 买家最高预算（买家的秘密底线）：{buyer_budget} 元
- 卖家最低售价（卖家的秘密底线）：{seller_cost} 元
- 最大谈判轮数：{max_rounds}
- 当前谈判空间类型：{zone_profile_name}
- 当前谈判空间比例：约 {gap_ratio_percent}%

## 本次样本模式
- 模式编号：{mode_id}
- 模式名称：{mode_name}
- 模式说明：{mode_description}
- 卖家策略：{seller_strategy}
- 买家策略：{buyer_strategy}
- 节奏要求：{pace_instruction}
- 让步规律：{concession_instruction}
- 额外约束：{extra_constraints}

## 对话格式要求

对话中每次发言必须严格遵循以下三种格式之一：

1. **报价/讨价还价**（继续谈判）：
   [报价：XXX] 你的发言内容...

2. **接受对方报价，达成交易**：
   <deal>价格</deal>

3. **放弃谈判**：
   <walkaway>

## 对话生成要求

1. 卖家先开口，买家后回复，交替进行
2. 谈判轮数控制在 {min_rounds}~{target_rounds} 轮（一轮 = 卖家说一次 + 买家说一次）
3. 谈判结局为「{outcome}」：
   - deal：双方达成交易，最终由{deal_initiator}发出 <deal>价格</deal>，成交价必须在 {seller_cost}~{buyer_budget} 之间
   - walkaway：谈判破裂，由某一方发出 <walkaway>
   - timeout：轮数用完仍未达成（双方在最后一轮仍在正常报价）
4. 卖家的报价应从高到低逐步让步，买家的报价应从低到高逐步加价
5. 双方都不知道对方的底线，不能泄露自己的底线
6. 对话要自然真实、有谈判策略，可以使用：
   - 锚定效应（首轮开高/低价）
   - 虚张声势（"别人出了更高价"、"我预算只有这么多"）
   - 强调商品优缺点
   - 最后通牒（"这是最后报价"）
   - 合理的让步理由
7. 每次发言控制在 1~3 句话，简洁有力
8. 不要把让步做成机械模板，例如不能每轮都刚好加 100 或降 100
9. 即使存在谈判空间，也不一定必须成交；请严格按照本次模式要求控制结局
10. 不要在对话中直接说出“底线”“最高预算”“最低售价”等明确泄露私密信息的话

## 输出格式

请严格输出以下 JSON 格式（不要输出其他任何内容）：

```json
{{
  "dialogue": [
    {{"round": 1, "role": "seller", "utterance": "[报价：XXXX] 卖家的发言..."}},
    {{"round": 1, "role": "buyer", "utterance": "[报价：XXXX] 买家的发言..."}},
    {{"round": 2, "role": "seller", "utterance": "[报价：XXXX] 卖家的发言..."}},
    {{"round": 2, "role": "buyer", "utterance": "[报价：XXXX] 买家的发言..."}},
    ...
  ],
  "terminated_reason": "{outcome}"
}}
```

注意：
- 只输出 JSON，不要有任何其他文字
- deal 结局的最后一条消息必须包含 <deal>价格</deal> 标签
- walkaway 结局的最后一条消息必须包含 <walkaway> 标签
- 报价数字必须是整数"""

PRICE_PATTERN = re.compile(r'\[报价[：:]\s*(\d+)\]')
DEAL_PATTERN = re.compile(r'<deal>\s*(\d+(?:\.\d+)?)\s*</deal>')
PRIVATE_INFO_PATTERNS = [
    re.compile(pattern) for pattern in [
        r"底线",
        r"最高预算",
        r"最低售价",
        r"卖家成本",
        r"买家预算上限",
    ]
]


# ============================================================
# LLM API 调用封装
# ============================================================

class RetryableGenerationError(Exception):
    """可通过重试恢复的生成错误"""

    def __init__(self, message: str, reason: Optional[str] = None):
        super().__init__(message)
        self.reason = reason


class NonRetryableGenerationError(Exception):
    """不应继续重试的生成错误"""


class LLMClient:
    """使用火山方舟 Ark Responses API 调用模型"""

    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self.base_url = base_url or "https://ark-cn-beijing.bytedance.net/api/v3"

        if not self.api_key:
            raise ValueError("未找到 API Key，请通过 --api-key 或环境变量 ARK_API_KEY 提供")

        self.client = Ark(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    @staticmethod
    def _format_incomplete_details(response) -> str:
        """格式化 incomplete / failed 的附加信息"""
        details = getattr(response, "incomplete_details", None)
        error = getattr(response, "error", None)

        detail_parts = []
        if details:
            reason = getattr(details, "reason", None)
            if reason:
                detail_parts.append(f"reason={reason}")
        if error:
            code = getattr(error, "code", None)
            message = getattr(error, "message", None)
            if code:
                detail_parts.append(f"code={code}")
            if message:
                detail_parts.append(f"message={message}")

        return ", ".join(detail_parts) if detail_parts else "no_details"

    @staticmethod
    def _extract_incomplete_reason(response) -> Optional[str]:
        """提取 incomplete 的原因字段"""
        details = getattr(response, "incomplete_details", None)
        return getattr(details, "reason", None) if details else None

    def _extract_text(self, response) -> str:
        """从 Ark Responses API 返回中提取模型文本输出"""
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text" and getattr(content, "text", None):
                    return content.text.strip()

        status = getattr(response, "status", "unknown")
        details = self._format_incomplete_details(response)
        reason = self._extract_incomplete_reason(response)

        if status in ("incomplete", "failed", "in_progress"):
            raise RetryableGenerationError(
                f"Ark Responses API 未返回可解析文本，status={status}, details={details}",
                reason=reason,
            )

        raise NonRetryableGenerationError(
            f"Ark Responses API 返回未知状态且无文本，status={status}, details={details}"
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> str:
        """调用 Ark Responses API 生成文本"""
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout=timeout,
        )
        return self._extract_text(response)


# ============================================================
# 对话生成与解析
# ============================================================

def load_dialogue_modes(path: str) -> List[Dict]:
    """加载对话模式配置"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["dialogue_mode_profiles"]


def choose_weighted(items: List[Dict]) -> Dict:
    """按权重随机抽取配置项"""
    weights = [item.get("weight", 1.0) for item in items]
    return random.choices(items, weights=weights, k=1)[0]


def compute_retry_delay(
    attempt_idx: int,
    base_delay: float,
    max_delay: float,
    jitter_ratio: float,
) -> float:
    """指数退避 + 抖动"""
    delay = min(max_delay, base_delay * (2 ** attempt_idx))
    jitter = delay * jitter_ratio
    return max(0.0, delay + random.uniform(-jitter, jitter))


def infer_zone_profile(scenario: NegotiationScenario) -> str:
    """从场景 metadata 或 gap ratio 推断谈判空间类型"""
    metadata_zone = scenario.metadata.get("zone_profile")
    if metadata_zone:
        return metadata_zone

    ratio = scenario.gap_ratio
    if ratio < 0.08:
        return "near_zero_space"
    if ratio < 0.18:
        return "narrow_space"
    if ratio < 0.40:
        return "balanced_space"
    return "wide_space"


def select_generation_mode(
    scenario: NegotiationScenario,
    mode_profiles: List[Dict],
) -> Dict:
    """根据场景空间类型抽取一个合适的对话模式"""
    zone_profile = infer_zone_profile(scenario)
    eligible = [
        profile for profile in mode_profiles
        if zone_profile in profile.get("zone_profiles", [])
    ]
    if not eligible:
        eligible = mode_profiles

    mode = choose_weighted(eligible)
    round_lo, round_hi = mode["round_range"]

    if mode["outcome"] == "timeout":
        min_rounds = max(2, scenario.max_rounds - 1)
        target_rounds = scenario.max_rounds
        deal_initiator = "无"
    else:
        round_hi = min(round_hi, scenario.max_rounds)
        round_lo = min(round_lo, round_hi)
        target_rounds = random.randint(round_lo, round_hi)
        min_rounds = max(2, round_lo)
        deal_initiator = random.choice(["买家", "卖家"]) if mode["outcome"] == "deal" else "无"

    return {
        "mode": mode,
        "zone_profile": zone_profile,
        "zone_profile_name": scenario.metadata.get("zone_profile_name", zone_profile),
        "gap_ratio_percent": round(scenario.gap_ratio * 100, 1),
        "outcome": mode["outcome"],
        "min_rounds": min_rounds,
        "target_rounds": target_rounds,
        "deal_initiator": deal_initiator,
    }


def build_generation_prompt(scenario: NegotiationScenario, generation_spec: Dict) -> str:
    """为一个场景构造带模式控制的 LLM 对话生成 prompt"""
    mode = generation_spec["mode"]

    return DIALOGUE_GENERATION_PROMPT.format(
        item_name=scenario.item_name,
        item_description=scenario.item_description,
        market_ref_price=int(scenario.market_ref_price),
        buyer_budget=int(scenario.buyer_budget),
        seller_cost=int(scenario.seller_cost),
        max_rounds=scenario.max_rounds,
        zone_profile_name=generation_spec["zone_profile_name"],
        gap_ratio_percent=generation_spec["gap_ratio_percent"],
        mode_id=mode["id"],
        mode_name=mode["name"],
        mode_description=mode["name"],
        seller_strategy=mode["seller_strategy"],
        buyer_strategy=mode["buyer_strategy"],
        pace_instruction=mode["pace_instruction"],
        concession_instruction=mode["concession_instruction"],
        extra_constraints=mode["extra_constraints"],
        min_rounds=generation_spec["min_rounds"],
        target_rounds=generation_spec["target_rounds"],
        outcome=generation_spec["outcome"],
        deal_initiator=generation_spec["deal_initiator"],
    )


def parse_llm_response(response_text: str) -> Optional[Dict]:
    """
    从 LLM 返回内容中解析 JSON 对话。
    兼容 LLM 可能在 JSON 外围加上 ```json ... ``` 的情况。
    """
    # 尝试提取 ```json ... ``` 代码块
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = response_text.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # 基本校验
    if "dialogue" not in data or not isinstance(data["dialogue"], list):
        return None
    if len(data["dialogue"]) < 2:
        return None

    return data


def extract_offer(utterance: str) -> Optional[int]:
    """提取报价整数"""
    match = PRICE_PATTERN.search(utterance)
    return int(match.group(1)) if match else None


def extract_deal_price(utterance: str) -> Optional[float]:
    """提取成交价格"""
    match = DEAL_PATTERN.search(utterance)
    return float(match.group(1)) if match else None


def has_private_info_leak(utterance: str) -> bool:
    """粗略检测是否直接泄露底线型私密信息"""
    return any(pattern.search(utterance) for pattern in PRIVATE_INFO_PATTERNS)


def has_constant_concession_pattern(prices: List[int], role: str) -> bool:
    """避免机械式固定金额加价/降价"""
    if len(prices) < 3:
        return False

    if role == "seller":
        deltas = [prices[i] - prices[i + 1] for i in range(len(prices) - 1)]
    else:
        deltas = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]

    positive_deltas = [delta for delta in deltas if delta > 0]
    return len(positive_deltas) >= 2 and len(set(positive_deltas)) == 1


def validate_dialogue(
    dialogue: List[Dict],
    scenario: NegotiationScenario,
    expected_outcome: str,
) -> bool:
    """校验生成的对话是否格式合规、终局一致，并具备合理的让步规律"""
    if not dialogue or dialogue[0].get("role") != "seller":
        return False

    seller_prices = []
    buyer_prices = []

    for idx, turn in enumerate(dialogue):
        if not all(k in turn for k in ("round", "role", "utterance")):
            return False

        role = turn["role"]
        utterance = turn["utterance"]
        expected_role = "seller" if idx % 2 == 0 else "buyer"
        expected_round = idx // 2 + 1

        if role not in ("buyer", "seller"):
            return False
        if role != expected_role or turn["round"] != expected_round:
            return False
        if has_private_info_leak(utterance):
            return False

        has_offer = bool(PRICE_PATTERN.search(utterance))
        has_deal = "<deal>" in utterance
        has_walk = "<walkaway>" in utterance
        if not (has_offer or has_deal or has_walk):
            return False

        if idx < len(dialogue) - 1 and (has_deal or has_walk):
            return False

        price = extract_offer(utterance)
        if price is not None:
            if role == "seller":
                seller_prices.append(price)
            else:
                buyer_prices.append(price)

    if seller_prices != sorted(seller_prices, reverse=True):
        return False
    if buyer_prices != sorted(buyer_prices):
        return False
    if has_constant_concession_pattern(seller_prices, "seller"):
        return False
    if has_constant_concession_pattern(buyer_prices, "buyer"):
        return False

    last_utt = dialogue[-1]["utterance"]
    if expected_outcome == "deal":
        deal_price = extract_deal_price(last_utt)
        if deal_price is None:
            return False
        if not (scenario.seller_cost <= deal_price <= scenario.buyer_budget):
            return False
    elif expected_outcome == "walkaway":
        if "<walkaway>" not in last_utt:
            return False
    elif expected_outcome == "timeout":
        if "<deal>" in last_utt or "<walkaway>" in last_utt:
            return False
    else:
        return False

    return True


# ============================================================
# 场景加载
# ============================================================

def load_scenarios(path: str) -> List[NegotiationScenario]:
    scenarios = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(NegotiationScenario.from_dict(json.loads(line)))
    return scenarios


# ============================================================
# 单条对话生成流程（供线程池调用）
# ============================================================

def generate_one_dialogue(
    llm: LLMClient,
    scenario: NegotiationScenario,
    builder: PromptBuilder,
    mode_profiles: List[Dict],
    max_retries: int = 3,
    temperature: float = 0.8,
    max_output_tokens: int = 3072,
    retry_max_output_tokens: int = 6144,
    request_timeout: float = 120.0,
    retry_base_delay: float = 2.0,
    retry_max_delay: float = 20.0,
    retry_jitter: float = 0.25,
) -> Optional[List[Dict]]:
    """
    调用 LLM 为一个场景生成对话，并转为 SFT 样本列表。
    失败时自动重试。
    """
    generation_spec = select_generation_mode(scenario, mode_profiles)
    prompt = build_generation_prompt(scenario, generation_spec)
    current_max_tokens = max_output_tokens

    for attempt in range(max_retries):
        current_temperature = max(0.55, temperature - attempt * 0.05)
        try:
            response_text = llm.generate(
                prompt,
                temperature=current_temperature,
                max_tokens=current_max_tokens,
                timeout=request_timeout,
            )
            parsed = parse_llm_response(response_text)

            if parsed is None:
                raise RetryableGenerationError("模型返回无法解析为 JSON")

            if parsed.get("terminated_reason") != generation_spec["outcome"]:
                raise RetryableGenerationError(
                    f"终局不匹配，expected={generation_spec['outcome']}, got={parsed.get('terminated_reason')}"
                )

            dialogue = parsed["dialogue"]
            if not validate_dialogue(dialogue, scenario, generation_spec["outcome"]):
                raise RetryableGenerationError("对话未通过格式或策略校验")

            # 提取 SFT 样本
            samples = builder.extract_all_sft_samples(scenario, dialogue)
            for sample in samples:
                sample.setdefault("metadata", {}).update({
                    "dialogue_mode": generation_spec["mode"]["id"],
                    "dialogue_mode_name": generation_spec["mode"]["name"],
                    "expected_outcome": generation_spec["outcome"],
                    "zone_profile": generation_spec["zone_profile"],
                    "zone_profile_name": generation_spec["zone_profile_name"],
                    "gap_ratio_percent": generation_spec["gap_ratio_percent"],
                })
            return samples

        except NonRetryableGenerationError as e:
            log_progress(
                f"  ✗ 不可重试错误 (attempt {attempt+1}/{max_retries}), "
                f"scenario={scenario.scenario_id}, mode={generation_spec['mode']['id']}: {e}"
            )
            return None

        except RetryableGenerationError as e:
            if e.reason == "length" and current_max_tokens < retry_max_output_tokens:
                next_max_tokens = min(retry_max_output_tokens, int(current_max_tokens * 1.5))
                next_max_tokens = max(next_max_tokens, current_max_tokens + 512)
                current_max_tokens = next_max_tokens

            if attempt + 1 >= max_retries:
                log_progress(
                    f"  ✗ 重试耗尽 (attempt {attempt+1}/{max_retries}), "
                    f"scenario={scenario.scenario_id}, mode={generation_spec['mode']['id']}: {e}"
                )
                return None

            delay = compute_retry_delay(
                attempt_idx=attempt,
                base_delay=retry_base_delay,
                max_delay=retry_max_delay,
                jitter_ratio=retry_jitter,
            )
            log_progress(
                f"  ⚠ 可重试错误 (attempt {attempt+1}/{max_retries}), "
                f"scenario={scenario.scenario_id}, mode={generation_spec['mode']['id']}, "
                f"temp={current_temperature:.2f}, max_tokens={current_max_tokens}, "
                f"wait={delay:.1f}s: {e}"
            )
            time.sleep(delay)

        except Exception as e:
            if attempt + 1 >= max_retries:
                log_progress(
                    f"  ✗ 未知错误且重试耗尽 (attempt {attempt+1}/{max_retries}), "
                    f"scenario={scenario.scenario_id}, mode={generation_spec['mode']['id']}: {e}"
                )
                return None

            delay = compute_retry_delay(
                attempt_idx=attempt,
                base_delay=retry_base_delay,
                max_delay=retry_max_delay,
                jitter_ratio=retry_jitter,
            )
            log_progress(
                f"  ⚠ 未知错误，准备重试 (attempt {attempt+1}/{max_retries}), "
                f"scenario={scenario.scenario_id}, mode={generation_spec['mode']['id']}, "
                f"wait={delay:.1f}s: {e}"
            )
            time.sleep(delay)

    return None


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="调用 Ark Responses API 生成 SFT 训练数据")
    parser.add_argument("--scenarios", type=str, default=str(ROOT / "data" / "scenarios" / "train.jsonl"))
    parser.add_argument("--output", type=str, default=str(ROOT / "data" / "sft" / "sft_dialogues.jsonl"))
    parser.add_argument(
        "--mode-config",
        type=str,
        default=str(ROOT / "configs" / "generation_profiles.yaml"),
        help="对话模式配置 YAML 路径",
    )
    parser.add_argument("--num-dialogues", type=int, default=1000, help="要生成的对话数量")
    parser.add_argument("--model", type=str, default="ep-20260309141351-27swm", help="方舟推理接入点 ID")
    parser.add_argument("--api-key", type=str, default=None, help="Ark API Key（默认读取环境变量 ARK_API_KEY）")
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://ark-cn-beijing.bytedance.net/api/v3",
        help="Ark Responses API base URL",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-workers", type=int, default=8, help="并发线程数")
    parser.add_argument("--max-retries", type=int, default=4, help="单条对话最大重试次数")
    parser.add_argument("--max-output-tokens", type=int, default=3072, help="单次生成的初始最大输出 token 数")
    parser.add_argument(
        "--retry-max-output-tokens",
        type=int,
        default=6144,
        help="遇到 length 截断时，重试允许扩展到的最大输出 token 数",
    )
    parser.add_argument("--request-timeout", type=float, default=120.0, help="单次 API 请求超时时间（秒）")
    parser.add_argument("--retry-base-delay", type=float, default=2.0, help="重试基础退避时间（秒）")
    parser.add_argument("--retry-max-delay", type=float, default=20.0, help="重试最大退避时间（秒）")
    parser.add_argument("--retry-jitter", type=float, default=0.25, help="重试抖动比例，避免并发雪崩")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载场景
    scenarios = load_scenarios(args.scenarios)
    mode_profiles = load_dialogue_modes(args.mode_config)
    print(f"加载了 {len(scenarios)} 个场景")
    print(f"加载了 {len(mode_profiles)} 种对话模式")

    # 随机抽样
    if len(scenarios) >= args.num_dialogues:
        selected = random.sample(scenarios, args.num_dialogues)
    else:
        selected = [random.choice(scenarios) for _ in range(args.num_dialogues)]
    print(f"将为 {len(selected)} 个场景生成对话")

    # 初始化
    llm = LLMClient(model=args.model, api_key=args.api_key, base_url=args.base_url)
    builder = PromptBuilder()

    # 并发生成
    all_samples = []
    failed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                generate_one_dialogue,
                llm,
                scenario,
                builder,
                mode_profiles,
                args.max_retries,
                args.temperature,
                args.max_output_tokens,
                args.retry_max_output_tokens,
                args.request_timeout,
                args.retry_base_delay,
                args.retry_max_delay,
                args.retry_jitter,
            ): i
            for i, scenario in enumerate(selected)
        }

        with tqdm(
            total=len(futures),
            desc="生成 SFT 对话",
            unit="dialogue",
            dynamic_ncols=True,
        ) as progress_bar:
            for future in as_completed(futures):
                samples = future.result()
                if samples:
                    all_samples.extend(samples)
                else:
                    failed += 1

                progress_bar.update(1)
                progress_bar.set_postfix(samples=len(all_samples), failed=failed)

    # 打乱顺序
    random.shuffle(all_samples)

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(
            all_samples,
            desc="保存 SFT 样本",
            unit="sample",
            dynamic_ncols=True,
            disable=not all_samples,
        ):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n=== SFT 数据生成完成 ===")
    print(f"目标对话数:      {len(selected)}")
    print(f"失败数:          {failed}")
    print(f"SFT 样本总数:    {len(all_samples)}")
    print(f"保存到:          {output_path}")

    # 打印样例
    if all_samples:
        print(f"\n=== 样例展示 ===")
        sample = all_samples[0]
        print(f"元信息: {sample.get('metadata', {})}")
        for msg in sample["messages"]:
            role_tag = msg["role"].upper()
            content_preview = msg["content"][:150] + ("..." if len(msg["content"]) > 150 else "")
            print(f"  [{role_tag}] {content_preview}")


if __name__ == "__main__":
    main()
