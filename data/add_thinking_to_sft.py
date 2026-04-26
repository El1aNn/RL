"""
Add short strategy-style thinking annotations to SFT JSONL files.

The script annotates only the final assistant message in each training sample.
By default it calls the Ark Responses API to produce a concise thinking field.
Use `--thinking-source heuristic` for a local deterministic fallback.

Examples:
    python data/add_thinking_to_sft.py \
        --input data/sft_1k/train_800.jsonl \
        --output data/sft_1k_thinking/train_800.jsonl \
        --max-workers 10

    python data/add_thinking_to_sft.py \
        --input-dir data/sft_1k \
        --output-dir data/sft_1k_thinking \
        --placement message_field \
        --max-workers 10

    python data/add_thinking_to_sft.py \
        --input data/sft_1k/train_800.jsonl \
        --output data/sft_1k_think_content/train_800.jsonl \
        --placement content_prefix \
        --thinking-tag think
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments.
    tqdm = None


PRICE_PATTERN = re.compile(r"\[报价[：:]\s*(\d+)\]")
DEAL_PATTERN = re.compile(r"<deal>\s*(\d+(?:\.\d+)?)\s*</deal>")
WALKAWAY_PATTERN = re.compile(r"<walkaway>")
ROUND_ROLE_PATTERN = re.compile(r"【第(\d+)轮-(卖家|买家)】")
SELLER_MIN_PATTERN = re.compile(r"最低售价[：:]\s*(\d+)")
BUYER_BUDGET_PATTERN = re.compile(r"最高预算[：:]\s*(\d+)")
MARKET_PRICE_PATTERN = re.compile(r"市场参考价[：:]\s*(\d+)")
ITEM_PATTERN = re.compile(r"正在和(?:买家|卖家)谈判(?:出售|购买)\"([^\"]+)\"")


ZONE_HINTS = {
    "near_zero_space": "谈判空间很小，措辞要谨慎，重点是守住约束并避免过早承诺。",
    "narrow_space": "谈判空间偏窄，适合小步试探，用诚意或稀缺性推动对方靠近。",
    "balanced_space": "谈判空间适中，可以在价值表达和适度让步之间保持平衡。",
    "wide_space": "谈判空间较宽，可以用更明显的锚定和让步制造推进感。",
}

MODE_HINTS = {
    "early_deal_anchor": "本轮服务于快速成交，报价应形成清晰锚点并尽快收敛。",
    "patient_balanced_deal": "本轮保持耐心试探，让步要显得有理由而不是机械变化。",
    "hard_bargain_late_deal": "本轮保留强硬姿态，等僵持充分后再释放关键让步。",
    "near_limit_deal": "本轮围绕很小价差推进，避免让步过大导致越过自身约束。",
    "buyer_walkaway_with_zone": "本轮允许谈崩，重点是让退出理由自然可信。",
    "seller_walkaway_protect_margin": "本轮优先保护利润和立场，不为成交牺牲约束。",
    "timeout_deadlock": "本轮维持僵持，不主动给出成交或退出信号。",
    "bluff_and_reversal": "本轮体现策略变化，用虚张声势或突然松动推动反转。",
}


@dataclass
class FileStats:
    read: int = 0
    written: int = 0
    annotated: int = 0
    skipped_existing: int = 0
    invalid: int = 0
    api_failed: int = 0


def progress(iterable: Iterable[Any], args: argparse.Namespace, **kwargs: Any) -> Iterable[Any]:
    """Wrap an iterable in tqdm unless disabled or unavailable."""
    if args.disable_tqdm or tqdm is None:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


class ArkThinkingClient:
    """Small wrapper around the Ark Responses API."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str],
        base_url: str,
        temperature: float,
        max_output_tokens: int,
        request_timeout: float,
    ) -> None:
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError as exc:
            raise RuntimeError(
                "volcenginesdkarkruntime is required for --thinking-source api. "
                "Run inside the project production environment or use --thinking-source heuristic."
            ) from exc

        resolved_api_key = api_key or os.getenv("ARK_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("ARK_API_KEY is not set. Export it or pass --api-key.")

        self.client = Ark(base_url=base_url, api_key=resolved_api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.request_timeout = request_timeout

    @staticmethod
    def _extract_text(response: Any) -> str:
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text" and getattr(content, "text", None):
                    return content.text.strip()

        status = getattr(response, "status", "unknown")
        raise RuntimeError(f"Ark Responses API returned no output_text, status={status}")

    def generate_text(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            timeout=self.request_timeout,
        )
        return self._extract_text(response)


def load_jsonl(path: Path, limit: Optional[int] = None) -> Iterable[Tuple[int, Dict[str, Any]]]:
    rows_seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rows_seen += 1
            if limit is not None and rows_seen > limit:
                break
            try:
                yield line_no, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def target_assistant_message(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    last = messages[-1]
    if isinstance(last, dict) and last.get("role") == "assistant":
        return last

    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message
    return None


def parse_system_info(sample: Dict[str, Any]) -> Dict[str, Any]:
    messages = sample.get("messages") or []
    system_content = ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            system_content = str(message.get("content", ""))
            break

    info: Dict[str, Any] = {}
    if "精明的卖家" in system_content:
        info["role"] = "seller"
    elif "精明的买家" in system_content:
        info["role"] = "buyer"

    for key, pattern in [
        ("seller_min", SELLER_MIN_PATTERN),
        ("buyer_budget", BUYER_BUDGET_PATTERN),
        ("market_price", MARKET_PRICE_PATTERN),
    ]:
        match = pattern.search(system_content)
        if match:
            info[key] = int(match.group(1))

    item_match = ITEM_PATTERN.search(system_content)
    if item_match:
        info["item"] = item_match.group(1)

    return info


def infer_role(sample: Dict[str, Any], content: str, system_info: Dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    role = metadata.get("role") or system_info.get("role")
    if role in {"seller", "buyer"}:
        return role

    match = ROUND_ROLE_PATTERN.search(content)
    if match:
        return "seller" if match.group(2) == "卖家" else "buyer"

    return "assistant"


def infer_round(sample: Dict[str, Any], content: str) -> Optional[int]:
    metadata = sample.get("metadata") or {}
    round_value = metadata.get("round")
    if isinstance(round_value, int):
        return round_value
    if isinstance(round_value, str) and round_value.isdigit():
        return int(round_value)

    match = ROUND_ROLE_PATTERN.search(content)
    if match:
        return int(match.group(1))
    return None


def infer_action(content: str) -> Tuple[str, Optional[int]]:
    deal_match = DEAL_PATTERN.search(content)
    if deal_match:
        return "deal", int(float(deal_match.group(1)))

    if WALKAWAY_PATTERN.search(content):
        return "walkaway", None

    quote_matches = PRICE_PATTERN.findall(content)
    if quote_matches:
        return "quote", int(quote_matches[-1])

    return "respond", None


def build_strategy_thinking(sample: Dict[str, Any], target: Dict[str, Any]) -> str:
    content = str(target.get("content", ""))
    metadata = sample.get("metadata") or {}
    system_info = parse_system_info(sample)
    role = infer_role(sample, content, system_info)
    round_no = infer_round(sample, content)
    action, price = infer_action(content)

    zone_profile = metadata.get("zone_profile")
    mode_id = metadata.get("dialogue_mode")
    zone_hint = ZONE_HINTS.get(zone_profile, "根据当前报价差距控制节奏，既推进谈判也守住自身约束。")
    mode_hint = MODE_HINTS.get(mode_id)

    actor = "卖家" if role == "seller" else "买家" if role == "buyer" else "当前角色"
    round_hint = f"第{round_no}轮" if round_no is not None else "当前轮"

    if action == "deal":
        if role == "seller":
            core = f"{round_hint}策略：对方价格已经进入可接受区间，直接确认成交，减少继续拉扯导致买家流失。"
        elif role == "buyer":
            core = f"{round_hint}策略：当前价格已经满足购买目标，及时接受以锁定交易，避免卖家反悔或继续抬价。"
        else:
            core = f"{round_hint}策略：价格已经足够接近目标区间，选择成交比继续拉扯更稳妥。"
    elif action == "walkaway":
        if role == "seller":
            core = f"{round_hint}策略：对方仍未给出足够诚意，主动结束谈判以保护价格立场。"
        elif role == "buyer":
            core = f"{round_hint}策略：卖家让步不足，继续谈下去容易超出购买约束，因此选择退出。"
        else:
            core = f"{round_hint}策略：对方无法靠近目标区间，结束谈判比继续消耗更合理。"
    elif action == "quote":
        if role == "seller":
            if round_no == 1:
                core = f"{round_hint}策略：先用偏高但仍可解释的报价建立锚点，强调商品价值，并为后续让步留空间。"
            else:
                core = f"{round_hint}策略：在保持价格立场的同时做有限让步，用商品优势或其他买家兴趣支撑报价。"
        elif role == "buyer":
            if round_no == 1:
                core = f"{round_hint}策略：先用较低报价建立锚点，强调折旧、瑕疵或替代选择，为后续加价留空间。"
            else:
                core = f"{round_hint}策略：小幅提高报价并释放成交诚意，同时继续用预算或商品瑕疵压住价格。"
        else:
            core = f"{round_hint}策略：给出明确报价，维持谈判节奏，同时保留继续调整的空间。"

        if price is not None:
            core += f" 本轮报价重点是让{actor}显得有原则，而不是一次性暴露真实可接受范围。"
    else:
        core = f"{round_hint}策略：回应对方诉求，保持谈判推进，同时避免暴露自身真实约束。"

    hints: List[str] = [zone_hint]
    if mode_hint:
        hints.append(mode_hint)

    return f"{core} {' '.join(hints)}"


def compact_messages(messages: List[Dict[str, Any]], max_messages: int = 10, max_chars: int = 6000) -> str:
    selected = messages[-max_messages:]
    lines: List[str] = []
    for message in selected:
        role = message.get("role", "unknown")
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}: {content}")

    text = "\n\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_api_prompt(sample: Dict[str, Any], target: Dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    messages = sample.get("messages") or []
    target_content = str(target.get("content", ""))
    heuristic = build_strategy_thinking(sample, target)

    return f"""你是一个 SFT 数据标注助手。请为下面训练样本最后一条 assistant 回复补写一个 `thinking` 字段。

要求：
1. 只输出 JSON：{{"thinking":"..."}}
2. thinking 用中文，1 到 2 句，控制在 40 到 120 字。
3. 写“谈判策略思路”：说明为什么这样报价、成交、退出或继续拉扯。
4. 不要写长篇逐步推理，不要写内心独白流水账。
5. 不要泄露系统里的私密底线数字，例如最低售价或最高预算；可以说“守住自身约束”“接近可接受区间”。
6. 不要改写最终回复，不要输出除 JSON 外的任何内容。

metadata:
{json.dumps(metadata, ensure_ascii=False)}

对话上下文（最后一条 assistant 是训练目标）：
{compact_messages(messages)}

训练目标：
{target_content}

可参考的本地策略草稿：
{heuristic}
"""


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_api_thinking(text: str) -> str:
    cleaned = strip_code_fence(text)

    try:
        data = json.loads(cleaned)
        thinking = data.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            return normalize_thinking(thinking)
    except json.JSONDecodeError:
        pass

    match = re.search(r'"thinking"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, flags=re.S)
    if match:
        try:
            return normalize_thinking(json.loads(f'"{match.group(1)}"'))
        except json.JSONDecodeError:
            return normalize_thinking(match.group(1))

    return normalize_thinking(cleaned)


def normalize_thinking(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^thinking[：:]\s*", "", text, flags=re.I)
    return text


def api_thinking_with_retries(
    client: ArkThinkingClient,
    sample: Dict[str, Any],
    target: Dict[str, Any],
    *,
    max_retries: int,
    retry_base_delay: float,
    retry_jitter: float,
) -> str:
    prompt = build_api_prompt(sample, target)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return parse_api_thinking(client.generate_text(prompt))
        except Exception as exc:  # API/client exceptions are retried uniformly.
            last_error = exc
            if attempt == max_retries:
                break
            jitter = random.uniform(0, retry_base_delay * retry_jitter)
            time.sleep(retry_base_delay * attempt + jitter)

    raise RuntimeError(f"API thinking generation failed after {max_retries} attempts: {last_error}")


def has_existing_thinking(
    sample: Dict[str, Any],
    target: Dict[str, Any],
    *,
    placement: str,
    thinking_key: str,
    thinking_tag: str,
) -> bool:
    if placement == "message_field":
        return thinking_key in target
    if placement == "metadata_field":
        return thinking_key in (sample.get("metadata") or {})
    if placement == "content_prefix":
        return str(target.get("content", "")).lstrip().startswith(f"<{thinking_tag}>")
    raise ValueError(f"unknown placement: {placement}")


def set_thinking(
    sample: Dict[str, Any],
    target: Dict[str, Any],
    thinking: str,
    *,
    placement: str,
    thinking_key: str,
    thinking_tag: str,
) -> None:
    if placement == "message_field":
        target[thinking_key] = thinking
    elif placement == "metadata_field":
        sample.setdefault("metadata", {})[thinking_key] = thinking
    elif placement == "content_prefix":
        content = str(target.get("content", ""))
        tag = re.escape(thinking_tag)
        cleaned = re.sub(rf"^\s*<{tag}>.*?</{tag}>\s*", "", content, flags=re.S)
        target["content"] = f"<{thinking_tag}>{thinking}</{thinking_tag}>\n{cleaned}"
    else:
        raise ValueError(f"unknown placement: {placement}")


def annotate_rows(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    label: str,
) -> Tuple[List[Dict[str, Any]], FileStats]:
    stats = FileStats(read=len(rows), written=len(rows))
    jobs: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []

    for index, sample in enumerate(rows):
        target = target_assistant_message(sample)
        if target is None:
            stats.invalid += 1
            continue

        if has_existing_thinking(
            sample,
            target,
            placement=args.placement,
            thinking_key=args.thinking_key,
            thinking_tag=args.thinking_tag,
        ):
            if args.overwrite_existing:
                jobs.append((index, sample, target))
            else:
                stats.skipped_existing += 1
            continue

        jobs.append((index, sample, target))

    if args.dry_run:
        stats.annotated += len(jobs)
        return rows, stats

    if args.thinking_source == "heuristic":
        for _, sample, target in progress(
            jobs,
            args,
            total=len(jobs),
            desc=f"{label} heuristic",
            unit="sample",
        ):
            set_thinking(
                sample,
                target,
                build_strategy_thinking(sample, target),
                placement=args.placement,
                thinking_key=args.thinking_key,
                thinking_tag=args.thinking_tag,
            )
            stats.annotated += 1
        return rows, stats

    client = ArkThinkingClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        request_timeout=args.request_timeout,
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                api_thinking_with_retries,
                client,
                sample,
                target,
                max_retries=args.max_retries,
                retry_base_delay=args.retry_base_delay,
                retry_jitter=args.retry_jitter,
            ): (sample, target)
            for _, sample, target in jobs
        }

        completed = 0
        for future in progress(
            as_completed(futures),
            args,
            total=len(futures),
            desc=f"{label} API thinking",
            unit="sample",
        ):
            sample, target = futures[future]
            try:
                thinking = future.result()
            except Exception as exc:
                stats.api_failed += 1
                if args.no_fallback_to_heuristic:
                    print(f"API failed and fallback disabled: {exc}", file=sys.stderr)
                    continue
                thinking = build_strategy_thinking(sample, target)

            set_thinking(
                sample,
                target,
                thinking,
                placement=args.placement,
                thinking_key=args.thinking_key,
                thinking_tag=args.thinking_tag,
            )
            stats.annotated += 1
            completed += 1
            if args.disable_tqdm and args.progress_every and completed % args.progress_every == 0:
                print(f"  API thinking progress: {completed}/{len(jobs)}", flush=True)

    return rows, stats


def write_jsonl(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> FileStats:
    rows = [
        sample
        for _, sample in progress(
            load_jsonl(input_path, limit=args.limit),
            args,
            desc=f"{input_path.name} read",
            unit="row",
        )
    ]
    rows, stats = annotate_rows(rows, args, label=input_path.name)

    if args.dry_run:
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in progress(
            rows,
            args,
            total=len(rows),
            desc=f"{output_path.name} write",
            unit="row",
        ):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return stats


def input_output_pairs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    if args.input and args.input_dir:
        raise SystemExit("Use either --input or --input-dir, not both.")

    if args.input:
        input_path = args.input
        if args.in_place:
            return [(input_path, input_path)]
        if not args.output:
            raise SystemExit("--output is required unless --in-place is used.")
        return [(input_path, args.output)]

    if args.input_dir:
        paths = [path for path in sorted(args.input_dir.glob(args.glob)) if path.is_file()]
        if args.in_place:
            return [(path, path) for path in paths]
        if not args.output_dir:
            raise SystemExit("--output-dir is required unless --in-place is used.")
        return [(path, args.output_dir / path.name) for path in paths]

    raise SystemExit("Use --input or --input-dir.")


def process_pair(input_path: Path, output_path: Path, args: argparse.Namespace) -> FileStats:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if args.in_place and not args.dry_run:
        temp_path = input_path.with_suffix(input_path.suffix + ".tmp")
        stats = write_jsonl(input_path, temp_path, args)
        temp_path.replace(input_path)
        return stats

    return write_jsonl(input_path, output_path, args)


def main() -> int:
    parser = argparse.ArgumentParser(description="Add strategy-style thinking annotations to SFT JSONL files.")
    parser.add_argument("--input", type=Path, help="Single input JSONL file.")
    parser.add_argument("--output", type=Path, help="Single output JSONL file.")
    parser.add_argument("--input-dir", type=Path, help="Directory of input JSONL files.")
    parser.add_argument("--output-dir", type=Path, help="Directory for output JSONL files.")
    parser.add_argument("--glob", default="*.jsonl", help="Glob used with --input-dir. Default: *.jsonl")
    parser.add_argument(
        "--placement",
        choices=["message_field", "metadata_field", "content_prefix"],
        default="message_field",
        help="Where to store thinking. Default: message_field.",
    )
    parser.add_argument(
        "--thinking-tag",
        default="think",
        help="XML-like tag used by --placement content_prefix. Default: think.",
    )
    parser.add_argument(
        "--thinking-source",
        choices=["api", "heuristic"],
        default="api",
        help="Generate thinking with Ark API or local heuristic rules. Default: api.",
    )
    parser.add_argument("--thinking-key", default="thinking", help="Field name for message_field/metadata_field.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Replace existing thinking annotations.")
    parser.add_argument("--in-place", action="store_true", help="Rewrite input files in place via a temporary file.")
    parser.add_argument("--dry-run", action="store_true", help="Compute stats without writing output files or calling API.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N non-empty JSONL rows.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bars.")

    parser.add_argument("--model", type=str, default="ep-20260309141351-27swm", help="Ark model endpoint ID.")
    parser.add_argument("--api-key", type=str, default=None, help="Ark API key. Defaults to ARK_API_KEY.")
    parser.add_argument("--base-url", type=str, default="https://ark-cn-beijing.bytedance.net/api/v3")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--max-workers", type=int, default=10, help="Concurrent API calls when --thinking-source api.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=1.5)
    parser.add_argument("--retry-jitter", type=float, default=0.25)
    parser.add_argument(
        "--no-fallback-to-heuristic",
        action="store_true",
        help="Leave rows unannotated when API fails instead of using local heuristic thinking.",
    )
    parser.add_argument("--progress-every", type=int, default=50, help="Print API progress every N rows. 0 disables.")
    args = parser.parse_args()

    total = FileStats()
    pairs = input_output_pairs(args)
    if not pairs:
        print("No input files matched.", file=sys.stderr)
        return 1

    for input_path, output_path in progress(
        pairs,
        args,
        total=len(pairs),
        desc="files",
        unit="file",
    ):
        stats = process_pair(input_path, output_path, args)
        total.read += stats.read
        total.written += stats.written
        total.annotated += stats.annotated
        total.skipped_existing += stats.skipped_existing
        total.invalid += stats.invalid
        total.api_failed += stats.api_failed

        destination = input_path if args.in_place else output_path
        action = "dry-run" if args.dry_run else "wrote"
        print(
            f"{input_path} -> {destination}: {action}, "
            f"read={stats.read}, annotated={stats.annotated}, "
            f"skipped_existing={stats.skipped_existing}, invalid={stats.invalid}, "
            f"api_failed={stats.api_failed}",
            flush=True,
        )

    print(
        "TOTAL: "
        f"read={total.read}, written={total.written}, annotated={total.annotated}, "
        f"skipped_existing={total.skipped_existing}, invalid={total.invalid}, "
        f"api_failed={total.api_failed}",
        flush=True,
    )
    return 0 if total.invalid == 0 and (total.api_failed == 0 or not args.no_fallback_to_heuristic) else 2


if __name__ == "__main__":
    raise SystemExit(main())
