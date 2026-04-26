"""
将 train_800.jsonl 转换为 GRPO ScenarioDataset 所需的格式。

输入格式（train_800.jsonl）：
  每行包含 messages + metadata，同一 scenario_id 会有 buyer/seller 两种角色的记录。

输出格式（scenarios.jsonl）：
  每行是一个 NegotiationScenario dict，字段：
    scenario_id, item_name, item_description,
    buyer_budget, seller_cost, market_ref_price,
    max_rounds, metadata

用法：
    python convert_to_scenarios.py \
        --input  train_800.jsonl \
        --output scenarios_grpo.jsonl \
        [--max_rounds 10]
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


# ----------------------------------------------------------------
# 从 system prompt 提取字段
# ----------------------------------------------------------------

def extract_item_name(content: str) -> Optional[str]:
    """匹配 出售/购买"商品名" 或 出售/购买「商品名」"""
    m = re.search(r'(?:出售|购买)["""「](.+?)["""」]', content)
    return m.group(1).strip() if m else None


def extract_item_description(content: str) -> Optional[str]:
    m = re.search(r'商品描述[:：]\s*(.+)', content)
    return m.group(1).strip() if m else None


def extract_market_ref_price(content: str) -> Optional[float]:
    m = re.search(r'市场参考价[:：]\s*([\d.]+)\s*元', content)
    return float(m.group(1)) if m else None


def extract_seller_cost(content: str) -> Optional[float]:
    """从卖家 system prompt 提取最低售价"""
    m = re.search(r'最低售价[:：]\s*([\d.]+)\s*元', content)
    return float(m.group(1)) if m else None


def extract_buyer_budget(content: str) -> Optional[float]:
    """从买家 system prompt 提取最高预算"""
    m = re.search(r'最高预算[:：]\s*([\d.]+)\s*元', content)
    return float(m.group(1)) if m else None


def parse_system_prompt(content: str) -> Dict:
    return {
        "item_name":        extract_item_name(content),
        "item_description": extract_item_description(content),
        "market_ref_price": extract_market_ref_price(content),
        "seller_cost":      extract_seller_cost(content),
        "buyer_budget":     extract_buyer_budget(content),
    }


# ----------------------------------------------------------------
# 主转换逻辑
# ----------------------------------------------------------------

def convert(input_path: str, output_path: str, max_rounds: int = 10) -> None:
    # 按 scenario_id 收集 buyer/seller 两侧的信息
    # 只需要提取一次 system prompt（取最早遇到的那条）
    buyer_info: Dict[str, Dict]  = {}   # scenario_id -> parsed fields
    seller_info: Dict[str, Dict] = {}
    meta_info: Dict[str, Dict]   = {}   # scenario_id -> metadata（来自任意一条）

    with open(input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] line {lineno}: JSON decode error: {e}", file=sys.stderr)
                continue

            meta     = obj.get("metadata", {})
            sid      = meta.get("scenario_id", "")
            role     = meta.get("role", "")
            messages = obj.get("messages", [])

            if not sid or not messages:
                continue

            # 取 system prompt（第一条 role==system 的消息）
            sys_content = next(
                (m["content"] for m in messages if m.get("role") == "system"), None
            )
            if sys_content is None:
                continue

            parsed = parse_system_prompt(sys_content)

            if role == "buyer" and sid not in buyer_info:
                buyer_info[sid] = parsed
            elif role == "seller" and sid not in seller_info:
                seller_info[sid] = parsed

            # 保存 metadata（用 scenario_id、zone_profile 等通用字段）
            if sid not in meta_info:
                meta_info[sid] = {
                    k: v for k, v in meta.items()
                    if k not in ("role", "round", "turn_index",
                                 "dialogue_mode", "dialogue_mode_name",
                                 "expected_outcome")
                }

    # 合并：以同时拥有 buyer + seller 信息的 scenario_id 为准
    all_ids = set(buyer_info) | set(seller_info)
    missing_buyer  = [sid for sid in all_ids if sid not in buyer_info]
    missing_seller = [sid for sid in all_ids if sid not in seller_info]
    if missing_buyer:
        print(f"[warn] {len(missing_buyer)} scenario(s) missing buyer info, will use None for buyer_budget",
              file=sys.stderr)
    if missing_seller:
        print(f"[warn] {len(missing_seller)} scenario(s) missing seller info, will use None for seller_cost",
              file=sys.stderr)

    written = 0
    skipped = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for sid in sorted(all_ids):
            b = buyer_info.get(sid, {})
            s = seller_info.get(sid, {})

            # 优先从 seller prompt 取公共字段（seller prompt 里也有市场参考价）
            item_name        = s.get("item_name")        or b.get("item_name")
            item_description = s.get("item_description") or b.get("item_description")
            market_ref_price = s.get("market_ref_price") or b.get("market_ref_price")
            buyer_budget     = b.get("buyer_budget")
            seller_cost      = s.get("seller_cost")

            # 必填字段缺失则跳过
            if buyer_budget is None or seller_cost is None:
                skipped += 1
                continue
            if item_name is None:
                skipped += 1
                continue

            scenario = {
                "scenario_id":     sid,
                "item_name":       item_name,
                "item_description": item_description or "",
                "buyer_budget":    buyer_budget,
                "seller_cost":     seller_cost,
                "market_ref_price": market_ref_price or 0.0,
                "max_rounds":      max_rounds,
                "metadata":        meta_info.get(sid, {}),
            }
            fout.write(json.dumps(scenario, ensure_ascii=False) + "\n")
            written += 1

    print(f"[done] written={written}  skipped={skipped}  output={output_path}")


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert train_800.jsonl to GRPO scenario format")
    parser.add_argument("--input",  default="/root/autodl-tmp/Final_project/data/sft_1k_think_content/val_100.jsonl", help="输入文件路径")
    parser.add_argument("--output", default="/root/autodl-tmp/Final_project/grpo/data/val_grpo.jsonl", help="输出文件路径")
    parser.add_argument("--max_rounds", type=int, default=10, help="每场最大轮数（默认10）")
    args = parser.parse_args()

    convert(args.input, args.output, args.max_rounds)


if __name__ == "__main__":
    main()
