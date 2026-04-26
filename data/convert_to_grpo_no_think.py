"""
将 LlamaFactory SFT 格式数据转换为 GRPO NegotiationScenario 格式

SFT 格式（每行）：
  {
    "messages": [{"role": "system", "content": "..."}, ...],
    "metadata": {"scenario_id": "train_00207", "role": "buyer", ...}
  }

输出格式（每行）：
  {
    "scenario_id": "train_00207",
    "item_name": "二手小米 14 Pro",
    "item_description": "512GB，白色，使用两个月，几乎全新带发票",
    "buyer_budget": 4192.0,
    "seller_cost": 3100.0,
    "market_ref_price": 3827.0,
    "max_rounds": 10
  }

转换逻辑：
1. 解析 system prompt，提取角色私密信息 + 公共信息
2. 按 scenario_id 分组，buyer 条目提供 buyer_budget，seller 条目提供 seller_cost
3. 同一 scenario_id 需要同时有 buyer 和 seller 才能输出完整 scenario
4. 若某 scenario_id 只有单角色，写入 incomplete 文件供排查
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional


# ─── 清除 <think>...</think> ────────────────────────────────────

def strip_think(text: str) -> str:
    """删除 <think>...</think>（含换行），并清理残留空白"""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def strip_think_from_messages(messages: list) -> list:
    """对 messages 中所有 assistant 消息的 content 去除 think 块"""
    result = []
    for m in messages:
        if m.get("role") == "assistant" and "<think>" in m.get("content", ""):
            m = dict(m, content=strip_think(m["content"]))
        result.append(m)
    return result


# ─── 解析 system prompt ──────────────────────────────────────────

def _find_system_content(messages: list) -> Optional[str]:
    for m in messages:
        if m.get("role") == "system":
            return m.get("content", "")
    return None


def parse_buyer_system(content: str) -> dict:
    """从 buyer 的 system prompt 提取信息"""
    result = {}

    # item_name：正在和卖家谈判购买"XXX"
    m = re.search(r'谈判购买[""「](.+?)[""」]', content)
    if m:
        result["item_name"] = m.group(1).strip()

    # buyer_budget：你的最高预算：XXX 元
    m = re.search(r'最高预算[：:]\s*([\d.]+)\s*元', content)
    if m:
        result["buyer_budget"] = float(m.group(1))

    # item_description：商品描述：XXX
    m = re.search(r'商品描述[：:]\s*(.+)', content)
    if m:
        result["item_description"] = m.group(1).strip()

    # market_ref_price：市场参考价：XXX 元
    m = re.search(r'市场参考价[：:]\s*([\d.]+)\s*元', content)
    if m:
        result["market_ref_price"] = float(m.group(1))

    return result


def parse_seller_system(content: str) -> dict:
    """从 seller 的 system prompt 提取信息"""
    result = {}

    # item_name：正在和买家谈判出售"XXX"
    m = re.search(r'谈判出售[""「](.+?)[""」]', content)
    if m:
        result["item_name"] = m.group(1).strip()

    # seller_cost：你的最低售价：XXX 元
    m = re.search(r'最低售价[：:]\s*([\d.]+)\s*元', content)
    if m:
        result["seller_cost"] = float(m.group(1))

    # item_description：商品描述：XXX
    m = re.search(r'商品描述[：:]\s*(.+)', content)
    if m:
        result["item_description"] = m.group(1).strip()

    # market_ref_price：市场参考价：XXX 元
    m = re.search(r'市场参考价[：:]\s*([\d.]+)\s*元', content)
    if m:
        result["market_ref_price"] = float(m.group(1))

    return result


# ─── 主转换逻辑 ──────────────────────────────────────────────────

def convert(input_path: str, output_path: str, incomplete_path: Optional[str] = None):
    # scenario_id -> {"buyer": {...}, "seller": {...}}
    groups = defaultdict(dict)

    total = 0
    parse_errors = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {lineno}: JSON parse error: {e}", file=sys.stderr)
                parse_errors += 1
                continue

            total += 1
            meta = obj.get("metadata", {})
            scenario_id = str(meta.get("scenario_id", f"unknown_{lineno}"))
            role = meta.get("role", "")
            messages = strip_think_from_messages(obj.get("messages", []))
            system_content = _find_system_content(messages)

            if not system_content:
                print(f"[WARN] line {lineno} ({scenario_id}): no system message", file=sys.stderr)
                parse_errors += 1
                continue

            if role == "buyer":
                parsed = parse_buyer_system(system_content)
                if "buyer_budget" not in parsed:
                    print(f"[WARN] line {lineno} ({scenario_id}): buyer_budget not found", file=sys.stderr)
                    parse_errors += 1
                groups[scenario_id]["buyer"] = parsed

            elif role == "seller":
                parsed = parse_seller_system(system_content)
                if "seller_cost" not in parsed:
                    print(f"[WARN] line {lineno} ({scenario_id}): seller_cost not found", file=sys.stderr)
                    parse_errors += 1
                groups[scenario_id]["seller"] = parsed

            else:
                print(f"[WARN] line {lineno} ({scenario_id}): unknown role '{role}'", file=sys.stderr)

    # 合并 buyer + seller，输出 scenario
    out_lines = []
    incomplete_lines = []

    for scenario_id, roles in groups.items():
        buyer_info = roles.get("buyer", {})
        seller_info = roles.get("seller", {})

        has_budget = "buyer_budget" in buyer_info
        has_cost = "seller_cost" in seller_info

        if not has_budget or not has_cost:
            incomplete_lines.append({
                "scenario_id": scenario_id,
                "missing": [] + (["buyer_budget"] if not has_budget else [])
                               + (["seller_cost"] if not has_cost else []),
                "buyer_info": buyer_info,
                "seller_info": seller_info,
            })
            continue

        # 公共信息优先取 buyer（item_name/item_description/market_ref_price 一致）
        shared = {k: v for k, v in buyer_info.items() if k != "buyer_budget"}
        # seller 可能补充 item_name/description（若 buyer 里没有）
        for k in ("item_name", "item_description", "market_ref_price"):
            if k not in shared and k in seller_info:
                shared[k] = seller_info[k]

        scenario = {
            "scenario_id": scenario_id,
            "item_name": shared.get("item_name", ""),
            "item_description": shared.get("item_description", ""),
            "buyer_budget": buyer_info["buyer_budget"],
            "seller_cost": seller_info["seller_cost"],
            "market_ref_price": shared.get("market_ref_price", 0.0),
            "max_rounds": 10,
        }
        out_lines.append(scenario)

    # 写输出
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in out_lines:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if incomplete_lines and incomplete_path:
        with open(incomplete_path, "w", encoding="utf-8") as f:
            for s in incomplete_lines:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"输入条数     : {total}")
    print(f"解析警告     : {parse_errors}")
    print(f"场景分组数   : {len(groups)}")
    print(f"完整场景输出 : {len(out_lines)}  -> {output_path}")
    print(f"不完整场景   : {len(incomplete_lines)}"
          + (f"  -> {incomplete_path}" if incomplete_path else " (未保存)"))


# ─── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="将 LlamaFactory SFT jsonl 转换为 GRPO NegotiationScenario jsonl"
    )
    parser.add_argument("--input", default="/root/autodl-tmp/Final_project/data/sft_1k_think_content/train_800.jsonl",      help="输入 SFT jsonl 路径")
    parser.add_argument("--output", default="/root/autodl-tmp/Final_project/grpo/data/training_grpo_no_think.jsonl",help="输出 scenario jsonl 路径")
    parser.add_argument("--incomplete", default=None,  help="不完整场景保存路径（可选）")
    args = parser.parse_args()

    convert(args.input, args.output, args.incomplete)


if __name__ == "__main__":
    main()
