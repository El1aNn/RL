"""
场景生成脚本

从 configs/scenario_templates.yaml 读取商品模板，
通过规则随机采样生成训练/验证/测试场景集。
同时按谈判空间分层，保证存在接近无空间、狭窄空间、中等空间、宽松空间等不同难度。
"""
import json
import random
import argparse
import math
from pathlib import Path

from tqdm import tqdm
import yaml

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent


def load_templates(yaml_path: str) -> list:
    """加载商品模板 YAML"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["categories"]


def load_zone_profiles(yaml_path: str) -> list:
    """加载场景谈判空间分层配置"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["scenario_zone_profiles"]


def choose_weighted(items: list) -> dict:
    """按权重随机抽取一个配置项"""
    weights = [item.get("weight", 1.0) for item in items]
    return random.choices(items, weights=weights, k=1)[0]


def sample_buyer_budget(
    seller_cost: int,
    market_hi: int,
    zone_profile: dict,
) -> int:
    """
    按谈判空间分层采样买家预算。

    gap_ratio = (buyer_budget - seller_cost) / seller_cost
    """
    gap_lo, gap_hi = zone_profile["gap_ratio_range"]
    min_gap = max(1, math.ceil(seller_cost * 0.03))
    budget_floor = seller_cost + min_gap
    budget_cap = max(budget_floor, int(market_hi * 1.3))

    requested_lo = math.ceil(seller_cost * (1 + gap_lo))
    requested_hi = int(seller_cost * (1 + gap_hi))
    profile_lo = max(requested_lo, budget_floor)
    profile_hi = min(requested_hi, budget_cap)

    if profile_lo > profile_hi:
        # Low-price items or tight market caps can squeeze a profile into an
        # empty interval. Keep the sample as close as possible to the selected
        # difficulty instead of falling back to an unrelated wide interval.
        if budget_floor > requested_hi:
            profile_lo = profile_hi = min(budget_floor, budget_cap)
        else:
            profile_lo = profile_hi = budget_cap

    return random.randint(profile_lo, profile_hi)


def sample_scenario(
    item_template: dict,
    description: str,
    scenario_id: str,
    zone_profiles: list,
) -> dict:
    """
    从一个商品模板中采样一个合法的谈判场景。

    保证：buyer_budget > seller_cost（正的谈判空间）
    市场参考价取 cost 和 budget 之间的某个值。
    """
    cost_lo, cost_hi = item_template["cost_range"]
    market_lo, market_hi = item_template["market_price_range"]

    # 1. 采样卖家成本
    seller_cost = random.randint(cost_lo, cost_hi)
    # 2. 采样市场参考价（>= seller_cost）
    market_ref = random.randint(max(market_lo, seller_cost), market_hi)
    # 3. 采样谈判空间分层，并据此采样买家预算
    zone_profile = choose_weighted(zone_profiles)
    buyer_budget = sample_buyer_budget(seller_cost, market_hi, zone_profile)
    gap_ratio = round((buyer_budget - seller_cost) / seller_cost, 4)

    return {
        "scenario_id": scenario_id,
        "item_name": item_template["name"],
        "item_description": description,
        "buyer_budget": buyer_budget,
        "seller_cost": seller_cost,
        "market_ref_price": market_ref,
        "max_rounds": 10,
        "metadata": {
            "zone_profile": zone_profile["id"],
            "zone_profile_name": zone_profile["name"],
            "gap_ratio": gap_ratio,
        },
    }


def generate_scenarios(
    templates: list,
    zone_profiles: list,
    num_scenarios: int,
    id_prefix: str = "scn",
    seed: int = 42,
) -> list:
    """生成指定数量的随机场景"""
    random.seed(seed)

    # 展平所有 (item_template, description) 对
    all_items = []
    for category in templates:
        for item in category["items"]:
            for desc in item["descriptions"]:
                all_items.append((item, desc))

    scenarios = []
    for i in tqdm(
        range(num_scenarios),
        desc=f"生成 {id_prefix}",
        unit="scenario",
        dynamic_ncols=True,
        disable=num_scenarios == 0,
    ):
        item_template, description = random.choice(all_items)
        scenario_id = f"{id_prefix}_{i:05d}"
        scenario = sample_scenario(item_template, description, scenario_id, zone_profiles)
        scenarios.append(scenario)

    return scenarios


def save_jsonl(data: list, path: str):
    """保存为 JSONL 格式"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in tqdm(
            data,
            desc=f"保存 {Path(path).name}",
            unit="item",
            dynamic_ncols=True,
            disable=not data,
        ):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  → 已保存 {len(data)} 条到 {path}")


def main():
    parser = argparse.ArgumentParser(description="生成谈判场景数据集")
    parser.add_argument(
        "--templates",
        type=str,
        default=str(ROOT / "configs" / "scenario_templates.yaml"),
        help="商品模板 YAML 路径",
    )
    parser.add_argument(
        "--profile-config",
        type=str,
        default=str(ROOT / "configs" / "generation_profiles.yaml"),
        help="谈判空间分层配置 YAML 路径",
    )
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "data" / "scenarios"))
    parser.add_argument("--num-train", type=int, default=5000, help="训练集数量")
    parser.add_argument("--num-val", type=int, default=500, help="验证集数量")
    parser.add_argument("--num-test", type=int, default=500, help="测试集数量")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    templates = load_templates(args.templates)
    zone_profiles = load_zone_profiles(args.profile_config)
    print(f"加载了 {sum(len(c['items']) for c in templates)} 种商品模板")
    print(f"加载了 {len(zone_profiles)} 种谈判空间分层")

    print("\n生成训练集...")
    train = generate_scenarios(templates, zone_profiles, args.num_train, "train", seed=args.seed)
    save_jsonl(train, f"{args.output_dir}/train.jsonl")

    print("生成验证集...")
    val = generate_scenarios(templates, zone_profiles, args.num_val, "val", seed=args.seed + 1)
    save_jsonl(val, f"{args.output_dir}/val.jsonl")

    print("生成测试集...")
    test = generate_scenarios(templates, zone_profiles, args.num_test, "test", seed=args.seed + 2)
    save_jsonl(test, f"{args.output_dir}/test.jsonl")

    # 统计摘要
    print("\n=== 数据统计 ===")
    for name, data in [("train", train), ("val", val), ("test", test)]:
        if not data:
            print(f"{name}: 0 条 | 参考价 [N/A] | 谈判空间 [N/A]")
            continue

        prices = [d["market_ref_price"] for d in data]
        zones = [d["buyer_budget"] - d["seller_cost"] for d in data]
        print(f"{name}: {len(data)} 条 | "
              f"参考价 [{min(prices)}-{max(prices)}] | "
              f"谈判空间 [{min(zones)}-{max(zones)}]")


if __name__ == "__main__":
    main()
