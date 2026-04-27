from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "presentation" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

RUNS = {
    "V1": "7vsxxlmbbwrdvkl7un8cb",
    "V2": "7vvby9yohopjt4relzhra",
    "V3": "vuqqyk78q26px8j4gw16h",
}


def load_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def fetch_swanlab_eval_curves() -> Dict[str, dict]:
    api_key = os.environ.get("SWANLAB_API_KEY") or os.environ.get("SWANLAB_API")
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY/SWANLAB_API is not set")

    import swanlab

    api = swanlab.Api(api_key=api_key)
    curves: Dict[str, dict] = {}
    keys = [
        "eval/active_avg_reward",
        "eval/raw_buyer_avg_reward",
        "eval/raw_seller_avg_reward",
        "eval/outcome_deal_rate",
    ]
    for name, run_id in RUNS.items():
        exp = api.run(f"El1an/grpo-negotiation/{run_id}")
        df = exp.metrics(keys=keys)
        curves[name] = {
            "run_id": run_id,
            "run_name": exp.name,
            "step": [int(x) for x in df.index.tolist()],
            "active_reward": [float(x) for x in df["eval/active_avg_reward"].tolist()],
            "raw_buyer_reward": [float(x) for x in df["eval/raw_buyer_avg_reward"].tolist()],
            "raw_seller_reward": [float(x) for x in df["eval/raw_seller_avg_reward"].tolist()],
            "deal_rate": [float(x) for x in df["eval/outcome_deal_rate"].tolist()],
        }
    (ASSETS / "swanlab_eval_curves.json").write_text(json.dumps(curves, indent=2))
    return curves


def fallback_eval_curves_from_logs() -> Dict[str, dict]:
    files = {
        "V1": "logs/grpo_stage3_balanced_from_s12_best_100_20260426_125411.log",
        "V2": "logs/grpo_stage3_balanced_v2_from_s12_best_100_20260426_144151.log",
        "V3": "logs/grpo_stage3_balanced_v3_from_s12_best_100_20260426_170213.log",
    }
    curves: Dict[str, dict] = {}
    pat = re.compile(r"\[eval step (\d+)\] reward=([-0-9.]+) deal_rate=([-0-9.]+)%")
    for name, rel in files.items():
        steps, rewards, deals = [], [], []
        for line in (ROOT / rel).read_text().splitlines():
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
                deals.append(float(m.group(3)) / 100)
        curves[name] = {
            "run_id": RUNS[name],
            "run_name": rel,
            "step": steps,
            "active_reward": rewards,
            "raw_buyer_reward": [],
            "raw_seller_reward": [],
            "deal_rate": deals,
        }
    return curves


def plot_swanlab_reward(curves: Dict[str, dict]) -> None:
    colors = {"V1": "#237e56", "V2": "#2d6396", "V3": "#aa3838"}
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=180)
    for name in ["V1", "V2", "V3"]:
        c = curves[name]
        ax.plot(c["step"], c["active_reward"], marker="o", linewidth=2.2, label=name, color=colors[name])
    ax.set_xlabel("Stage 3 eval step")
    ax.set_ylabel("Eval active average reward")
    ax.set_title("SwanLab eval reward curves")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(ASSETS / "swanlab_reward_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_ablation() -> None:
    summary = load_json("logs/rollout_v123_15/summary_15_final.json")["versions"]
    names = ["V1", "V2", "V3"]
    keys = ["v1", "v2", "v3"]
    deal = [summary[k]["summary"]["deal_rate"] * 100 for k in keys]
    total_violation = [
        (summary[k]["summary"]["violation_buyer_rate"] + summary[k]["summary"]["violation_seller_rate"]) * 100
        for k in keys
    ]
    avg_reward = [
        (summary[k]["summary"]["avg_buyer_reward"] + summary[k]["summary"]["avg_seller_reward"]) / 2
        for k in keys
    ]

    fig, ax1 = plt.subplots(figsize=(7.0, 3.0), dpi=180)
    x = range(len(names))
    ax1.bar([i - 0.18 for i in x], deal, width=0.36, label="Deal rate", color="#237e56")
    ax1.bar([i + 0.18 for i in x], total_violation, width=0.36, label="Violation rate", color="#aa3838")
    ax1.set_xticks(list(x), names)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Rate (%) on final 15-rollout set")
    ax2 = ax1.twinx()
    ax2.plot(list(x), avg_reward, marker="o", color="#750f6d", label="Mean role reward")
    ax2.set_ylabel("Mean role reward")
    ax2.set_ylim(0, max(avg_reward) * 1.4)
    ax1.grid(axis="y", alpha=0.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "v123_ablation_final.png", bbox_inches="tight")
    plt.close(fig)


def plot_trace() -> None:
    rollout = load_json("logs/rollout_selected_50/v1_final_rollouts_50.json")["rollouts"][47]
    dialogue = rollout["dialogue"]
    buyer_x, buyer_y, seller_x, seller_y = [], [], [], []
    for i, turn in enumerate(dialogue):
        if turn.get("action_type") == "offer" and turn.get("price") is not None:
            if turn["role"] == "buyer":
                buyer_x.append(i)
                buyer_y.append(float(turn["price"]))
            else:
                seller_x.append(i)
                seller_y.append(float(turn["price"]))

    fig, ax = plt.subplots(figsize=(7.0, 2.5), dpi=180)
    ax.plot(seller_x, seller_y, marker="o", linewidth=2.2, color="#750f6d", label="Seller offers")
    ax.plot(buyer_x, buyer_y, marker="o", linewidth=2.2, color="#2d6396", label="Buyer offers")
    ax.axhline(1220, color="#237e56", linestyle="--", linewidth=1.5, label="Deal at 1220")
    ax.set_xticks(range(len(dialogue)), ["S0", "B0", "S1", "B1", "S2", "B2", "Deal"])
    ax.set_ylabel("Price")
    ax.set_title("V1 rollout trace: concessions converge to a balanced deal")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "v1_trace_price_path.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    try:
        curves = fetch_swanlab_eval_curves()
        source = "swanlab"
    except Exception as exc:
        print(f"[warn] SwanLab fetch failed, falling back to local logs: {exc}")
        curves = fallback_eval_curves_from_logs()
        source = "local_logs"
    plot_swanlab_reward(curves)
    plot_ablation()
    plot_trace()
    print(f"Built final slide assets from {source}")


if __name__ == "__main__":
    main()
