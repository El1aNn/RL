from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT_FIGS = ROOT / "report" / "figures"
REPORT_FIGS.mkdir(parents=True, exist_ok=True)
METRICS_JSON = REPORT_FIGS / "swanlab_stage3_metrics.json"

RUNS = {
    "V1": "7vsxxlmbbwrdvkl7un8cb",
    "V2": "7vvby9yohopjt4relzhra",
    "V3": "vuqqyk78q26px8j4gw16h",
}

KEYS = [
    "eval/active_avg_reward",
    "eval/raw_buyer_avg_reward",
    "eval/raw_seller_avg_reward",
    "eval/outcome_deal_rate",
    "eval/outcome_violation_buyer_rate",
    "eval/outcome_violation_seller_rate",
    "eval/outcome_walkaway_rate",
    "eval/outcome_format_error_rate",
    "eval/avg_rounds",
]

COLORS = {"V1": "#237e56", "V2": "#2d6396", "V3": "#aa3838"}


def _col_values(df, key: str) -> List[float]:
    return [float(x) for x in df[key].tolist()]


def fetch_metrics() -> Dict[str, dict]:
    api_key = os.environ.get("SWANLAB_API_KEY") or os.environ.get("SWANLAB_API")
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY/SWANLAB_API is not set")

    import swanlab

    api = swanlab.Api(api_key=api_key)
    out: Dict[str, dict] = {}
    for name, run_id in RUNS.items():
        exp = api.run(f"El1an/grpo-negotiation/{run_id}")
        df = exp.metrics(keys=KEYS)
        out[name] = {
            "run_id": run_id,
            "run_name": exp.name,
            "step": [int(x) for x in df.index.tolist()],
            "active_reward": _col_values(df, "eval/active_avg_reward"),
            "raw_buyer_reward": _col_values(df, "eval/raw_buyer_avg_reward"),
            "raw_seller_reward": _col_values(df, "eval/raw_seller_avg_reward"),
            "deal_rate": _col_values(df, "eval/outcome_deal_rate"),
            "violation_buyer_rate": _col_values(df, "eval/outcome_violation_buyer_rate"),
            "violation_seller_rate": _col_values(df, "eval/outcome_violation_seller_rate"),
            "walkaway_rate": _col_values(df, "eval/outcome_walkaway_rate"),
            "format_error_rate": _col_values(df, "eval/outcome_format_error_rate"),
            "avg_rounds": _col_values(df, "eval/avg_rounds"),
        }
    METRICS_JSON.write_text(json.dumps(out, indent=2))
    return out


def load_cached_metrics() -> Dict[str, dict]:
    return json.loads(METRICS_JSON.read_text())


def plot_stage3_dynamics(metrics: Dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), dpi=180)
    panels = [
        ("deal_rate", "Deal rate", (0.6, 1.0), True),
        ("violation_seller_rate", "Seller violation rate", (0.0, 0.32), True),
        ("avg_rounds", "Average rounds", (1.5, 4.2), False),
    ]
    for ax, (key, title, ylim, pct) in zip(axes, panels):
        for name in ["V1", "V2", "V3"]:
            series = metrics[name]
            y = [v * 100 for v in series[key]] if pct else series[key]
            ax.plot(series["step"], y, marker="o", linewidth=2.1, color=COLORS[name], label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Eval step")
        ax.set_ylim(*([(v * 100) for v in ylim] if pct else ylim))
        if pct:
            ax.set_ylabel("Percent")
        else:
            ax.set_ylabel("Turns")
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(REPORT_FIGS / "swanlab_stage3_dynamics.png", bbox_inches="tight")
    plt.close(fig)


def plot_balance(metrics: Dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=180)
    for name in ["V1", "V2", "V3"]:
        series = metrics[name]
        gap = [abs(b - s) for b, s in zip(series["raw_buyer_reward"], series["raw_seller_reward"])]
        axes[0].plot(series["step"], gap, marker="o", linewidth=2.1, color=COLORS[name], label=name)
        axes[1].plot(
            series["step"],
            series["raw_seller_reward"],
            marker="o",
            linewidth=2.1,
            color=COLORS[name],
            label=name,
        )
    axes[0].set_title("Buyer-seller reward gap", fontsize=10)
    axes[0].set_xlabel("Eval step")
    axes[0].set_ylabel("Absolute reward difference")
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].set_title("Seller raw reward", fontsize=10)
    axes[1].set_xlabel("Eval step")
    axes[1].set_ylabel("Reward")
    axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=1.2)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, ncol=3, loc="lower right")

    fig.tight_layout()
    fig.savefig(REPORT_FIGS / "swanlab_stage3_balance.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    try:
        metrics = fetch_metrics()
        source = "swanlab"
    except Exception as exc:
        if not METRICS_JSON.exists():
            raise RuntimeError(f"SwanLab fetch failed and no cache exists: {exc}") from exc
        metrics = load_cached_metrics()
        source = "cache"
    plot_stage3_dynamics(metrics)
    plot_balance(metrics)
    print(f"Built report SwanLab assets from {source}")


if __name__ == "__main__":
    main()
