"""
Run a tiny qualitative demo: stage1 best buyer vs cold-start seller.

Example:
    python -m Final_project.grpo.eval.demo_best_buyer_vs_cold_seller
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer

from Final_project.grpo.reward.config import RewardConfig
from Final_project.grpo.rollout.selfplay import SelfPlayRollout
from Final_project.grpo.rollout.vllm_client import VLLMClient
from Final_project.grpo.trainer.negotiation_grpo import ScenarioDataset
from Final_project.src.agent.prompt_builder import PromptBuilder


REPO_ROOT = Path(__file__).resolve().parents[3]


def _default_seller_adapter() -> str:
    """Prefer the current stage2 cold-start seller; fall back to stage1 init."""
    for path in (
        REPO_ROOT / "checkpoints/grpo/stage2/_init_seller",
        REPO_ROOT / "checkpoints/grpo/stage1/_init_seller",
    ):
        if (path / "adapter_config.json").exists():
            return str(path)
    return str(REPO_ROOT / "checkpoints/grpo/stage1/_init_seller")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=str(REPO_ROOT / "models/merged_think"))
    p.add_argument("--buyer-adapter", default=str(REPO_ROOT / "checkpoints/grpo/stage1/best/buyer"))
    p.add_argument("--seller-adapter", default=None, help="default: stage2/_init_seller, fallback stage1/_init_seller")
    p.add_argument("--scenarios", default=str(REPO_ROOT / "Final_project/grpo/data/val_grpo_no_think.jsonl"))
    p.add_argument("--scenario-index", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-prompt-length", type=int, default=1536)
    p.add_argument("--temperature-buyer", type=float, default=0.7)
    p.add_argument("--temperature-seller", type=float, default=0.7)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="optional JSON output path")
    return p.parse_args()


def _price(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}"


def _json_default(value):
    try:
        return float(value)
    except Exception:
        return str(value)


def main():
    args = parse_args()
    seller_adapter = args.seller_adapter or _default_seller_adapter()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    client = VLLMClient(
        base_model=args.base_model,
        adapters={
            "buyer": args.buyer_adapter,
            "seller": seller_adapter,
        },
        max_lora_rank=64,
        max_loras=2,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
        seed=args.seed,
    )

    dataset = ScenarioDataset(args.scenarios)
    scenario = dataset.scenarios[args.scenario_index]

    rollout = SelfPlayRollout(
        vllm_client=client,
        tokenizer=tokenizer,
        prompt_builder=PromptBuilder(),
        reward_cfg=RewardConfig(),
        env_config={"format_error_budget": 2},
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        temperature_active=args.temperature_buyer,
        temperature_opponent=args.temperature_seller,
        top_p=0.9,
    )

    groups = rollout.rollout_batch(
        scenarios=[scenario],
        group_size=1,
        active_role="buyer",
        active_adapter="buyer",
        opponent_adapter="seller",
    )
    traj = groups[0].trajectories[0]
    state = traj.final_state

    print("\n=== Demo: best buyer vs cold-start seller ===")
    print(f"buyer_adapter : {args.buyer_adapter}")
    print(f"seller_adapter: {seller_adapter}")
    print(f"scenario      : {scenario.scenario_id} | {scenario.item_name}")
    print(f"description   : {scenario.item_description}")
    print(
        "private zone  : "
        f"buyer_budget={_price(float(scenario.buyer_budget))}, "
        f"seller_cost={_price(float(scenario.seller_cost))}, "
        f"market_ref={_price(float(scenario.market_ref_price))}"
    )
    print("\n--- Dialogue ---")
    for i, turn in enumerate(state.history, start=1):
        parsed = turn.parsed
        parsed_price = "" if parsed.price is None else f" @{parsed.price:.0f}"
        print(f"{i:02d}. {turn.role} [{parsed.action_type}{parsed_price}]")
        print(turn.utterance.strip())
        print()

    print("--- Result ---")
    print(f"outcome      : {state.outcome.value}")
    print(f"deal_price   : {_price(state.deal_price)}")
    print(f"reason       : {state.terminated_reason}")
    print(f"buyer_reward : {traj.buyer_reward:.2f}")
    print(f"seller_reward: {traj.seller_reward:.2f}")
    print("breakdown    : " + json.dumps(traj.reward_breakdown, ensure_ascii=False, default=_json_default))

    if args.output:
        out = {
            "buyer_adapter": args.buyer_adapter,
            "seller_adapter": seller_adapter,
            "scenario": scenario.to_dict(),
            "dialogue": [
                {
                    "round": turn.round_num,
                    "role": turn.role,
                    "utterance": turn.utterance,
                    "action_type": turn.parsed.action_type,
                    "price": turn.parsed.price,
                    "is_format_valid": turn.parsed.is_format_valid,
                }
                for turn in state.history
            ],
            "result": {
                "outcome": state.outcome.value,
                "deal_price": state.deal_price,
                "terminated_reason": state.terminated_reason,
                "buyer_reward": traj.buyer_reward,
                "seller_reward": traj.seller_reward,
                "reward_breakdown": traj.reward_breakdown,
            },
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nsaved_json   : {path}")


if __name__ == "__main__":
    main()
