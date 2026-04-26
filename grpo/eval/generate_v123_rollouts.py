import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer

from Final_project.grpo.reward.config import RewardConfig
from Final_project.grpo.rollout.selfplay import SelfPlayRollout
from Final_project.grpo.rollout.vllm_client import VLLMClient
from Final_project.grpo.trainer.negotiation_grpo import ScenarioDataset
from Final_project.src.agent.prompt_builder import PromptBuilder


REPO_ROOT = Path(__file__).resolve().parents[3]


def _json_default(value):
    try:
        return float(value)
    except Exception:
        return str(value)


def _trajectory_to_record(version: str, scenario_index: int, traj) -> Dict[str, Any]:
    state = traj.final_state
    return {
        "version": version,
        "scenario_index": scenario_index,
        "scenario": traj.scenario.to_dict(),
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
            "raw_buyer_reward": traj.raw_buyer_reward,
            "raw_seller_reward": traj.raw_seller_reward,
            "reward_breakdown": traj.reward_breakdown,
            "rounds": len(state.history) / 2,
        },
    }


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records) or 1
    outcomes: Dict[str, int] = {}
    for rec in records:
        outcome = rec["result"]["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return {
        "n": len(records),
        "outcomes": outcomes,
        "deal_rate": sum(1 for r in records if r["result"]["outcome"] == "deal") / n,
        "violation_buyer_rate": sum(1 for r in records if r["result"]["outcome"] == "violation_buyer") / n,
        "violation_seller_rate": sum(1 for r in records if r["result"]["outcome"] == "violation_seller") / n,
        "walkaway_rate": sum(1 for r in records if r["result"]["outcome"] == "walkaway") / n,
        "format_error_rate": sum(1 for r in records if r["result"]["outcome"] == "format_error") / n,
        "avg_buyer_reward": sum(float(r["result"]["buyer_reward"]) for r in records) / n,
        "avg_seller_reward": sum(float(r["result"]["seller_reward"]) for r in records) / n,
        "avg_rounds": sum(float(r["result"]["rounds"]) for r in records) / n,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=str(REPO_ROOT / "models/merged_think"))
    parser.add_argument("--scenarios", default=str(REPO_ROOT / "Final_project/grpo/data/val_grpo_no_think.jsonl"))
    parser.add_argument("--num-scenarios", type=int, default=15)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "Final_project/logs/rollout_v123_15"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--temperature-buyer", type=float, default=0.7)
    parser.add_argument("--temperature-seller", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-tag", default="final")
    parser.add_argument(
        "--selection",
        default=None,
        help="comma-separated specs like v1:final,v3:step_40; defaults to v1/v2/v3 at --checkpoint-tag",
    )
    return parser.parse_args()


def _paths_for(version: str, checkpoint_tag: str) -> Dict[str, Path]:
    run_names = {
        "v1": "stage3_balanced_from_s12_best_100",
        "v2": "stage3_balanced_v2_from_s12_best_100",
        "v3": "stage3_balanced_v3_from_s12_best_100",
    }
    if version not in run_names:
        raise ValueError(f"unknown version {version!r}; expected one of {sorted(run_names)}")
    root = REPO_ROOT / f"checkpoints/grpo/{run_names[version]}/{checkpoint_tag}"
    return {"buyer": root / "buyer", "seller": root / "seller"}


def _build_specs(selection: str, checkpoint_tag: str) -> List[Dict[str, Any]]:
    if selection:
        raw_specs = [item.strip() for item in selection.split(",") if item.strip()]
        specs = []
        for raw in raw_specs:
            if ":" in raw:
                version, tag = raw.split(":", 1)
            else:
                version, tag = raw, checkpoint_tag
            label = f"{version}_{tag}"
            specs.append({
                "label": label,
                "version": version,
                "checkpoint_tag": tag,
                "paths": _paths_for(version, tag),
            })
        return specs

    return [
        {
            "label": version,
            "version": version,
            "checkpoint_tag": checkpoint_tag,
            "paths": _paths_for(version, checkpoint_tag),
        }
        for version in ("v1", "v2", "v3")
    ]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_specs(args.selection, args.checkpoint_tag)
    adapters = {}
    for spec in specs:
        label = spec["label"]
        paths = spec["paths"]
        for role, path in paths.items():
            if not (path / "adapter_config.json").exists():
                raise FileNotFoundError(f"missing adapter: {path}")
            adapters[f"{label}_{role}"] = str(path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    client = VLLMClient(
        base_model=args.base_model,
        adapters=adapters,
        max_lora_rank=64,
        max_loras=len(adapters),
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
        seed=args.seed,
    )
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
        top_p=args.top_p,
    )

    dataset = ScenarioDataset(args.scenarios)
    end = args.start_index + args.num_scenarios
    selected = dataset.scenarios[args.start_index:end]
    scenario_indices = list(range(args.start_index, end))

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": args.base_model,
        "scenarios": args.scenarios,
        "start_index": args.start_index,
        "num_scenarios": len(selected),
        "checkpoint_tag": args.checkpoint_tag,
        "selection": args.selection,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_length": args.max_prompt_length,
            "temperature_buyer": args.temperature_buyer,
            "temperature_seller": args.temperature_seller,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "adapters": {
            spec["label"]: {role: str(path) for role, path in spec["paths"].items()}
            for spec in specs
        },
    }

    summary = {"metadata": metadata, "versions": {}}
    for spec in specs:
        label = spec["label"]
        groups = rollout.rollout_batch(
            scenarios=selected,
            group_size=1,
            active_role="buyer",
            active_adapter=f"{label}_buyer",
            opponent_adapter=f"{label}_seller",
        )
        records = [
            _trajectory_to_record(label, scenario_idx, group.trajectories[0])
            for scenario_idx, group in zip(scenario_indices, groups)
        ]
        version_payload = {"metadata": metadata, "rollouts": records, "summary": _summarize(records)}
        version_path = output_dir / f"{label}_rollouts_{len(selected)}.json"
        with open(version_path, "w", encoding="utf-8") as f:
            json.dump(version_payload, f, ensure_ascii=False, indent=2, default=_json_default)
        summary["versions"][label] = {
            "output": str(version_path),
            "summary": version_payload["summary"],
        }
        print(f"[{label}] wrote {version_path}")
        print(json.dumps(version_payload["summary"], ensure_ascii=False, default=_json_default))

    if args.selection:
        summary_name = "summary_" + "_".join(spec["label"] for spec in specs) + f"_{len(selected)}.json"
    else:
        summary_name = f"summary_{len(selected)}_{args.checkpoint_tag}.json"
    summary_path = output_dir / summary_name
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
