"""
评估脚本入口

用法：
    python -m Final_project.grpo.eval.run_eval \
        --base-model ./checkpoints/sft_base \
        --buyer-adapter ./checkpoints/grpo/buyer_best \
        --seller-adapter ./checkpoints/grpo/seller_best \
        --scenarios Final_project/data/scenarios_rl_5k/test.jsonl \
        --group-size 4 \
        --output eval_results.json

对比基线可以通过 --baseline-mode 切换：
    - "ours":     buyer_adapter + seller_adapter 互打
    - "sft_only": 双方都用 base（不挂 adapter）
    - "zero_shot": 同 sft_only 但用原始 base
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from Final_project.src.agent.prompt_builder import PromptBuilder
from Final_project.grpo.reward.config import RewardConfig
from Final_project.grpo.rollout.vllm_client import VLLMClient
from Final_project.grpo.rollout.selfplay import SelfPlayRollout
from Final_project.grpo.trainer.negotiation_grpo import ScenarioDataset
from Final_project.grpo.eval.metrics import compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, help="SFT merge 后的 base 路径")
    p.add_argument("--buyer-adapter", default=None)
    p.add_argument("--seller-adapter", default=None)
    p.add_argument("--scenarios", required=True, help="scenario jsonl 路径")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-scenarios", type=int, default=-1, help="最多评估多少个场景")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output", default="eval_results.json")
    p.add_argument("--baseline-mode", default="ours",
                   choices=["ours", "sft_only", "zero_shot"],
                   help="ours=两个 adapter 互打; sft_only/zero_shot=都用 base")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()

    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. vLLM client
    adapters = {}
    if args.baseline_mode == "ours":
        assert args.buyer_adapter and args.seller_adapter, \
            "ours 模式需要 --buyer-adapter 和 --seller-adapter"
        adapters = {"buyer": args.buyer_adapter, "seller": args.seller_adapter}

    client = VLLMClient(
        base_model=args.base_model,
        adapters=adapters,
        max_lora_rank=64,
        max_loras=max(len(adapters), 1),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 3. rollout engine
    rollout = SelfPlayRollout(
        vllm_client=client,
        tokenizer=tokenizer,
        prompt_builder=PromptBuilder(),
        reward_cfg=RewardConfig(),
        temperature_active=args.temperature,
        temperature_opponent=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # 4. scenarios
    dataset = ScenarioDataset(args.scenarios)
    if args.max_scenarios > 0:
        dataset.scenarios = dataset.scenarios[: args.max_scenarios]
    print(f"Loaded {len(dataset)} scenarios from {args.scenarios}")

    # 5. 跑 rollout（评估时 active_role 随意，结果对称；取 "buyer"）
    if args.baseline_mode == "ours":
        active_adp, opp_adp = "buyer", "seller"
    else:
        # 用一个虚拟 adapter name；adapters 为空时 VLLMClient._make_lora_request 会报错
        # 因此这里我们直接让 active/opp 都传 None（generate() 支持 None）
        active_adp, opp_adp = None, None

    all_trajectories = []
    batch_size = 8
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        groups = rollout.rollout_batch(
            scenarios=batch,
            group_size=args.group_size,
            active_role="buyer",
            active_adapter=active_adp,
            opponent_adapter=opp_adp,
        )
        for g in groups:
            all_trajectories.extend(g.trajectories)

        print(f"  processed {start + len(batch)} / {len(dataset)}")

    # 6. 计算指标
    metrics = compute_all_metrics(all_trajectories)
    metrics["config"] = {
        "base_model": args.base_model,
        "buyer_adapter": args.buyer_adapter,
        "seller_adapter": args.seller_adapter,
        "baseline_mode": args.baseline_mode,
        "group_size": args.group_size,
        "temperature": args.temperature,
    }

    # 7. 保存
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        if k == "config":
            continue
        print(f"  {k}: {v}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
