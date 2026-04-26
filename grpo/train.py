"""
统一训练入口

用法：
    # 阶段 1
    python -m Final_project.grpo.train --config Final_project/grpo/configs/default.yaml --stage stage1

    # 阶段 2（需要先完成 stage1）
    python -m Final_project.grpo.train --config ... --stage stage2 \
        --override "adapter_init.buyer=./checkpoints/grpo/stage1/best"

    # 阶段 3（需要 stage1 + stage2）
    python -m Final_project.grpo.train --config ... --stage stage3 \
        --override "adapter_init.buyer=./checkpoints/grpo/stage1/best" \
        --override "adapter_init.seller=./checkpoints/grpo/stage2/best"
"""
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

from Final_project.src.agent.prompt_builder import PromptBuilder
from Final_project.grpo.reward.config import RewardConfig
from Final_project.grpo.rollout.vllm_client import VLLMClient
from Final_project.grpo.rollout.selfplay import SelfPlayRollout
from Final_project.grpo.trainer.adapter_manager import AdapterManager
from Final_project.grpo.trainer.negotiation_grpo import (
    GRPOConfig, NegotiationGRPOTrainer, ScenarioDataset,
)


# ============================================================
# Config 合并与覆盖
# ============================================================

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_override(cfg: Dict[str, Any], key_path: str, value: str) -> None:
    """把 'a.b.c=xxx' 形式的 override 写进 cfg"""
    keys = key_path.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    # 尝试类型推断
    v = _auto_cast(value)
    d[keys[-1]] = v


def _auto_cast(s: str) -> Any:
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("null", "none"):
        return None
    try:
        if "." in s or "e" in s or "E" in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def merge_stage_config(cfg: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """把 stageX 段平铺到 train 段"""
    if stage not in cfg:
        raise ValueError(f"stage '{stage}' not in config")
    stage_cfg = cfg[stage]
    merged = dict(cfg.get("train", {}))
    merged.update(stage_cfg)
    cfg["_merged_train"] = merged
    return cfg


def _sanitize_for_logging(obj: Any) -> Any:
    """复制配置用于实验记录，避免把 api key 等敏感字段写进看板。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k).lower()
            if any(s in key for s in ("api_key", "apikey", "token", "secret", "password")):
                out[k] = "***" if v else v
            else:
                out[k] = _sanitize_for_logging(v)
        return out
    if isinstance(obj, list):
        return [_sanitize_for_logging(v) for v in obj]
    return obj


class SwanLabMetricLogger:
    """很薄的一层封装，避免 SwanLab 报错直接中断训练。"""

    def __init__(self, swanlab_module):
        self._swanlab = swanlab_module

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        cleaned = {}
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, bool):
                cleaned[k] = int(v)
            elif isinstance(v, int):
                cleaned[k] = v
            elif isinstance(v, float) and math.isfinite(v):
                cleaned[k] = v
        if cleaned:
            self._swanlab.log(cleaned, step=step)

    def finish(self, error: BaseException = None) -> None:
        try:
            if error is None:
                self._swanlab.finish()
            else:
                from swanlab.data.run import SwanLabRunState
                self._swanlab.finish(SwanLabRunState.CRASHED, error=str(error))
        except Exception as e:
            print(f"[swanlab] finish failed: {e}", file=sys.stderr)


def build_metric_logger(cfg: Dict[str, Any], stage: str):
    scfg = cfg.get("swanlab") or {}
    if not bool(scfg.get("enabled", False)):
        return None

    try:
        import swanlab

        api_key = (
            scfg.get("api_key")
            or os.getenv("SWANLAB_API_KEY")
            or os.getenv("SWANLAB_API")
        )
        mode = scfg.get("mode", "cloud")
        if mode == "cloud" and not api_key:
            print("[swanlab] enabled but swanlab.api_key is empty; disable SwanLab logging.", file=sys.stderr)
            return None

        if api_key:
            swanlab.login(
                api_key=api_key,
                host=scfg.get("host"),
                web_host=scfg.get("web_host"),
                save=bool(scfg.get("save_api_key", False)),
            )

        logged_cfg = _sanitize_for_logging(cfg)
        logged_cfg["stage"] = stage

        init_kwargs = {
            "project": scfg.get("project", "grpo-negotiation"),
            "workspace": scfg.get("workspace"),
            "experiment_name": scfg.get("experiment_name") or f"grpo-{stage}",
            "description": scfg.get("description"),
            "job_type": scfg.get("job_type", "train"),
            "group": scfg.get("group"),
            "tags": scfg.get("tags"),
            "config": logged_cfg,
            "logdir": scfg.get("logdir"),
            "mode": mode,
        }
        swanlab.init(**{k: v for k, v in init_kwargs.items() if v is not None})
        print("[swanlab] logging enabled", file=sys.stderr)
        return SwanLabMetricLogger(swanlab)
    except Exception as e:
        if bool(scfg.get("fail_on_error", False)):
            raise
        print(f"[swanlab] disabled because init failed: {e}", file=sys.stderr)
        return None


# ============================================================
# 构建各个组件
# ============================================================

def build_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model_path"], trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model_with_adapters(cfg, stage_cfg):
    """加载 base model 并挂上 buyer / seller 两个 adapter"""
    dtype_str = cfg["model"].get("dtype", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_str, torch.bfloat16)

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model_path"],
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    mgr = AdapterManager(base)

    # 构造 LoraConfig
    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["rank"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=list(cfg["lora"]["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 加载或新建 buyer / seller adapter
    init = cfg.get("adapter_init") or {}

    if init.get("buyer"):
        mgr.load_adapter("buyer", init["buyer"])
    else:
        mgr.ensure_adapter("buyer", lora_cfg)

    if init.get("seller"):
        mgr.load_adapter("seller", init["seller"])
    else:
        mgr.ensure_adapter("seller", lora_cfg)

    # 移动到 GPU
    if torch.cuda.is_available():
        mgr.peft_model.to("cuda")

    return mgr


def build_vllm_client(cfg, adapter_paths: Dict[str, str]) -> VLLMClient:
    vcfg = cfg.get("vllm", {})
    return VLLMClient(
        base_model=cfg["model"]["base_model_path"],
        adapters=adapter_paths,
        max_lora_rank=int(vcfg.get("max_lora_rank", 64)),
        max_loras=int(vcfg.get("max_loras", 3)),
        gpu_memory_utilization=float(vcfg.get("gpu_memory_utilization", 0.55)),
        dtype=cfg["model"].get("dtype", "bfloat16"),
        max_model_len=int(cfg["model"].get("max_model_len", 4096)),
        enforce_eager=bool(vcfg.get("enforce_eager", False)),
        seed=int(cfg.get("seed", 42)),
    )


def build_reward_config(cfg) -> RewardConfig:
    return RewardConfig.from_dict(cfg.get("reward", {}))


def build_rollout(vllm_client, tokenizer, cfg, stage_cfg) -> SelfPlayRollout:
    rcfg = cfg.get("rollout", {})
    return SelfPlayRollout(
        vllm_client=vllm_client,
        tokenizer=tokenizer,
        prompt_builder=PromptBuilder(),
        reward_cfg=build_reward_config(cfg),
        env_config={"format_error_budget": int(cfg["env"]["format_error_budget"])},
        max_new_tokens=int(rcfg.get("max_new_tokens", 128)),
        max_prompt_length=int(stage_cfg.get("max_prompt_length", 1536)),
        temperature_active=float(rcfg.get("temperature_active", 0.9)),
        temperature_opponent=float(rcfg.get("temperature_opponent", 0.7)),
        top_p=float(rcfg.get("top_p", 0.9)),
        seller_cold_guard=stage_cfg.get("seller_cold_guard"),
    )


def build_grpo_config(cfg, stage_cfg) -> GRPOConfig:
    rcfg = cfg.get("rollout", {})
    return GRPOConfig(
        total_steps=int(stage_cfg["total_steps"]),
        per_device_train_batch_size=int(stage_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(stage_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(stage_cfg["learning_rate"]),
        weight_decay=float(stage_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(stage_cfg.get("max_grad_norm", 1.0)),
        policy_mini_batch_size=int(stage_cfg.get("policy_mini_batch_size", 1)),
        group_size=int(rcfg.get("group_size", 16)),
        clip_epsilon=float(stage_cfg["clip_epsilon"]),
        beta_kl=float(stage_cfg["beta_kl"]),
        advantage_eps=float(stage_cfg.get("advantage_eps", 1e-4)),
        max_new_tokens=int(rcfg.get("max_new_tokens", 128)),
        temperature_active=float(rcfg.get("temperature_active", 0.9)),
        temperature_opponent=float(rcfg.get("temperature_opponent", 0.7)),
        top_p=float(rcfg.get("top_p", 0.9)),
        max_prompt_length=int(stage_cfg["max_prompt_length"]),
        max_completion_length=int(stage_cfg["max_completion_length"]),
        active_role=stage_cfg["active_role"],
        active_adapter=stage_cfg["active_adapter"],
        opponent_adapter=stage_cfg["opponent_adapter"],
        output_dir=stage_cfg["output_dir"],
        save_every=int(stage_cfg["save_every"]),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        eval_every=int(stage_cfg["eval_every"]),
        logging_every=int(cfg.get("logging_every", 10)),
        seed=int(cfg.get("seed", 42)),
        gradient_checkpointing=bool(stage_cfg.get("gradient_checkpointing", True)),
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--override", action="append", default=[],
                        help="覆盖配置，格式 key.path=value，可多次")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    for ov in args.override:
        if "=" not in ov:
            print(f"skip bad override: {ov}", file=sys.stderr)
            continue
        k, v = ov.split("=", 1)
        apply_override(cfg, k, v)

    cfg = merge_stage_config(cfg, args.stage)
    stage_cfg = cfg["_merged_train"]
    metric_logger = build_metric_logger(cfg, args.stage)

    # === 构建各组件 ===
    print(f"[train] stage={args.stage}  active_role={stage_cfg['active_role']}")
    tokenizer = build_tokenizer(cfg)

    # 1. 训练模型（HF + PEFT，用来算 logp 和更新参数）
    mgr = build_model_with_adapters(cfg, stage_cfg)

    # 2. 推理引擎（vLLM，用来 rollout）
    adapter_paths = {}
    init = cfg.get("adapter_init") or {}
    if init.get("buyer"):
        adapter_paths["buyer"] = init["buyer"]
    else:
        # 未加载已有 adapter 时，先把当前（新建的）adapter 保存一份给 vLLM 用
        tmp = Path(stage_cfg["output_dir"]) / "_init_buyer"
        mgr.save_adapter("buyer", str(tmp))
        adapter_paths["buyer"] = str(tmp)
    if init.get("seller"):
        adapter_paths["seller"] = init["seller"]
    else:
        tmp = Path(stage_cfg["output_dir"]) / "_init_seller"
        mgr.save_adapter("seller", str(tmp))
        adapter_paths["seller"] = str(tmp)

    vllm_client = build_vllm_client(cfg, adapter_paths)

    # 3. rollout engine
    rollout = build_rollout(vllm_client, tokenizer, cfg, stage_cfg)

    # 4. 训练 / 验证数据
    train_ds = ScenarioDataset(stage_cfg["scenarios_path"])
    val_ds = ScenarioDataset(stage_cfg["val_path"]) if stage_cfg.get("val_path") else None

    # 5. trainer
    grpo_cfg = build_grpo_config(cfg, stage_cfg)
    trainer = NegotiationGRPOTrainer(
        adapter_manager=mgr,
        tokenizer=tokenizer,
        rollout_engine=rollout,
        config=grpo_cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        metric_logger=metric_logger,
    )

    # 6. 训练
    train_error = None
    try:
        trainer.train()
    except BaseException as e:
        train_error = e
        raise
    finally:
        if metric_logger is not None:
            metric_logger.finish(train_error)


if __name__ == "__main__":
    main()
