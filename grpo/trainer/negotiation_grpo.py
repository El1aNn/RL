"""
NegotiationGRPOTrainer

因为 trl.GRPOTrainer 默认假设 single-turn rollout（一次 prompt -> 一次 completion），
而我们的 self-play 是 multi-turn（一场对话里 active role 可能说 1~5 次），
直接继承并覆盖 _generate_and_score_completions 改动面太大。

所以这里自实现一个轻量级 GRPO trainer：
- rollout 由 SelfPlayRollout 产生
- 训练循环手写，loss 公式与 DeepSeek-GRPO 论文一致
- 用 AdapterManager 做 old/ref 计算（disable adapter = ref；训练分支 = policy）
- 支持 stage1 / stage2 / stage3 三种 active_role 模式

Loss 公式：
    ratio = exp(log_pi_new - log_pi_old)   # per-token importance ratio
    surr1 = ratio * advantage
    surr2 = clip(ratio, 1-eps, 1+eps) * advantage
    pg_loss = -mean(min(surr1, surr2))    # per-token、mask 掉 padding

    kl = (log_pi_new - log_pi_ref) - exp(log_pi_ref - log_pi_new) + 1.0   # 近似 KL
    loss = pg_loss + beta * mean(kl)

Advantage：同 scenario 内的 group_size 条 trajectory 按 advantage_reward 做 z-score。
同一条 trajectory 中 active_role 的所有 turn 共享同一 advantage（broadcast）。
"""
import math
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Final_project.src.environment.scenario import NegotiationScenario
from Final_project.grpo.rollout.selfplay import (
    SelfPlayRollout, RolloutGroup, RolloutTrajectory, ActiveTurnRecord,
)
from Final_project.grpo.trainer.adapter_manager import AdapterManager


# ============================================================
# 配置
# ============================================================

@dataclass
class GRPOConfig:
    # 训练
    total_steps: int = 500
    per_device_train_batch_size: int = 4         # 每 step 处理的 scenario 数
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    policy_mini_batch_size: int = 1

    # GRPO loss
    group_size: int = 16
    clip_epsilon: float = 0.2
    beta_kl: float = 0.04
    advantage_eps: float = 1e-4

    # Rollout
    max_new_tokens: int = 128
    temperature_active: float = 0.9
    temperature_opponent: float = 0.7
    top_p: float = 0.9
    max_prompt_length: int = 1536
    max_completion_length: int = 128

    # Multi-role
    active_role: str = "buyer"          # "buyer" | "seller" | "alternating"
    active_adapter: str = "buyer"
    opponent_adapter: str = "seller"

    # Checkpoint
    output_dir: str = "./checkpoints/grpo"
    save_every: int = 100
    save_total_limit: int = 2
    eval_every: int = 50
    logging_every: int = 10

    # 其它
    seed: int = 42
    gradient_checkpointing: bool = True


# ============================================================
# 场景数据集
# ============================================================

class ScenarioDataset(Dataset):
    """从 jsonl 读取 NegotiationScenario 列表"""

    def __init__(self, jsonl_path: str):
        self.scenarios: List[NegotiationScenario] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.scenarios.append(NegotiationScenario.from_dict(json.loads(line)))

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> NegotiationScenario:
        return self.scenarios[idx]


def scenario_collate(batch: List[NegotiationScenario]) -> List[NegotiationScenario]:
    """保持为 list，不做 tensor 化（rollout 阶段才处理）"""
    return list(batch)


# ============================================================
# 展平后的训练样本
# ============================================================

@dataclass
class FlatSample:
    """一个 active turn = 一条训练样本"""
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    advantage: float


# ============================================================
# Trainer
# ============================================================

class NegotiationGRPOTrainer:
    """
    自实现的轻量级 GRPO Trainer（面向 negotiation self-play）。
    """

    def __init__(
        self,
        adapter_manager: AdapterManager,
        tokenizer,
        rollout_engine: SelfPlayRollout,
        config: GRPOConfig,
        train_dataset: ScenarioDataset,
        val_dataset: Optional[ScenarioDataset] = None,
        device: Optional[str] = None,
        logger=None,
        metric_logger=None,
    ):
        self.mgr = adapter_manager
        self.tokenizer = tokenizer
        self.rollout = rollout_engine
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or _StderrLogger()
        self.metric_logger = metric_logger

        self.global_step = 0
        self.best_val_reward = -math.inf
        self._grad_accum_counts: Dict[str, int] = {"buyer": 0, "seller": 0}

        torch.manual_seed(config.seed)

        # optimizer 只优化 active adapter
        self._setup_optimizer()
        self._setup_dataloader()

    # ------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------

    def _setup_optimizer(self) -> None:
        """
        为 buyer 和 seller 各自建一个 optimizer。

        - Stage 1/2：只激活其中一个（另一个 optimizer 永远不会 step）
        - Stage 3 alternating：每 step 根据角色选其中一个 optimizer
        - 可训练参数同时对两个 adapter 解冻，通过「梯度只流经当前 active 的 adapter
          + 只 step 对应 optimizer」保证隔离
        """
        cfg = self.config
        model = self.mgr.peft_model
        assert model is not None, "adapter_manager has no loaded model"

        # 1. 先冻结所有参数
        for _, p in model.named_parameters():
            p.requires_grad = False

        # 2. 按 adapter 分组解冻（两个 adapter 的 lora_ 参数都需要 requires_grad=True
        #    alternating 才能在切换时算 grad；但 optimizer.step 只会更新对应那组）
        self._adapter_params: Dict[str, List] = {"buyer": [], "seller": []}
        for n, p in model.named_parameters():
            if "lora_" not in n:
                continue
            if ".buyer." in n:
                p.requires_grad = True
                self._adapter_params["buyer"].append(p)
            elif ".seller." in n:
                p.requires_grad = True
                self._adapter_params["seller"].append(p)

        for name in ("buyer", "seller"):
            cnt = sum(p.numel() for p in self._adapter_params[name])
            self.logger.log(f"trainable LoRA params for adapter '{name}': {cnt:,}")

        # 3. 每个 adapter 独立的 optimizer
        self._optimizers: Dict[str, torch.optim.Optimizer] = {
            "buyer": torch.optim.AdamW(
                self._adapter_params["buyer"],
                lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
            ),
            "seller": torch.optim.AdamW(
                self._adapter_params["seller"],
                lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
            ),
        }

        if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if hasattr(model, "config"):
                model.config.use_cache = False

    # 兼容旧字段：保留 self.optimizer 指向当前 active 的那个
    @property
    def optimizer(self):
        return self._optimizers[self.config.active_adapter]

    def _setup_dataloader(self) -> None:
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=scenario_collate,
            drop_last=True,
        )

    # ------------------------------------------------------------
    # 主训练循环
    # ------------------------------------------------------------

    def train(self) -> None:
        cfg = self.config
        model = self.mgr.peft_model
        model.train()

        train_iter = self._infinite_iterator(self.train_loader)

        while self.global_step < cfg.total_steps:
            t_start = time.time()

            # === 1. 选定本 step 的角色（stage3 alternating）===
            role, active_adp, opp_adp = self._pick_role_for_step()

            # === 2. 采一个 scenario batch ===
            scenarios: List[NegotiationScenario] = next(train_iter)

            # === 3. Rollout（no_grad，在 vLLM 里跑）===
            model.eval()
            with torch.no_grad():
                groups: List[RolloutGroup] = self.rollout.rollout_batch(
                    scenarios=scenarios,
                    group_size=cfg.group_size,
                    active_role=role,
                    active_adapter=active_adp,
                    opponent_adapter=opp_adp,
                )
            model.train()

            # === 4. 组内 advantage 标准化 + 展平样本 ===
            flat_samples = self._build_flat_samples(groups, role)
            if not flat_samples:
                self.logger.log(f"[step {self.global_step}] no active turns, skip")
                self.global_step += 1
                continue

            # === 5. 计算 old / ref log-probs（no_grad）===
            with torch.no_grad():
                old_logps = self._compute_logps(flat_samples, use_adapter=active_adp)
                ref_logps = self._compute_logps(flat_samples, use_adapter=None)  # base = ref

            # === 6. 计算 loss 并反向 ===
            loss_info = self._grpo_step(flat_samples, old_logps, ref_logps, active_adp)

            # === 6.5 Optimizer step 成功后立即同步 vLLM adapter ===
            if loss_info.get("did_optimizer_step"):
                self._sync_vllm_adapter(active_adp)

            # === 7. 日志 ===
            step_time = time.time() - t_start
            self._log_step(role, groups, flat_samples, loss_info, step_time)

            self.global_step += 1

            # === 8. eval / save ===
            if self.global_step % cfg.eval_every == 0 and self.val_dataset is not None:
                self._evaluate()
            if self.global_step % cfg.save_every == 0:
                self._save_checkpoint()

        # 训完前把没满 accumulation window 的梯度也落一次，避免最后几步白算。
        self._flush_pending_optimizer_steps()

        # 训完最后保存一次
        self._save_checkpoint(final=True)

    # ------------------------------------------------------------
    # 角色选择（stage 3 交替）
    # ------------------------------------------------------------

    def _pick_role_for_step(self) -> Tuple[str, str, str]:
        """返回 (role, active_adapter, opponent_adapter)"""
        cfg = self.config
        if cfg.active_role in ("buyer", "seller"):
            return cfg.active_role, cfg.active_adapter, cfg.opponent_adapter

        # alternating
        if self.global_step % 2 == 0:
            return "buyer", "buyer", "seller"
        return "seller", "seller", "buyer"

    # ------------------------------------------------------------
    # 展平 + advantage
    # ------------------------------------------------------------

    def _build_flat_samples(
        self, groups: List[RolloutGroup], role: str,
    ) -> List[FlatSample]:
        """
        组内 z-score 后，把 active_role 的每个 turn 展开为独立样本。
        同一条 trajectory 的所有 turn 共享同一 advantage。
        """
        cfg = self.config
        flat: List[FlatSample] = []

        for group in groups:
            rewards = [t.advantage_reward for t in group.trajectories]
            if not rewards:
                continue
            mu = sum(rewards) / len(rewards)
            var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
            sigma = math.sqrt(var) + cfg.advantage_eps

            for traj in group.trajectories:
                adv = (traj.advantage_reward - mu) / sigma
                for turn in traj.active_turns:
                    trimmed_history = self.rollout.pb.trim_dialogue_history_to_budget(
                        tokenizer=self.tokenizer,
                        role=role,
                        scenario=traj.scenario,
                        dialogue_history=turn.prompt_dialogue_history,
                        max_prompt_tokens=cfg.max_prompt_length,
                    )
                    prompt_messages = self.rollout.pb.build_messages(
                        role=role,
                        scenario=traj.scenario,
                        dialogue_history=trimmed_history,
                    )
                    p_ids = self.tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    c_ids = turn.completion_token_ids[:cfg.max_completion_length]
                    if not c_ids:
                        continue
                    flat.append(FlatSample(
                        prompt_token_ids=list(p_ids),
                        completion_token_ids=list(c_ids),
                        advantage=adv,
                    ))

        return flat

    # ------------------------------------------------------------
    # logprob 计算
    # ------------------------------------------------------------

    def _compute_logps(
        self,
        samples: List[FlatSample],
        use_adapter: Optional[str],
    ) -> List[torch.Tensor]:
        """
        计算每个样本 completion token 的 log_prob。
        返回: list of 1D tensors，长度 = len(samples)，每个 tensor 长度 = completion_len
        """
        model = self.mgr.peft_model
        device = self.device
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        logps_list: List[torch.Tensor] = []

        ctx = self.mgr.use_no_adapter() if use_adapter is None else self.mgr.use_adapter(use_adapter)
        with ctx:
            # 逐个样本前向（self-play 下每条样本长度不同，简单实现先不做 bucket batching）
            for s in samples:
                input_ids = s.prompt_token_ids + s.completion_token_ids
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                # attention_mask 全 1（假设没 padding）
                attn = torch.ones_like(input_tensor)

                outputs = model(input_ids=input_tensor, attention_mask=attn)
                logits = outputs.logits  # [1, L, V]

                # 预测 token t 的 logits 在位置 t-1
                completion_len = len(s.completion_token_ids)
                prompt_len = len(s.prompt_token_ids)
                # 目标是 completion tokens；logits[prompt_len-1 : prompt_len-1 + completion_len]
                target_logits = logits[0, prompt_len - 1: prompt_len - 1 + completion_len, :]
                target_ids = torch.tensor(s.completion_token_ids, dtype=torch.long, device=device)
                logp = F.log_softmax(target_logits, dim=-1).gather(
                    -1, target_ids.unsqueeze(-1),
                ).squeeze(-1)
                logps_list.append(logp)

        return logps_list

    def _compute_logps_with_grad(
        self,
        samples: List[FlatSample],
        use_adapter: str,
    ) -> List[torch.Tensor]:
        """算 log-prob，保留 grad（给 loss 用）"""
        model = self.mgr.peft_model
        device = self.device

        logps_list: List[torch.Tensor] = []
        with self.mgr.use_adapter(use_adapter):
            for s in samples:
                input_ids = s.prompt_token_ids + s.completion_token_ids
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                attn = torch.ones_like(input_tensor)

                outputs = model(input_ids=input_tensor, attention_mask=attn)
                logits = outputs.logits

                completion_len = len(s.completion_token_ids)
                prompt_len = len(s.prompt_token_ids)
                target_logits = logits[0, prompt_len - 1: prompt_len - 1 + completion_len, :]
                target_ids = torch.tensor(s.completion_token_ids, dtype=torch.long, device=device)
                logp = F.log_softmax(target_logits, dim=-1).gather(
                    -1, target_ids.unsqueeze(-1),
                ).squeeze(-1)
                logps_list.append(logp)

        return logps_list

    def _compute_single_logp_with_grad(
        self,
        sample: FlatSample,
        use_adapter: str,
    ) -> torch.Tensor:
        """对单条样本计算带梯度的 completion log-prob。"""
        model = self.mgr.peft_model
        device = self.device

        with self.mgr.use_adapter(use_adapter):
            input_ids = sample.prompt_token_ids + sample.completion_token_ids
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            attn = torch.ones_like(input_tensor)

            outputs = model(input_ids=input_tensor, attention_mask=attn)
            logits = outputs.logits

            completion_len = len(sample.completion_token_ids)
            prompt_len = len(sample.prompt_token_ids)
            target_logits = logits[0, prompt_len - 1: prompt_len - 1 + completion_len, :]
            target_ids = torch.tensor(sample.completion_token_ids, dtype=torch.long, device=device)
            return F.log_softmax(target_logits, dim=-1).gather(
                -1, target_ids.unsqueeze(-1),
            ).squeeze(-1)

    # ------------------------------------------------------------
    # GRPO loss
    # ------------------------------------------------------------

    def _grpo_step(
        self,
        samples: List[FlatSample],
        old_logps: List[torch.Tensor],
        ref_logps: List[torch.Tensor],
        active_adp: str,
    ) -> Dict[str, float]:
        cfg = self.config
        optimizer = self._optimizers[active_adp]

        total_pg = 0.0
        total_kl = 0.0
        total_tokens = sum(len(s.completion_token_ids) for s in samples)
        mini_bs = max(1, int(getattr(cfg, "policy_mini_batch_size", 1)))

        # 逐个小块 forward/backward，避免把所有样本的计算图同时留在显存里。
        for start in range(0, len(samples), mini_bs):
            chunk_num = 0.0
            chunk_den = 0

            for s, lp_old, lp_ref in zip(
                samples[start: start + mini_bs],
                old_logps[start: start + mini_bs],
                ref_logps[start: start + mini_bs],
            ):
                lp_new = self._compute_single_logp_with_grad(s, use_adapter=active_adp)
                adv = s.advantage

                # Importance ratio
                ratio = torch.exp(lp_new - lp_old)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv
                pg_loss = -torch.min(surr1, surr2)   # per-token

                # KL（Schulman 估计器 k3）：kl ≈ exp(r) - r - 1，r = lp_ref - lp_new
                r = lp_ref - lp_new
                kl = torch.exp(r) - r - 1.0

                per_tok = pg_loss + cfg.beta_kl * kl
                chunk_num = chunk_num + per_tok.sum()
                chunk_den += per_tok.numel()
                total_pg += pg_loss.detach().sum().item()
                total_kl += kl.detach().sum().item()

            if chunk_den > 0:
                chunk_loss = chunk_num / max(total_tokens, 1)
                chunk_loss = chunk_loss / cfg.gradient_accumulation_steps
                chunk_loss.backward()

        self._grad_accum_counts[active_adp] = self._grad_accum_counts.get(active_adp, 0) + 1

        did_step = False
        if self._grad_accum_counts[active_adp] >= cfg.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(
                self._adapter_params[active_adp],
                cfg.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
            self._grad_accum_counts[active_adp] = 0
            did_step = True

        return {
            "loss": (total_pg + cfg.beta_kl * total_kl) / max(total_tokens, 1) / cfg.gradient_accumulation_steps,
            "pg_loss": total_pg / max(total_tokens, 1),
            "kl": total_kl / max(total_tokens, 1),
            "did_optimizer_step": did_step,
        }

    def _flush_pending_optimizer_steps(self) -> None:
        """Step adapters that still have accumulated gradients at train end."""
        for name, pending in list(getattr(self, "_grad_accum_counts", {}).items()):
            if pending <= 0:
                continue
            optimizer = self._optimizers[name]
            torch.nn.utils.clip_grad_norm_(
                self._adapter_params[name],
                self.config.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
            self._grad_accum_counts[name] = 0
            self.logger.log(f"[optimizer] flushed pending gradients for adapter '{name}' ({pending} accumulation step(s))")

    # ------------------------------------------------------------
    # vLLM adapter 同步
    # ------------------------------------------------------------

    def _sync_vllm_adapter(self, adapter_name: str) -> None:
        """
        把最新的 adapter 权重落盘，并通知 vLLM 客户端热重载。

        使用"双缓冲 + 原子切换"策略，避免 vLLM 正在读盘时文件被覆写：
          _live_a/adapter_X  <--  optimizer.step 完成后写入这里
          _live_b/adapter_X  <--  下一次写到这里
        每次交替目标目录，reload_adapter 传新路径 → vLLM 重新加载。
        老目录即使被 vLLM 延后使用也不会损坏。
        """
        try:
            slot = getattr(self, "_vllm_sync_slot", {})
            # 交替目录
            cur = slot.get(adapter_name, "a")
            nxt = "b" if cur == "a" else "a"
            slot[adapter_name] = nxt
            self._vllm_sync_slot = slot

            sync_root = Path(self.config.output_dir) / "_vllm_sync" / adapter_name / nxt
            # 先清空旧内容（同 slot 的上一次残留），再保存
            if sync_root.exists():
                import shutil
                shutil.rmtree(sync_root, ignore_errors=True)
            self.mgr.save_adapter(adapter_name, str(sync_root))

            # 通知 vLLM 重载（VLLMClient.reload_adapter 内部会递增 lora_int_id）
            client = getattr(self.rollout, "client", None)
            if client is not None and hasattr(client, "reload_adapter"):
                client.reload_adapter(adapter_name, str(sync_root))
        except Exception as e:
            # 同步失败不应该中断训练；仅记录
            self.logger.log(f"[vllm_sync] failed to sync '{adapter_name}': {e}")

    # ------------------------------------------------------------
    # 日志 / Eval / Save
    # ------------------------------------------------------------

    def _log_step(
        self,
        role: str,
        groups: List[RolloutGroup],
        flat_samples: List[FlatSample],
        loss_info: Dict[str, float],
        step_time: float,
    ) -> None:
        # 从 groups 汇总成交率等粗指标
        metrics = self._rollout_metrics(role, groups, len(flat_samples), step_time)
        metrics.update({
            "train/loss": loss_info["loss"],
            "train/pg_loss": loss_info["pg_loss"],
            "train/kl": loss_info["kl"],
            "train/did_optimizer_step": int(bool(loss_info.get("did_optimizer_step"))),
            "train/global_step": self.global_step,
        })

        self._emit_metrics(metrics, step=self.global_step)

        if self.global_step % self.config.logging_every == 0:
            self.logger.log(
                f"[step {self.global_step}] role={role} loss={loss_info['loss']:.4f} "
                f"pg={loss_info['pg_loss']:.4f} kl={loss_info['kl']:.4f} "
                f"deal_rate={metrics['rollout/deal_rate']:.2%} "
                f"reward={metrics['rollout/avg_reward']:.2f} "
                f"rounds={metrics['rollout/avg_rounds']:.1f} "
                f"samples={len(flat_samples)} time={step_time:.1f}s"
            )

    def _rollout_metrics(
        self,
        role: str,
        groups: List[RolloutGroup],
        num_samples: int,
        step_time: float,
        prefix: str = "rollout",
    ) -> Dict[str, float]:
        all_trajs = [t for g in groups for t in g.trajectories]
        n = len(all_trajs) or 1
        outcome_counts: Dict[str, int] = {}
        for traj in all_trajs:
            key = getattr(traj.final_state.outcome, "value", str(traj.final_state.outcome))
            outcome_counts[key] = outcome_counts.get(key, 0) + 1

        metrics = {
            f"{prefix}/trajectories": len(all_trajs),
            f"{prefix}/samples": num_samples,
            f"{prefix}/avg_reward": sum(t.advantage_reward for t in all_trajs) / n,
            f"{prefix}/avg_rounds": sum(len(t.final_state.history) for t in all_trajs) / n / 2,
            f"{prefix}/deal_rate": sum(1 for t in all_trajs if t.final_state.outcome.is_deal) / n,
            f"{prefix}/step_time_sec": step_time,
            f"{prefix}/active_is_buyer": int(role == "buyer"),
            f"{prefix}/active_is_seller": int(role == "seller"),
        }
        for name in ("deal", "violation_buyer", "violation_seller", "walkaway", "timeout", "format_error"):
            metrics[f"{prefix}/outcome_{name}_rate"] = outcome_counts.get(name, 0) / n
        return metrics

    def _emit_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self.metric_logger is None:
            return
        try:
            self.metric_logger.log(metrics, step=step)
        except Exception as e:
            self.logger.log(f"[metrics] failed to log metrics: {e}")

    def _evaluate(self) -> None:
        """简单评估：用当前 active adapter 对 val scenarios 做 rollout，计算 reward"""
        if self.val_dataset is None:
            return
        cfg = self.config
        role, active_adp, opp_adp = self._pick_role_for_step()

        # 仅抽取 32 个 val scenario 快速评估
        n_eval = min(32, len(self.val_dataset))
        scenarios = [self.val_dataset[i] for i in range(n_eval)]

        self.mgr.peft_model.eval()
        with torch.no_grad():
            groups = self.rollout.rollout_batch(
                scenarios=scenarios,
                group_size=4,   # eval 时少一点
                active_role=role,
                active_adapter=active_adp,
                opponent_adapter=opp_adp,
            )
        self.mgr.peft_model.train()

        all_trajs = [t for g in groups for t in g.trajectories]
        avg_reward = sum(t.advantage_reward for t in all_trajs) / max(len(all_trajs), 1)
        deal_rate = sum(1 for t in all_trajs if t.final_state.outcome.is_deal) / max(len(all_trajs), 1)
        self.logger.log(f"[eval step {self.global_step}] reward={avg_reward:.2f} deal_rate={deal_rate:.2%}")
        eval_metrics = self._rollout_metrics(
            role=role,
            groups=groups,
            num_samples=0,
            step_time=0.0,
            prefix="eval",
        )
        eval_metrics.update({
            "eval/avg_reward": avg_reward,
            "eval/deal_rate": deal_rate,
            "eval/best_reward": max(self.best_val_reward, avg_reward),
        })
        self._emit_metrics(eval_metrics, step=self.global_step)
        self._save_eval_result(
            metrics=eval_metrics,
            role=role,
            active_adapter=active_adp,
            opponent_adapter=opp_adp,
            n_eval_scenarios=n_eval,
            group_size=4,
        )

        # best ckpt
        if avg_reward > self.best_val_reward:
            self.best_val_reward = avg_reward
            self._save_checkpoint(tag="best")

    def _save_eval_result(
        self,
        metrics: Dict[str, float],
        role: str,
        active_adapter: Optional[str],
        opponent_adapter: Optional[str],
        n_eval_scenarios: int,
        group_size: int,
    ) -> None:
        """Persist each training-time eval result locally."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "step": self.global_step,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "role": role,
            "active_adapter": active_adapter,
            "opponent_adapter": opponent_adapter,
            "n_eval_scenarios": n_eval_scenarios,
            "group_size": group_size,
            "metrics": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in sorted(metrics.items())
            },
        }

        jsonl_path = out_dir / "eval_results.jsonl"
        latest_path = out_dir / "eval_results_latest.json"
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            self.logger.log(f"[eval] saved results to {jsonl_path}")
        except Exception as e:
            self.logger.log(f"[eval] failed to save results: {e}")

    def _save_checkpoint(self, tag: Optional[str] = None, final: bool = False) -> None:
        """
        保存 checkpoint。

        - stage1/2：只保存 active_adapter
        - stage3 alternating：两个 adapter 都保存（它们都在被更新）
        - 每个 adapter 放自己的子目录：<root>/<adapter_name>/
        """
        cfg = self.config
        if final:
            sub = "final"
        elif tag:
            sub = tag
        else:
            sub = f"step_{self.global_step}"

        # 决定要保存的 adapter 列表
        if cfg.active_role == "alternating":
            names = ["buyer", "seller"]
        else:
            names = [cfg.active_adapter]

        for name in names:
            path = Path(cfg.output_dir) / sub / name
            self.mgr.save_adapter(name, str(path))
            self.logger.log(f"[save] adapter '{name}' → {path}")

        # 清理旧 ckpt（保留 save_total_limit 个）
        if not final and tag is None:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        cfg = self.config
        root = Path(cfg.output_dir)
        if not root.exists():
            return
        ckpts = sorted(
            [p for p in root.iterdir() if p.is_dir() and p.name.startswith("step_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        if len(ckpts) <= cfg.save_total_limit:
            return
        for p in ckpts[: len(ckpts) - cfg.save_total_limit]:
            try:
                import shutil
                shutil.rmtree(p)
                self.logger.log(f"[cleanup] removed old ckpt {p}")
            except Exception as e:
                self.logger.log(f"[cleanup] failed to remove {p}: {e}")

    # ------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------

    @staticmethod
    def _infinite_iterator(dataloader):
        while True:
            for batch in dataloader:
                yield batch


# ============================================================
# 简单 Logger
# ============================================================

class _StderrLogger:
    def log(self, msg: str) -> None:
        import sys
        print(msg, file=sys.stderr, flush=True)
