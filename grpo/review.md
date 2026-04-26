# Review

## Findings

1. Missing `Final_project.grpo.env` package blocks the GRPO training path.
`SelfPlayRollout` and reward code import `Final_project.grpo.env.negotiation_env` and `Final_project.grpo.env.outcome`, but the repository does not currently contain a `Final_project/grpo/env/` directory. This makes the rollout and reward modules fail at import time before training can start. References: `Final_project/grpo/rollout/selfplay.py:16`, `Final_project/grpo/reward/reward_fn.py:9`, `Final_project/grpo/reward/reward_fn.py:10`, `Final_project/grpo/reward/shaping.py:9`.

2. Stage 2 and Stage 3 adapter restore paths do not match the checkpoint layout written by the trainer.
`_save_checkpoint()` saves adapters under `<output_dir>/<tag>/<adapter_name>/`, but the training scripts pass `./checkpoints/grpo/stage1/best` and `./checkpoints/grpo/stage2/best` as adapter roots. `AdapterManager.load_adapter()` expects the actual adapter directory, so the scripts should point to `.../best/buyer` and `.../best/seller`. References: `Final_project/grpo/trainer/negotiation_grpo.py:585`, `Final_project/grpo/trainer/negotiation_grpo.py:607`, `Final_project/grpo/scripts/stage2_train_seller.sh:10`, `Final_project/grpo/scripts/stage3_alternating.sh:10`, `Final_project/grpo/scripts/stage3_alternating.sh:11`.

3. `alternating` mode has incorrect gradient accumulation semantics when `gradient_accumulation_steps > 1`.
Role selection alternates by global step, but optimizer stepping also uses the same global-step boundary. That means buyer and seller gradients do not accumulate on independent schedules; one adapter can keep stale gradients across the other adapter's turns, and the optimizer step can happen on a boundary that does not correspond to that adapter's own accumulation window. The default config uses `gradient_accumulation_steps: 1`, so this is masked today, but the implementation is not correct once accumulation is enabled. References: `Final_project/grpo/trainer/negotiation_grpo.py:303`, `Final_project/grpo/trainer/negotiation_grpo.py:310`, `Final_project/grpo/trainer/negotiation_grpo.py:472`.

4. The provided training scripts assume a `python` executable instead of an environment-aware launcher.
On the current machine, `python` is not available and only `python3` resolves, so the stage scripts fail before the training code runs. If the project expects a virtual environment, the scripts should either use `python3` or document the required launcher explicitly. References: `Final_project/grpo/scripts/stage1_train_buyer.sh:7`, `Final_project/grpo/scripts/stage2_train_seller.sh:7`, `Final_project/grpo/scripts/stage3_alternating.sh:7`.

5. The colocated rollout design appears to be an intentional freshness-throughput trade-off, but the comments overstate how much memory is actually shared.
`train.py` loads a Hugging Face model for log-prob computation and optimization, while `VLLMClient` separately constructs a vLLM `LLM` from the same base model path. That split is plausibly intentional so rollout can stay on a dedicated vLLM engine and pick up adapter updates through `reload_adapter()` after optimizer steps, rather than an outright correctness bug. The risk is that the current comments describe this as "shared base model weights", which understates the real GPU/host-memory cost of running two model runtimes on a single card and can make OOM risk easier to miss. References: `Final_project/grpo/train.py:101`, `Final_project/grpo/train.py:141`, `Final_project/grpo/trainer/negotiation_grpo.py:493`, `Final_project/grpo/rollout/vllm_client.py:74`, `Final_project/grpo/rollout/vllm_client.py:104`.

## Validation Notes

- Repository inspection confirmed that `Final_project/grpo/env/` is absent.
- A local `python3` import check failed on `Final_project.grpo.rollout.selfplay` and `Final_project.grpo.reward.reward_fn` with `ModuleNotFoundError: No module named 'Final_project.grpo.env'`.
- The same import check also showed that the current environment is missing `peft`, so end-to-end execution was not attempted.

## 中文版

### 主要问题

1. 缺少 `Final_project.grpo.env` 包，导致 GRPO 训练主链路无法启动。
`SelfPlayRollout` 和 reward 相关代码都在导入 `Final_project.grpo.env.negotiation_env` 与 `Final_project.grpo.env.outcome`，但当前仓库里并不存在 `Final_project/grpo/env/` 目录。因此 rollout 和 reward 模块在导入阶段就会直接报错，训练甚至还没开始就会被阻塞。参考：`Final_project/grpo/rollout/selfplay.py:16`，`Final_project/grpo/reward/reward_fn.py:9`，`Final_project/grpo/reward/reward_fn.py:10`，`Final_project/grpo/reward/shaping.py:9`。

2. Stage 2 和 Stage 3 的 adapter 恢复路径与 trainer 实际保存的 checkpoint 目录结构不一致。
`_save_checkpoint()` 实际把 adapter 保存到 `<output_dir>/<tag>/<adapter_name>/`，但训练脚本传入的却是 `./checkpoints/grpo/stage1/best` 和 `./checkpoints/grpo/stage2/best` 这一级目录。`AdapterManager.load_adapter()` 需要的是真正的 adapter 目录，所以这里应当传 `.../best/buyer` 和 `.../best/seller`。参考：`Final_project/grpo/trainer/negotiation_grpo.py:585`，`Final_project/grpo/trainer/negotiation_grpo.py:607`，`Final_project/grpo/scripts/stage2_train_seller.sh:10`，`Final_project/grpo/scripts/stage3_alternating.sh:10`，`Final_project/grpo/scripts/stage3_alternating.sh:11`。

3. 当 `gradient_accumulation_steps > 1` 时，`alternating` 模式的梯度累积语义是不正确的。
当前实现按全局 step 交替 buyer/seller 角色，同时也按同一个全局 step 边界决定何时执行 optimizer step。这会导致 buyer 和 seller 没有各自独立的累积窗口：一个 adapter 的梯度可能跨越另一个 adapter 的训练轮次继续残留，最终在并不对应自身累积边界的时刻被更新。默认配置里 `gradient_accumulation_steps: 1`，所以暂时被掩盖了，但一旦打开梯度累积，逻辑就不再正确。参考：`Final_project/grpo/trainer/negotiation_grpo.py:303`，`Final_project/grpo/trainer/negotiation_grpo.py:310`，`Final_project/grpo/trainer/negotiation_grpo.py:472`。

4. 提供的训练脚本假设环境里存在 `python` 可执行文件，而不是使用更稳妥的环境感知启动方式。
在当前机器上，`python` 不存在，只有 `python3` 可用，因此这些 stage 脚本会在进入训练逻辑之前就直接失败。如果项目依赖虚拟环境，脚本至少应该统一改成 `python3`，或者明确约定启动命令。参考：`Final_project/grpo/scripts/stage1_train_buyer.sh:7`，`Final_project/grpo/scripts/stage2_train_seller.sh:7`，`Final_project/grpo/scripts/stage3_alternating.sh:7`。

5. 当前 colocated rollout 更像是为“adapter 新鲜度和 rollout 吞吐”做的设计取舍，但注释对内存共享程度的表述过强。
`train.py` 里先加载了一份 Hugging Face 模型用于 log-prob 计算和参数更新，而 `VLLMClient` 又基于同一个 base model path 单独构建了一份 vLLM `LLM`。这种拆分很可能是有意为之：让 rollout 始终跑在独立的 vLLM engine 上，并在 optimizer step 后通过 `reload_adapter()` 尽快看到最新 adapter，而不是一个纯粹的实现错误。真正需要指出的是，当前注释把它描述成“共享 base 权重”，这会弱化单卡下同时维护两套模型运行时的真实 GPU/CPU 内存成本，也更容易低估 OOM 风险。参考：`Final_project/grpo/train.py:101`，`Final_project/grpo/train.py:141`，`Final_project/grpo/trainer/negotiation_grpo.py:493`，`Final_project/grpo/rollout/vllm_client.py:74`，`Final_project/grpo/rollout/vllm_client.py:104`。

### 验证说明

- 已检查仓库目录，确认 `Final_project/grpo/env/` 当前不存在。
- 已用本地 `python3` 做最小导入验证，`Final_project.grpo.rollout.selfplay` 和 `Final_project.grpo.reward.reward_fn` 都会报 `ModuleNotFoundError: No module named 'Final_project.grpo.env'`。
- 同一次导入验证还显示当前环境缺少 `peft`，因此没有继续做端到端训练执行测试。
