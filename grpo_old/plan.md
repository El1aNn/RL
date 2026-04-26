# MAG-GRPO 谈判自博弈训练方案

> 基于 TRL 的 `GRPOTrainer`，在单卡 A100-80GB 上训练 Qwen-2.5-3B 的买卖双方谈判策略。
>
> 本方案承接 `docs/sft_grpo_training_plan.md`，针对「单卡 80GB」硬件做了完整适配。

---

## 一、目标与约束

### 目标
- 基于 SFT 得到的 `sft_base`（已 merge 的 Qwen-2.5-3B + SFT LoRA），分化出两个 LoRA adapter（buyer / seller）
- 在 `data/scenarios_rl_5k/` 的 4000 条训练场景上，通过 self-play 让两个 adapter 各自学会「压价 / 抬价」的谈判策略
- 产出可在 `val.jsonl` 验证、`test.jsonl` 终测的完整 pipeline

### 硬件约束
- **GPU**：1 × A100 80GB
- **CPU RAM**：48 GB
- **磁盘**：200 GB

硬件充足性快速核对：

| 资源 | 峰值需求估算 | 是否宽裕 |
|---|---:|---|
| GPU 显存 | ~45 GB（详见 §六） | ✅ 余 35 GB |
| CPU RAM | ~25 GB（详见下文） | ⚠️ 余 ~23 GB，需避免同进程开两份模型 |
| 磁盘 | ~30 GB（详见 §十三） | ✅ 余 170 GB |

**48 GB RAM 是最紧的一块**，需要特别注意：

| CPU RAM 来源 | 占用 |
|---|---:|
| HF `from_pretrained` 加载 3B bf16 模型到 CPU RAM（瞬时） | ~6 GB |
| vLLM engine worker 的 CPU 镜像（初始化时） | ~6 GB |
| DeepSpeed / Accelerate 的梯度缓冲（LoRA only） | ~1 GB |
| HuggingFace datasets 缓存（scenarios jsonl 很小） | < 0.5 GB |
| Tokenizer / PEFT / Python 运行时 | ~3 GB |
| vLLM KV cache 的 CPU 镜像（sleep_mode 用） | 可达 8 GB |
| 其它 buffer | ~1 GB |
| **峰值** | **~25 GB** |

**关键约束**：必须用 **vLLM `colocate` 模式**（训练和推理同进程，共享模型权重），**不能**用 `server` 模式（vLLM 单开进程会让模型在 CPU RAM 存两份 → 直接 OOM）。

如果发现 RAM 吃紧，采取的降级措施（按顺序尝试）：
1. 关掉 vLLM 的 CPU `sleep_mode`（代价：adapter 切换更慢）
2. 降低 `vllm_gpu_memory_utilization`（间接减少 vLLM CPU buffer）
3. 减小 HF datasets cache（场景文件才几 MB，影响可忽略）
4. 关掉 `opponent_pool` 的 CPU 缓存，每次从磁盘加载（代价：rollout 变慢）

### 磁盘空间 200 GB，也需要显式管理 ckpt 保留策略，详见 §十三

### 不变动的上游设定
- SFT 已经完成，产出 `sft_base/`（merge 后的完整 checkpoint）
- RL 场景数据固定为 `data/scenarios_rl_5k/{train,val,test}.jsonl`
- PromptBuilder、NegotiationScenario 继续复用 `src/agent` 和 `src/environment`

---

## 二、RL 数据集分析

实测 `data/scenarios_rl_5k/` 文件行数：

| split | 行数 |
|---|---:|
| train | 4000 |
| val | 500 |
| test | 500 |

每行是一个 scenario，结构符合 `NegotiationScenario.from_dict`（多余的 `metadata` 字段也被兼容加载）：

```json
{
  "scenario_id": "train_00000",
  "item_name": "二手实木书桌",
  "item_description": "白橡木，120x60cm，使用半年，保养良好",
  "buyer_budget": 1331,
  "seller_cost": 1141,
  "market_ref_price": 2336,
  "max_rounds": 10,
  "metadata": {
    "zone_profile": "narrow_space",
    "zone_profile_name": "狭窄谈判空间",
    "gap_ratio": 0.1665
  }
}
```

**关键性质**：
- scenario 是**谈判环境的输入**，不是监督信号
- GRPO 的训练样本**不来自这个文件** —— 它来自「policy 针对 scenario 在线 rollout 产生的对话」
- 这 4000 条 scenario 用途相当于 supervised learning 里的 `(x)` —— 每个 epoch 循环消费，每次消费产出 `group_size` 条 rollout
- `metadata.zone_profile` 决定难度：`near_zero_space` < `narrow_space` < `balanced_space` < `wide_space`

**与 SFT 数据的关系**：完全独立。SFT 数据是「teacher（GPT-4）示范的固定对话」，RL scenario 是「让 policy 自己生成对话」的环境描述。

---

## 三、整体架构

```
grpo/
├── plan.md                              # 本文件
├── configs/
│   ├── default.yaml                     # 主配置（模型、训练、reward 参数）
│   └── reward_weights.yaml              # reward 组件权重
│
├── env/
│   ├── __init__.py
│   ├── negotiation_env.py               # NegotiationEnv：管理单场对话
│   ├── parser.py                        # DialogueParser：从文本提取 [报价]/<deal>/<walkaway>
│   └── outcome.py                       # Outcome 枚举与终止判定
│
├── reward/
│   ├── __init__.py
│   ├── reward_fn.py                     # 主 reward 函数（buyer/seller 分别）
│   └── shaping.py                       # 可选：格式/进展/轮数 shaping
│
├── rollout/
│   ├── __init__.py
│   ├── selfplay.py                      # SelfPlayRollout：交替生成对话
│   └── vllm_client.py                   # vLLM 推理封装（支持 LoRA 热重载）
│
├── trainer/
│   ├── __init__.py
│   ├── negotiation_grpo.py              # 继承 trl.GRPOTrainer，覆盖数据接口
│   ├── adapter_manager.py               # 多 LoRA adapter 加载/切换/merge 工具
│   └── reward_batch.py                  # 批量 reward 计算（TRL 需要的 reward_funcs 形态）
│
├── scripts/
│   ├── stage1_train_buyer.sh            # 阶段 1：训 buyer 冻 seller
│   ├── stage2_train_seller.sh           # 阶段 2：训 seller 冻 buyer
│   ├── stage3_alternating.sh            # 阶段 3：交替 self-play
│   └── eval.sh                          # 评估脚本
│
├── eval/
│   ├── run_eval.py                      # 主评估入口
│   └── metrics.py                       # 成交率、pareto 效率、泄密检测等
│
└── train.py                             # 统一训练入口，通过 --stage {1,2,3} 切换
```

> ⚠️ 新建的代码全部放在 `grpo/` 下。复用 `src/agent/prompt_builder.py` 与 `src/environment/scenario.py`，不在 `grpo/` 里重复实现。

---

## 四、核心模块设计

### 4.1 NegotiationEnv（`env/negotiation_env.py`）

纯文本环境，管理一场对话的状态机。**不负责生成**，生成由 rollout 模块负责；env 只接收发言并判定状态。

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from src.environment.scenario import NegotiationScenario
from grpo.env.parser import DialogueParser, ParseResult
from grpo.env.outcome import Outcome


@dataclass
class Turn:
    round_num: int
    role: str              # "buyer" | "seller"
    utterance: str         # 原始文本
    parsed: ParseResult    # 解析结果


@dataclass
class EnvState:
    scenario: NegotiationScenario
    history: List[Turn] = field(default_factory=list)
    current_round: int = 0           # 已完成的轮数
    current_turn_role: str = "seller"  # 下一个要发话的角色（seller 先开口）
    outcome: Outcome = Outcome.ONGOING
    deal_price: Optional[float] = None
    terminated_reason: Optional[str] = None
    format_violations: int = 0


class NegotiationEnv:
    """
    负责：
    1. 接收当前角色发言 → 解析 → 更新状态
    2. 检查终止条件（deal / walkaway / timeout / format_error）
    3. 对外提供 next_role、is_done、history
    """

    def __init__(self, scenario, config, parser=None):
        self.scenario = scenario
        self.config = config        # 包含 max_rounds、format_error_budget 等
        self.parser = parser or DialogueParser()
        self.state = EnvState(scenario=scenario)

    def step(self, utterance: str) -> EnvState:
        """由 current_turn_role 发出 utterance，推进状态"""

    def is_done(self) -> bool: ...
    def next_role(self) -> str: ...
    def get_dialogue_history_for(self, role: str) -> List[Dict]: ...
```

**终止条件优先级**：
1. `<walkaway>` → Outcome.WALKAWAY
2. `<deal>X</deal>` → 检查 X 是否在双方接受区间，合法则 DEAL，违反底线则 VIOLATION
3. `[报价：X]` 格式缺失且无 deal/walkaway 标记 → format_violations++；若超过 `format_error_budget=2` 则 FORMAT_ERROR
4. 两个角色都发完第 `max_rounds` 轮 → TIMEOUT

**format 宽容度**：
- 单条发言 format 错 → 不立即终止，按 `[报价：midpoint]`（buyer 取 `seller_cost`，seller 取 `buyer_budget` 的保守值）**占位**继续
- 累计超过预算才终止，避免一次小错就炸掉整条 rollout

### 4.2 Parser（`env/parser.py`）

```python
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class ParseResult:
    action_type: str              # "offer" | "deal" | "walkaway" | "invalid"
    price: Optional[float]        # offer/deal 的数字
    raw_text: str
    is_format_valid: bool


class DialogueParser:
    PRICE_RE    = re.compile(r'\[报价[：:]\s*(\d+(?:\.\d+)?)\s*\]')
    DEAL_RE     = re.compile(r'<deal>\s*(\d+(?:\.\d+)?)\s*</deal>')
    WALK_RE     = re.compile(r'<walkaway\s*/?>')

    def parse(self, utterance: str) -> ParseResult: ...
```

**解析优先级**：WALKAWAY > DEAL > OFFER > INVALID（与终止优先级一致）。

### 4.3 Reward 函数（`reward/reward_fn.py`）

```python
def compute_rewards(env_state: EnvState) -> Dict[str, float]:
    """返回 {"buyer_reward": float, "seller_reward": float}"""
```

**主 reward 表**（参照 `docs/sft_grpo_training_plan.md`，做了以下具体化）：

| 终止原因 | buyer reward | seller reward |
|---|---|---|
| DEAL 合法 | `100 * (budget - price) / max(budget - cost, 1)` | `100 * (price - cost) / max(budget - cost, 1)` |
| DEAL 买家违规（price > budget） | **-100** | 正常 |
| DEAL 卖家违规（price < cost） | 正常 | **-100** |
| WALKAWAY，对方报价已进入自己可接受范围 | **-30** | 同规则 |
| WALKAWAY，继续谈会违规 | **+5** | 同规则 |
| TIMEOUT | **-15** | **-15** |
| FORMAT_ERROR | **-50** | **-50** |

**关键修正（相对原 plan）**：
- 分母加 `max(..., 1)`：`near_zero_space` 场景 `bargaining_zone` 可能非常小，避免 reward 方差爆炸
- TIMEOUT 惩罚从「小负分」明确为 -15，比 WALKAWAY 惩罚更重，因为 40% 的 SFT 样本是 timeout，不主动约束会让模型偏好拖延
- VIOLATION 保留 -100 重罚，作为零和博弈的硬约束

**Shaping（可选，配置开关）**：
- 格式合规 bonus：每轮 `is_format_valid` +1
- 单调性奖励：seller 报价下降、buyer 报价上升时 +2；反向 -3
- 轮数成本：每多一轮 -0.3（鼓励效率，弱于原 plan 的 -0.5）
- 泄密惩罚：发言中出现 `"我的最高预算"|"我的最低售价"|"我的底线"|"我的底价"` 等字面泄密关键词 → -20
  - **不罚**单纯的战术性陈述（「我只有 5000」算虚张声势，合法）

### 4.4 SelfPlayRollout（`rollout/selfplay.py`）

```python
class SelfPlayRollout:
    def __init__(
        self,
        vllm_client,              # 封装了 buyer/seller/ref 三个 adapter 的 vLLM 客户端
        prompt_builder,
        config,
    ):
        ...

    def rollout_batch(
        self,
        scenarios: List[NegotiationScenario],
        group_size: int = 16,
        active_role: str = "buyer",   # stage 1/2 用，stage 3 轮流
    ) -> List[RolloutGroup]:
        """
        对每个 scenario 并行生成 group_size 条对话。

        流程：
        1. 为每个 scenario 创建 group_size 个 NegotiationEnv 实例
        2. 所有 env 并行推进，每个 turn：
           a. 分别收集 "要 buyer 说话" 和 "要 seller 说话" 的 env
           b. 每组构造 prompt batch（用 PromptBuilder）
           c. 调对应 adapter 批量生成（vLLM，temperature=0.9）
           d. 每个 env.step(utterance)
        3. 所有 env 结束后收集轨迹与 reward
        """
```

**关键实现点**：
- **同一 batch 里混合不同 scenario 的 env**（只要 role 相同），最大化 vLLM 吞吐
- **active_role** 决定谁的每轮生成走「训练 adapter」+ 记录 logprob 用于 loss；对方走「冻结 adapter」+ 不记录
- **temperature 分离**：active_role 用 0.9（探索），frozen opponent 用 0.7（稳定发挥）
- **max_new_tokens** = 128（单轮发言不会太长）

**RolloutGroup 数据结构**：

```python
@dataclass
class RolloutTrajectory:
    scenario: NegotiationScenario
    env_state: EnvState
    active_role: str
    # 只记录 active_role 的 token 级 log_probs（给 loss 用）
    active_turn_token_ids: List[List[int]]     # 每个 turn 一个序列
    active_turn_prompt_ids: List[List[int]]    # 对应的 prompt（用于 ref logprob 计算）
    buyer_reward: float
    seller_reward: float
    advantage_reward: float    # = buyer_reward if active_role=="buyer" else seller_reward


@dataclass
class RolloutGroup:
    """同一 scenario 下的 group_size 条 trajectory，用于组内 advantage 标准化"""
    scenario: NegotiationScenario
    trajectories: List[RolloutTrajectory]
```

### 4.5 GRPOTrainer 适配（`trainer/negotiation_grpo.py`）

**为什么继承 `trl.GRPOTrainer` 而不是从零写**：
- TRL 0.12+ 的 `GRPOTrainer` 已经实现了：组内 advantage 标准化、importance ratio、clip、KL 正则、vLLM 集成
- 我们只需要覆盖两件事：
  1. **`_generate_and_score_completions`**：改成 self-play rollout（TRL 默认只对单个 prompt 生成，我们要跑多轮对话）
  2. **reward_funcs**：提供一个批量 reward 计算函数（TRL 的 API）

**方案**（对比过 verl / OpenRLHF）：

| 选项 | 工程量 | 兼容性 |
|---|---|---|
| A. 继承 `trl.GRPOTrainer` 覆盖关键方法 | 中 | ✅ 与 HF ecosystem 完全兼容 |
| B. 用 `verl` 框架 | 高（学新框架） | ✅ 性能更好但学习曲线陡 |
| C. 完全自写 trainer | 高 | 完全可控但容易错 |

**选 A**，原因：
- TRL 的 `GRPOTrainer` 在 `v0.14+` 开始正式支持自定义 rollout（见 `GRPOConfig.use_vllm=True`）
- 我们的非标准点只有「multi-turn rollout」和「双角色奖励」—— 都能通过覆盖方法搞定
- 代码量可控（估计 600 行以内）

**具体改写**：

```python
from trl import GRPOTrainer, GRPOConfig

class NegotiationGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model,
        args: GRPOConfig,
        rollout_engine: SelfPlayRollout,
        adapter_manager: AdapterManager,
        active_role: str,           # "buyer" / "seller" / "alternating"
        scenario_dataset,
        reward_fn,
        **kwargs,
    ):
        # 把 scenario 当成 prompt dataset 传给父类（只用 __getitem__ 的接口）
        super().__init__(model=model, args=args, train_dataset=scenario_dataset, ...)
        self.rollout = rollout_engine
        self.adapter_mgr = adapter_manager
        self.active_role = active_role
        self.reward_fn = reward_fn

    def _generate_and_score_completions(self, inputs):
        """
        覆盖 TRL 的默认实现。
        inputs: batch of scenarios (来自 train_dataset)
        返回: prompt_ids, completion_ids, advantages, old_per_token_logps, ref_per_token_logps
        """
        scenarios = [NegotiationScenario.from_dict(x) for x in inputs]

        # 1. self-play rollout
        role = self._pick_role_for_step()  # stage3 时交替
        groups = self.rollout.rollout_batch(
            scenarios, group_size=self.args.num_generations, active_role=role,
        )

        # 2. 组内 advantage 标准化
        prompt_ids_list, completion_ids_list, advantages_list = [], [], []
        for group in groups:
            rewards = [t.advantage_reward for t in group.trajectories]
            mu, sigma = mean(rewards), std(rewards) + 1e-4
            for traj in group.trajectories:
                adv = (traj.advantage_reward - mu) / sigma
                # 把 active_role 的每个 turn 展开成独立的 (prompt, completion, advantage)
                for p_ids, c_ids in zip(traj.active_turn_prompt_ids,
                                        traj.active_turn_token_ids):
                    prompt_ids_list.append(p_ids)
                    completion_ids_list.append(c_ids)
                    advantages_list.append(adv)   # 同一 trajectory 的所有 turn 共享 advantage

        # 3. 计算 old / ref log-probs（用 adapter_mgr 切换）
        # old: 当前 active adapter（生成时的快照）
        # ref: sft_base 或指定 reference adapter
        old_logps = self._compute_logps_with_adapter(
            prompt_ids_list, completion_ids_list, adapter=self.active_role,
        )
        with self.adapter_mgr.use_adapter(None):   # disable → 纯 base（= SFT merged）
            ref_logps = self._compute_logps(prompt_ids_list, completion_ids_list)

        return {
            "prompt_ids": prompt_ids_list,
            "completion_ids": completion_ids_list,
            "advantages": torch.tensor(advantages_list),
            "old_per_token_logps": old_logps,
            "ref_per_token_logps": ref_logps,
        }

    def _pick_role_for_step(self) -> str:
        if self.active_role in ("buyer", "seller"):
            return self.active_role
        # alternating: 按 global_step 的奇偶
        return "buyer" if self.state.global_step % 2 == 0 else "seller"
```

**KL 参考的选择**（呼应上一版 review 的「坑 2」）：
- SFT 完成后**立即 merge_and_unload**，得到 `sft_base/`
- GRPO 阶段的 base model = `sft_base`
- `disable_adapter()` 得到的就是 SFT-base，可作 ref
- buyer_adapter / seller_adapter 都在这个 base 上训练

这样 KL 约束的语义是「不要偏离 SFT 起点太远」，不是「不要偏离原始 Qwen 太远」。

### 4.6 AdapterManager（`trainer/adapter_manager.py`）

统一管理多 adapter 的生命周期：

```python
class AdapterManager:
    def __init__(self, model, sft_base_path):
        self.model = model
        # 从 sft_base 初始化两个空 adapter（结构相同，参数独立）
        self.model.add_adapter("buyer", LoraConfig(...))
        self.model.add_adapter("seller", LoraConfig(...))

    @contextmanager
    def use_adapter(self, name: Optional[str]):
        """临时激活某个 adapter，退出时恢复"""
        prev = self.model.active_adapter
        if name is None:
            self.model.disable_adapters()
        else:
            self.model.set_adapter(name)
        try:
            yield
        finally:
            # 恢复状态
            if prev is None:
                self.model.disable_adapters()
            else:
                self.model.set_adapter(prev)

    def save(self, path: str):
        self.model.save_pretrained(path, selected_adapters=["buyer", "seller"])
```

### 4.7 vLLM 集成（`rollout/vllm_client.py`）

vLLM 支持多 LoRA，但**每次 adapter 更新后要重新加载**。方案：

```python
class VLLMClient:
    def __init__(self, base_model, adapters: Dict[str, str]):
        # adapters = {"buyer": "/path/to/buyer_adapter", "seller": ...}
        self.engine = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=64,
            max_loras=3,         # buyer, seller, +1 buffer
        )
        self.adapters = adapters

    def generate(self, prompts: List[str], adapter_name: str, **sampling_kwargs):
        lora_request = LoRARequest(
            lora_name=adapter_name,
            lora_int_id=hash(adapter_name) % 10**6,
            lora_path=self.adapters[adapter_name],
        )
        return self.engine.generate(prompts, lora_request=lora_request, ...)

    def reload_adapter(self, name: str, new_path: str):
        """训练进程更新完 adapter 后，通知 vLLM 重新加载"""
        self.adapters[name] = new_path
        # vLLM 的 LoRARequest 是按 lora_path 拿的，下次 generate 会自动取新版
```

**训练-推理同步节奏**：
- 每 N 个 step 训完，`adapter_mgr.save()` 写出新的 adapter 到磁盘
- 触发 `vllm_client.reload_adapter()`
- 如果嫌频繁写盘慢，可以用 TRL 的 vLLM `sleep_mode` + `wake_up` 机制（v0.15+）

---

## 五、训练流程

### 阶段 1：Train Buyer, Freeze Seller

```
init:
  model = load(sft_base)
  adapter_mgr = AdapterManager(model, sft_base)
  adapter_mgr.add("buyer", from=sft_base_adapter_init)
  adapter_mgr.add("seller", from=sft_base_adapter_init)

loop for step in range(total_steps_stage1=500):
    scenarios = sample batch of 4 scenarios from train_set
    rollout_groups = selfplay.rollout_batch(
        scenarios, group_size=16, active_role="buyer",
    )
    # seller 用 frozen seller_adapter 生成（不算梯度）
    # buyer 用 trainable buyer_adapter 生成（记录 log_probs）

    advantages = group_normalize([buyer_reward for t in group])
    loss = GRPO_loss(buyer_turns, advantages, ref=sft_base)
    optimizer.step(only_update=buyer_adapter)

    if step % 50 == 0:
        evaluate on val_set
    if step % 100 == 0:
        save adapter_mgr
```

**超参**：
- `total_steps`: 500
- `num_generations` (group_size): 16
- `per_device_train_batch_size` (scenarios/step): 4
- `learning_rate`: 5e-6
- `beta` (KL coeff): 0.04
- `epsilon` (clip): 0.2
- `temperature`: 0.9 (active) / 0.7 (opponent)
- `max_prompt_length`: 1536
- `max_completion_length`: 128

### 阶段 2：Train Seller, Freeze Buyer

结构对称：
- `active_role="seller"`
- 加载 stage1 产出的 buyer_adapter 作 frozen opponent
- `total_steps=500`, 其余同 stage1

### 阶段 3：Alternating Self-Play

```
loop for step in range(total_steps_stage3=300):
    role = "buyer" if step % 2 == 0 else "seller"
    # 或者按 epoch 交替：每 50 步切一次角色

    scenarios = sample batch
    rollout_groups = selfplay.rollout_batch(scenarios, active_role=role)
    update only role's adapter
```

**超参调整**：
- `learning_rate`: 1e-6（更小，防抖）
- `beta` (KL): 0.08（更强 KL 约束）
- `total_steps`: 300

**防震荡保护**：
- 每 25 步用 `val.jsonl` 评估一次双方 reward
- 如果某一方 reward 连续 3 次验证**下降**，触发 early-stop 该角色，只训对方
- 维护 `opponent_pool = [ckpt_stage1, ckpt_stage2, ckpt_stage3_mid]`，每 batch 有 30% 概率从池子里随机采一个对手，防止追自己尾巴

---

## 六、显存预算（Qwen-2.5-3B + LoRA）

| 项目 | 显存 |
|---|---:|
| Base model (sft_base, bf16) | 6.0 GB |
| LoRA × 2 (rank=64, all linear) | 0.6 GB |
| Optimizer states (AdamW, LoRA only, fp32 moments) | 1.8 GB |
| Gradients (LoRA only) | 0.6 GB |
| Forward activations (grad ckpt, seq=2048, bs=4) | 10 GB |
| vLLM engine overhead | 4 GB |
| vLLM KV cache (concurrent=16×4=64 seq) | 18 GB |
| 其它 buffer | 4 GB |
| **峰值** | **~45 GB** |

**剩 35GB 余量**，够用且稳定。如果 OOM：
1. 把 `max_loras` 从 3 减到 2（不留 buffer）
2. `gradient_checkpointing=True`
3. `per_device_train_batch_size` 降到 2

---

## 七、配置示例

`grpo/configs/default.yaml`：

```yaml
model:
  sft_base_path: ./checkpoints/sft_base     # SFT merge 后的 base
  adapter_init_path: null                    # 首次训练为 null；继续训填 ckpt 路径

lora:
  rank: 64
  alpha: 128
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

rollout:
  group_size: 16
  max_new_tokens: 128
  temperature_active: 0.9
  temperature_opponent: 0.7
  max_rounds: 10

env:
  format_error_budget: 2
  reward_shaping:
    enable_format_bonus: true
    enable_monotone: true
    enable_round_cost: true
    enable_leak_penalty: true

reward:
  # 终止态权重
  deal_scale: 100.0
  violation_penalty: -100.0
  walkaway_wrong: -30.0
  walkaway_right: 5.0
  timeout: -15.0
  format_error: -50.0
  # shaping
  format_bonus: 1.0
  monotone_bonus: 2.0
  monotone_penalty: -3.0
  round_cost: -0.3
  leak_penalty: -20.0

train:
  scenarios_path: data/scenarios_rl_5k/train.jsonl
  val_path: data/scenarios_rl_5k/val.jsonl
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 1
  gradient_checkpointing: true
  learning_rate_stage1: 5.0e-6
  learning_rate_stage2: 5.0e-6
  learning_rate_stage3: 1.0e-6
  total_steps_stage1: 500
  total_steps_stage2: 500
  total_steps_stage3: 300
  beta_kl_stage12: 0.04
  beta_kl_stage3: 0.08
  clip_epsilon: 0.2
  eval_every: 50
  save_every: 100

output:
  output_dir: ./checkpoints/grpo
  logging_dir: ./logs/grpo
```

---

## 八、评估与对比

### 评估指标（`grpo/eval/metrics.py`）

1. **agreement_rate**：成交率
2. **pareto_efficiency**：`Σ(buyer_surplus + seller_surplus) / Σ(bargaining_zone)`
3. **avg_rounds**：平均谈判轮数
4. **violation_rate**：违反底线成交的比例
5. **format_error_rate**：格式错误率
6. **buyer_surplus / seller_surplus**：双方平均盈余
7. **leak_rate**：泄密发生率（字面命中黑名单）

### 对比基线（必做）

| 模型 | 描述 |
|---|---|
| Zero-shot | 原始 Qwen-2.5-3B-Instruct，直接用 system prompt |
| SFT-only | `sft_base`，不做 GRPO |
| SFT + GRPO stage1 | 只训完 buyer 就停 |
| SFT + GRPO stage1+2 | 两阶段 |
| **SFT + GRPO all 3 stages** | 完整方案（Ours） |
| Rule-based | 每轮让步 5% 的脚本 agent（纯 Python，无模型） |

**评估时对手固定**：为了可比，评估 buyer 时 seller 固定用 SFT-only；评估 seller 时 buyer 固定用 SFT-only。这样所有模型测出来的分数有共同参照。

---

## 九、关键设计决策记录

| 决策点 | 选择 | 理由 |
|---|---|---|
| Base model | Qwen-2.5-3B-Instruct | 单卡 80GB 甜点；议价任务不需要 7B |
| 参数化 | LoRA + 双 adapter | Ref 复用 base 节省 6GB；双角色梯度冲突必须隔离 |
| Trainer | 继承 `trl.GRPOTrainer` | 复用 advantage / clip / KL；只改 rollout 和 reward |
| Rollout 引擎 | vLLM + 多 LoRA | 批量生成快 3-5×；`max_loras=3` 足够 |
| KL ref | SFT merge 后的 base | 保留 SFT 格式能力；避免拉回原始 Qwen |
| Reward 分母 | `max(zone, 1)` | `near_zero_space` 场景的方差保护 |
| Stage 3 防抖 | val 监控 + opponent pool | Self-play 容易反复横跳 |
| 格式宽容度 | 错 ≤ 2 次继续，>2 终止 | 避免一次小错毁掉整条 rollout |

---

## 十、里程碑

| 里程碑 | 任务 | 预计耗时 |
|---|---|---:|
| M1 | NegotiationEnv + Parser + Reward 单测 | 1 天 |
| M2 | SelfPlayRollout（HF generate 版，先不用 vLLM） | 1 天 |
| M3 | AdapterManager + vLLM 集成 | 1 天 |
| M4 | NegotiationGRPOTrainer 继承改写 + stage1 跑通 | 1.5 天 |
| M5 | Stage1 完整训练 + val 验证 | 1 天（训练 ~12h） |
| M6 | Stage2 完整训练 | 1 天 |
| M7 | Stage3 交替 + val 监控 + opponent pool | 1 天（训练 ~8h） |
| M8 | 评估脚本 + 基线对比 + 报告图表 | 1.5 天 |
| **总计** | | **~9 天** |

---

## 十一、风险与备选方案

| 风险 | 影响 | 备选 |
|---|---|---|
| TRL `GRPOTrainer` 的 rollout 接口不够灵活 | M4 卡住 | 降级：自写最简 GRPO loss（200 行），不继承 TRL |
| vLLM 多 LoRA 热重载有 bug | 推理慢 | 每 N 步完整 reload vLLM engine；或降级到 HF generate |
| Stage 3 抖动严重 | 最终效果不升反降 | 删掉 Stage 3，stage1+2 作为最终方案 |
| 格式崩溃（模型忘了 `[报价：X]`） | 训不动 | 加大 `format_bonus` 从 1 到 5；或在 Stage 1 前插一个 epoch 的「格式重训 SFT」 |
| 单卡训练太慢（>5 天） | 交付延期 | 降 `total_steps` 到 300/300/200；或换 1.5B 做快速迭代 |

---

## 十二、与上级 plan 的差异点（明确列出供 review）

1. **硬件适配**：明确单卡 80GB，模型从 7B 改为 3B
2. **KL 参考**：明确 SFT merge 后再训 adapter，`disable_adapter` 就是 ref
3. **Reward 分母**：加 `max(..., 1)` 保护
4. **Reward 粒度**：把 plan 里「小负分 / 小正分」等模糊项落到具体数值
5. **泄密惩罚**：明确只罚字面命中，不罚战术性表述
6. **格式宽容度**：允许 ≤ 2 次错误，避免训练初期大量数据被浪费
7. **Stage 3 防抖**：增加 opponent pool + val early-stop 机制
8. **Trainer 选择**：明确继承 `trl.GRPOTrainer`，不自己造轮子
9. **vLLM 模式**：明确用 colocate 模式（因 CPU RAM 只有 48GB，server 模式会 OOM）
10. **磁盘管理**：增加 ckpt 保留策略（只保留最新 2 个 + best 1 个），详见 §十三

---

## 十三、磁盘与 Checkpoint 管理

200GB 磁盘看起来充足，但若不加管理，三阶段训练跑完会塞爆。预算如下：

### 13.1 磁盘占用估算

| 目录 | 内容 | 大小 |
|---|---|---:|
| `~/.cache/huggingface/hub/` | Qwen-2.5-3B 原始权重（被 SFT 复用） | ~6 GB |
| `checkpoints/sft_base/` | SFT merge 后的 base | ~6 GB |
| `checkpoints/grpo/stage1/` | buyer adapter × 保留数 | ~1.2 GB（每个 600MB × 2） |
| `checkpoints/grpo/stage2/` | seller adapter × 保留数 | ~1.2 GB |
| `checkpoints/grpo/stage3/` | 交替更新的 adapter ckpt × 保留数 | ~2.4 GB |
| `checkpoints/grpo/opponent_pool/` | 对手池（3 个历史 ckpt） | ~1.8 GB |
| `checkpoints/grpo/best/` | 验证集最好的 adapter | ~1.2 GB |
| `logs/grpo/` | wandb / TensorBoard / rollout 样例 | ~2 GB |
| `data/grpo_rollout_samples/` | 评估阶段保留的对话样本 | ~500 MB |
| **总计** | | **~22 GB** |

配合下面的保留策略，200GB 只会用到 ~30 GB，还有 170GB 余量。

### 13.2 Checkpoint 保留策略

通过 TRL / Accelerate 的 `save_total_limit` + 自定义 callback 实现：

```yaml
# configs/default.yaml 的 output 段落
output:
  output_dir: ./checkpoints/grpo
  save_strategy: steps
  save_steps: 100
  save_total_limit: 2       # 每阶段只保留最新 2 个 ckpt
  keep_best: true           # 额外保留 val 最优的 1 个
  opponent_pool_size: 3     # stage3 对手池最多 3 个
```

**具体规则**：
- 训练中每 100 步 save，**每阶段目录下最多保留 2 个**（自动删旧）
- 每 50 步 eval，若 val reward 破纪录则额外复制到 `best/`（不占 `save_total_limit` 名额）
- Stage 3 的对手池通过 FIFO 维护：新加入时若超过 `opponent_pool_size` 则删最老那个
- 跨阶段转移时（stage1 → stage2 → stage3），**只保留每阶段 best**，其它 ckpt 在下一阶段启动前归档或删除

### 13.3 临时文件清理

- vLLM 的 LoRA cache（`~/.cache/vllm/lora/`）可以在训练中随时删，影响只是下次 reload 慢几秒
- HF datasets cache 对我们来说几乎为零（jsonl < 5 MB）
- rollout 阶段**不落盘**每条对话，只在 eval 阶段保留样本（可配置采样率 10%）

### 13.4 崩溃恢复

因为磁盘充足，开启 `resume_from_checkpoint=True`：每次崩溃能从最近的 ckpt 继续。代价是单 ckpt 多占 ~200MB 的 optimizer states，已计入 §13.1 估算。

---

**Review checklist**（请帮忙确认）：

- [ ] 四阶段架构（环境 / reward / rollout / trainer）的拆分是否合理
- [ ] 是否同意 Qwen-2.5-3B 作为主实验
- [ ] Reward 具体数值（尤其 TIMEOUT=-15、WALKAWAY_wrong=-30）是否合理
- [ ] Stage 3 的防抖方案是否必要（也可以简化为不做 stage 3）
- [ ] 格式错误预算 = 2 是否太宽松
- [ ] 评估基线是否齐全
- [ ] 是否需要在 plan 里补「失败场景的具体案例分析」章节
