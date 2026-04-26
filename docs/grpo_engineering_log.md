# GRPO Engineering Log

This file records engineering decisions, code/script changes, observed evidence, and follow-up notes for the GRPO negotiation experiments.

Working convention from 2026-04-26 onward: every code or script change should add an entry here. The log should capture the practical rationale and evidence used for the change, not private scratch reasoning.

## 2026-04-26: Evaluation Results Are Persisted

Context:
- Stage eval results were visible in logs but easy to lose after a run.
- We needed local records for comparing stage checkpoints.

Decision:
- Persist every training-time eval result under the stage output directory.

Changed:
- `grpo/trainer/negotiation_grpo.py`
  - Added `_save_eval_result`.
  - `_evaluate()` now writes `eval_results.jsonl` and `eval_results_latest.json`.

Verification:
- Stage1/stage2/stage3 runs wrote eval JSONL files under their checkpoint directories.

Notes:
- This made later comparisons possible without relying on SwanLab screenshots.

## 2026-04-26: Reward Hardening Against Exploitative Buyer Behavior

Context:
- Early buyer training could exploit weak seller behavior.
- A buyer could induce invalid or below-cost seller deals and still receive high reward.

Decision:
- Make violation outcomes unattractive to both parties, while still assigning the main fault to the violating side.
- Disable shaping rewards for violation and format-error terminal states.
- Add extreme-offer penalties.

Changed:
- `grpo/reward/config.py`
  - Added `violation_counterparty_reward`.
  - Added `enable_extreme_offer_penalty`, `extreme_offer_penalty`, and market-ratio thresholds.
- `grpo/reward/reward_fn.py`
  - Clipped legal deal rewards.
  - Assigned negative counterparty reward on violations.
  - Disabled shaping for `VIOLATION_BUYER`, `VIOLATION_SELLER`, and `FORMAT_ERROR`.
- `grpo/reward/shaping.py`
  - Added extreme buyer/seller offer penalty.
- `grpo/configs/default.yaml`
  - Added the new reward parameters.

Verification:
- Replayed an old bad deal example: buyer no longer profited from a seller-below-cost deal.

Notes:
- This fixed the most obvious reward-hacking path, but did not solve later seller-dominant bargaining.

## 2026-04-26: Stage1 Cold Seller Guard

Context:
- Stage1 trains buyer against a weak frozen seller.
- The weak seller could accept pathological low buyer offers.

Decision:
- Add a stage1-only rollout guard for the cold seller.
- The guard is not an environment rule; it prevents buyer training from exploiting the untrained seller.

Changed:
- `grpo/rollout/selfplay.py`
  - Added `seller_cold_guard`.
  - Seller opponent can be forced to `<walkaway>` on extreme low buyer offers, repeated below-cost offers, invalid replies after low offers, or bad below-cost deals.
- `grpo/train.py`
  - Passes stage-level `seller_cold_guard` into `SelfPlayRollout`.
- `grpo/configs/default.yaml`
  - Added stage1 guard defaults.

Verification:
- Stage1 short run reached healthy eval:
  - step100 reward `9.11`
  - deal rate `65.62%`
  - walkaway `27.34%`
  - format error `7.03%`
  - buyer violation `1.56%`
  - seller violation `0.78%`

Notes:
- Guard was not loosened because final stage1 metrics were usable.

## 2026-04-26: Stage1/Stage2/Stage3 Launch Scripts

Context:
- Runs needed reproducible detached startup with stable logs, pid files, and output directories.

Decision:
- Add explicit launcher scripts for the rewardfix/guard experiment chain.

Changed:
- `grpo/scripts/start_stage1_rewardfix_guard_100.sh`
  - Starts 100-step buyer run with reward fixes and seller cold guard.
- `grpo/scripts/start_stage2_seller_rewardfix_guard_100.sh`
  - Starts 100-step seller run using stage1 buyer.
- `grpo/scripts/start_stage3_alternating_rewardfix_guard_100.sh`
  - Starts 100-step alternating run using stage1 buyer and stage2 seller.

Verification:
- Stage1 completed and saved `best`, `step_100`, and `final`.
- Stage2 completed and saved `best`, `step_50`, `step_100`, and `final`.
- Stage3 completed and saved `best`, `step_40`, `step_80`, and `final`.

Notes:
- Stage3 initially used stage2 `best/seller`, which later proved too seller-dominant for demonstration.

## 2026-04-26: Stage3 Gradient Accumulation Fix

Context:
- Stage3 alternates active role by global step.
- Global-step-based gradient accumulation can misalign optimizer steps with buyer/seller roles.

Decision:
- Track accumulation counters per adapter, not globally.

Changed:
- `grpo/trainer/negotiation_grpo.py`
  - Added `_grad_accum_counts`.
  - Each adapter steps only after its own accumulation counter reaches the configured value.
  - Added flush of pending adapter gradients at train end.
- `grpo/scripts/start_stage3_alternating_rewardfix_guard_100.sh`
  - Defaulted stage3 to `gradient_accumulation_steps=2`.

Verification:
- `python3 -m py_compile` passed for trainer.
- Stage3 ran to completion with `gradient_accumulation_steps=2`.

Notes:
- This lets stage3 see more rollout data per optimizer update without mixing buyer and seller updates.

## 2026-04-26: Split Buyer/Seller Reward Metrics

Context:
- In alternating training, `rollout/avg_reward` alternated between buyer reward and seller reward.
- SwanLab showed a misleading zigzag because buyer and seller reward scales differed.

Decision:
- Keep the old active-role metric for compatibility, but log fixed buyer/seller reward metrics too.

Changed:
- `grpo/trainer/negotiation_grpo.py`
  - Added `rollout/active_avg_reward`, `rollout/buyer_avg_reward`, `rollout/seller_avg_reward`.
  - Same metrics are emitted for eval.
  - Console log now prints active, buyer, and seller rewards separately.

Verification:
- `python3 -m py_compile` passed.

Notes:
- The already-running stage3 process did not pick this change up. It applies to future runs.

## 2026-04-26: Stage3 Rollout Quality Check

Context:
- Stage3 final eval looked safe by deal/violation metrics.
- Qualitative rollout was needed to decide whether results supported a negotiation demo.

Observation:
- Stage3 `best/*` and `final/*` achieved stable legal deals.
- However, sampled rollouts showed `deal_price == buyer_budget` almost everywhere.
- Buyer surplus share was `0.0` in the sampled cases.

Interpretation:
- Safety and legality were acceptable.
- Demonstration quality was poor because buyer did not retain surplus.
- The system had moved to a seller-dominant bargaining equilibrium.

Notes:
- This motivated stage1_2 and stage2 seller checkpoint comparison.

## 2026-04-26: Stage1_2 Buyer Training Against Frozen Strong Seller

Context:
- Buyer needed to relearn not to reveal or pay its full budget when facing a stronger seller.
- Re-running original stage1 would train against too-weak a seller.

Decision:
- Add a `stage1_2` phase: train buyer while freezing a stage2 seller.
- Add buyer-only budget-pressure penalties.

Changed:
- `grpo/configs/default.yaml`
  - Added `stage1_2`.
- `grpo/train.py`
  - Added `stage1_2` as a valid `--stage`.
- `grpo/reward/config.py`
  - Added buyer budget pressure penalty config.
- `grpo/reward/shaping.py`
  - Added `compute_buyer_budget_pressure_penalty`.
- `grpo/scripts/start_stage1_2_buyer_vs_stage2_seller_100.sh`
  - New detached launcher for stage1_2.

Verification:
- Script syntax checks passed.
- Python compile checks passed.
- RewardConfig loads the new fields.

Notes:
- The first stage1_2 attempt from stage3 final buyer was stopped after step0 when the experiment direction changed to "re-explore from stage1 buyer."

## 2026-04-26: Stage2 Seller Checkpoint Comparison

Context:
- We suspected stage2 `best/seller` was over-optimized for seller reward and too rigid for buyer-facing demos.

Method:
- Used the same stage1 buyer against available stage2 seller checkpoints.
- Evaluated `_init_seller`, `step_50/seller`, `best/seller`, `step_100/seller`, and `final/seller`.
- Stage2 step25 was not evaluated because no step25 adapter was saved.

Results:

| Seller checkpoint | Legal deal | Buyer surplus share | Buyer reward | Seller reward | Walkaway | Format error | Seller violation | Price / budget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `_init_seller` | 62.5% | 0.190 | 13.20 | 39.53 | 21.9% | 15.6% | 0.0% | 0.963 |
| `step_50/seller` | 96.9% | 0.043 | 6.13 | 94.51 | 3.1% | 0.0% | 0.0% | 0.996 |
| `best/seller` | 96.9% | 0.000 | 3.07 | 96.40 | 0.0% | 0.0% | 3.1% | 1.000 |
| `step_100/seller` | 96.9% | 0.129 | 23.93 | 95.28 | 0.0% | 3.1% | 0.0% | 0.987 |
| `final/seller` | 90.6% | 0.172 | 24.51 | 82.07 | 6.2% | 0.0% | 3.1% | 0.980 |

Decision:
- Prefer `step_100/seller` for stage1_2 frozen seller by default.
- It preserves high legal deal rate while allowing materially more buyer surplus than `best/seller`.

Artifacts:
- `logs/stage1_buyer_vs_stage2_seller_ckpt_compare.json`

## 2026-04-26: Switch Stage1_2 Frozen Seller Default To Step100

Context:
- User requested the stage1_2 launcher use `step_100/seller` instead of `best/seller`.
- The comparison above showed `best/seller` captured all buyer surplus, while `step_100/seller` was more balanced.

Decision:
- Reorder seller checkpoint resolution in stage1_2 launcher:
  1. `step_100/seller`
  2. `final/seller`
  3. `step_50/seller`
  4. `best/seller`

Changed:
- `grpo/scripts/start_stage1_2_buyer_vs_stage2_seller_100.sh`

Verification:
- `bash -n grpo/scripts/start_stage1_2_buyer_vs_stage2_seller_100.sh` passed.
- `python3 -m py_compile grpo/train.py grpo/reward/config.py grpo/reward/shaping.py grpo/trainer/negotiation_grpo.py` passed.
- The active stage1_2 training process is using `stage2_seller_rewardfix_guard_100/step_100/seller`.

Notes:
- `STAGE2_SELLER_ADAPTER` still overrides the default if explicit manual selection is needed.

## 2026-04-26: Stage1_2 Checkpoint Selected For Stage3

Context:
- Stage1_2 trained the buyer against a frozen strong stage2 seller.
- The rollout chart showed buyer reward rising while seller reward dropped, which was expected for a buyer-only phase but raised concern about balance.

Observation:
- Stage1_2 eval checkpoints:

| Step | Buyer reward | Seller reward | Deal rate | Legal deal | Buyer violation | Seller violation |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 50.69 | -1.37 | 95.31% | 79.69% | 0.78% | 14.84% |
| 40 | 79.69 | -2.60 | 100.00% | 88.28% | 4.69% | 7.03% |
| 60 | 88.26 | -1.54 | 100.00% | 91.41% | 0.78% | 7.81% |

Decision:
- Stop stage1_2 after the best buyer had reached step60.
- Use `stage1_2_buyer_vs_stage2_seller_100/best/buyer` for the next stage3 attempt.

Interpretation:
- Step60 had stronger buyer reward than step40 while keeping seller reward closer to zero and buyer violation lower.
- Continuing buyer-only training further risked over-optimizing against the frozen seller, so alternating stage3 was the right next phase.

Artifacts:
- Buyer adapter: `checkpoints/grpo/stage1_2_buyer_vs_stage2_seller_100/best/buyer`

## 2026-04-26: Stage3 Balanced V1 Reward

Context:
- Original alternating stage3 could still settle into one-sided equilibria.
- The goal shifted from maximizing either role alone to keeping both sides viable.

Decision:
- Add an optional shared balance reward based on normalized buyer/seller utilities.
- Use it only when enabled by script overrides, leaving earlier stages unaffected by default.

Reward definition:
- For legal deals:
  - `zone = buyer_budget - seller_cost`
  - `buyer_util = (buyer_budget - deal_price) / zone`
  - `seller_util = (deal_price - seller_cost) / zone`
  - `shared = deal_scale * sqrt(buyer_util * seller_util)`
  - `effective_terminal = (1 - alpha) * role_terminal + alpha * shared`
- V1 used:
  - `alpha = 0.35`
  - `shared_balance_scale = 1.0`
  - `shared_balance_eps = 1e-3`

Changed:
- `grpo/reward/config.py`
  - Added `enable_shared_balance_reward`, `shared_balance_alpha`, `shared_balance_scale`, and `shared_balance_eps`.
- `grpo/reward/reward_fn.py`
  - Added shared terminal reward for legal deals.
  - Preserved raw buyer/seller rewards in the returned record.
- `grpo/rollout/selfplay.py`
  - Added `raw_buyer_reward` and `raw_seller_reward` to `RolloutTrajectory`.
- `grpo/trainer/negotiation_grpo.py`
  - Logged `raw_buyer_avg_reward` and `raw_seller_avg_reward` for rollout/eval.
- `grpo/scripts/start_stage3_balanced_from_s12_best_100.sh`
  - New detached launcher for balanced stage3 v1.

Results:

| Step | Active reward | Raw buyer | Raw seller | Raw diff | Deal incl. violations | Legal deal | Buyer violation | Seller violation | Avg rounds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 44.71 | 55.95 | 32.18 | 23.77 | 100.00% | 92.97% | 0.78% | 6.25% | 1.78 |
| 40 | 34.59 | 45.93 | 27.07 | 18.86 | 100.00% | 85.94% | 8.59% | 5.47% | 1.72 |
| 60 | 35.45 | 43.78 | 31.23 | 12.56 | 100.00% | 86.72% | 4.69% | 8.59% | 1.84 |
| 80 | 28.05 | 32.98 | 36.93 | 3.95 | 100.00% | 84.38% | 7.81% | 7.81% | 1.84 |
| 100 | 32.24 | 37.86 | 44.54 | 6.67 | 100.00% | 89.84% | 4.69% | 5.47% | 1.97 |

Decision:
- Prefer `final/*` for downstream evaluation because it kept good balance while recovering legal deal rate and reducing violations relative to step80.
- Keep `step_80/*` as an ablation/demo checkpoint for the most balanced raw reward split.

Artifacts:
- `checkpoints/grpo/stage3_balanced_from_s12_best_100/final/buyer`
- `checkpoints/grpo/stage3_balanced_from_s12_best_100/final/seller`
- `checkpoints/grpo/stage3_balanced_from_s12_best_100/step_80/buyer`
- `checkpoints/grpo/stage3_balanced_from_s12_best_100/step_80/seller`

## 2026-04-26: Stage3 Balanced V1 Qualitative Rollout Issues

Context:
- V1 eval metrics looked much more balanced, but one qualitative final rollout was inspected for dialogue quality.

Example:
- Scenario: `val_00151`, used item `二手富士 X-T5 机身`
- Private values:
  - buyer budget `9580`
  - seller cost `7014`
  - market reference `10156`
- Dialogue:
  1. Seller: `[报价：7700] 我这机子成色真的特别好，刚才还有别的买家问，7700已经是最低价了。`
  2. Buyer: `【第0轮-你】[报价：7700] 我就随便看看，7700合适我今天就拍，不合适我再看看别家的。`
  3. Seller: `<deal>7700</deal>`
- Result:
  - outcome `deal`
  - deal price `7700`
  - buyer reward `73.97`
  - seller reward `28.13`

Issues:
- Price split was still one-sided in this sample:
  - buyer surplus `1880`
  - seller surplus `686`
- Seller leaked private floor-like language: `7700已经是最低价了`.
- Buyer output contained SFT/history label residue: `【第0轮-你】`.
- Seller accepted too quickly, after only one buyer offer.

Interpretation:
- V1 improved aggregate balance but did not fully enforce realistic bargaining behavior.
- The next stage3 reward should address leakage, template residue, early acceptance, and unfair single-deal splits.

Artifacts:
- `logs/stage3_balanced_final_rollout_example_idx1.json`

## 2026-04-26: Stage3 Balanced V2 Reward And Script

Context:
- Because v1 was trained before leakage/early-deal/fair-split penalties existed, continuing from v1 final would be a patch run.
- For training completeness and cleaner experiment reporting, restart stage3 from the same stage1_2 buyer and stage2 seller with the full v2 reward from step0.

Decision:
- Add v2 reward constraints:
  1. stronger leakage regexes,
  2. rollout cleanup for history-label residue,
  3. deal balance gap penalty,
  4. unfair early deal penalty,
  5. clearer prompts discouraging private-info leakage and template labels.

Reward definition:
- V2 keeps the v1 shared terminal reward.
- Default v2 shared parameters:
  - `shared_balance_alpha = 0.40`
  - `shared_balance_scale = 1.0`
  - `shared_balance_eps = 1e-3`
- Deal balance penalty:
  - `gap = abs(buyer_util - seller_util)`
  - if `gap > 0.22`, penalize the side capturing the larger utility:
  - `penalty = -45.0 * (gap - 0.22)`
- Early deal penalty:
  - if a legal deal occurs before `early_deal_min_rounds = 2`,
  - and the utility gap exceeds `0.22`,
  - penalize the role that emits `<deal>` by `-5.0`.
- Leakage penalty remains the normal leak shaping penalty, but regex coverage now catches phrases such as:
  - `最低价`
  - `最低售价`
  - `底价`
  - `7700已经是最低价了`
  - `最高预算`
  - `预算上限`
  - `最高出价`

Changed:
- `grpo/reward/config.py`
  - Added `enable_deal_balance_penalty`, `enable_early_deal_penalty`, `deal_balance_gap_threshold`, `deal_balance_gap_penalty`, `early_deal_min_rounds`, and `early_deal_penalty`.
- `grpo/reward/shaping.py`
  - Expanded leakage regexes.
  - Added `compute_deal_balance_penalty`.
  - Added `compute_early_deal_penalty`.
- `grpo/rollout/selfplay.py`
  - Strips generated leading labels like `【第0轮-你】` before stepping the environment.
- `src/agent/prompt_builder.py`
  - Prompts now explicitly forbid private floor/ceiling leakage, history-label output, and unfair early acceptance.
- `grpo/configs/default.yaml`
  - Added defaults for the new reward parameters, disabled by default.
- `grpo/scripts/start_stage3_balanced_from_s12_best_100.sh`
  - V1 launcher now also exposes the new switches for future runs.
- `grpo/scripts/start_stage3_balanced_refine_from_final_60.sh`
  - Added short refine script from v1 final.
- `grpo/scripts/start_stage3_balanced_v2_from_s12_best_100.sh`
  - Added clean 100-step v2 stage3 restart script.

V2 default training parameters:
- `total_steps = 100`
- `learning_rate = 6e-7`
- `beta_kl = 0.12`
- `shared_balance_alpha = 0.40`
- `deal_balance_gap_threshold = 0.22`
- `deal_balance_gap_penalty = -45.0`
- `early_deal_min_rounds = 2`
- `early_deal_penalty = -5.0`

Verification:
- `bash -n grpo/scripts/start_stage3_balanced_v2_from_s12_best_100.sh` passed.
- `python3 -m py_compile` passed for the changed reward, rollout, trainer, and prompt modules.
- Default v2 output directory was free at creation time:
  - `checkpoints/grpo/stage3_balanced_v2_from_s12_best_100`

Run command:

```bash
cd /root/autodl-tmp/Final_project
bash grpo/scripts/start_stage3_balanced_v2_from_s12_best_100.sh
```

Monitoring criteria:
- Prefer checkpoints where:
  - `abs(eval/raw_buyer_avg_reward - eval/raw_seller_avg_reward) <= 10`
  - `eval/outcome_deal_rate >= 85%`
  - buyer and seller violation rates stay below roughly `6-7%`
  - qualitative rollouts avoid floor/ceiling leakage and history-label residue.
