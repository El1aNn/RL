# Result

## Summary

Created a 12-slide final project presentation deck for the RL negotiation project. The deck follows the rubric emphasis on problem-to-method-to-results flow, clear technical trade-offs, and an intuitive demo.

Update: installed a local LaTeX compiler and rebuilt the deck with the provided CUHKSZ Beamer template.

Update: rewrote the Beamer deck narrative to follow the logic of the two docs files:
`docs/sft_grpo_training_plan.md` and `docs/grpo_engineering_log.md`.

Update: added final presentation evidence requested by the user:
one V1 rollout trace, a V1/V2/V3 ablation, and SwanLab reward curves.

## Main Changes

- Generated the ready-to-open PDF deck:
  - `presentation/RL_Final_Project_Presentation.pdf`
- Installed Tectonic LaTeX locally:
  - `/Users/bytedance/.local/bin/tectonic`
- Rebuilt the Beamer PDF:
  - `presentation/slide.pdf`
  - copied to `presentation/RL_Final_Project_Presentation.pdf`
- Reorganized `presentation/slide.tex` around:
  - private-value bargaining problem formulation
  - scenario data and SFT role
  - online self-play GRPO
  - role-specific adapters and trajectory views
  - three-stage curriculum
  - engineering iterations and evidence
  - V1 rollout results and demo
- Added a new asset builder:
  - `scripts/build_final_slide_assets.py`
- Added final result assets:
  - `presentation/assets/swanlab_reward_curve.png`
  - `presentation/assets/v123_ablation_final.png`
  - `presentation/assets/v1_trace_price_path.png`
  - `presentation/assets/swanlab_eval_curves.json`
- Updated the results/demo section to show:
  - SwanLab V1/V2/V3 eval reward curves
  - final-policy V1/V2/V3 ablation
  - a concrete V1 rollout trace and dialogue
- Reused the provided CUHKSZ Beamer template files under:
  - `presentation/`
- Generated an editable Beamer source version:
  - `presentation/slide.tex`
- Added a reproducible build script:
  - `scripts/build_presentation.py`
- Added generated chart assets:
  - `presentation/assets/stage3_v1_eval.png`
  - `presentation/assets/rollout_compare.png`
  - `presentation/assets/selected_surplus.png`

## Validation

- Extracted the rubric text with `pypdf`.
- Verified the ReportLab PDF had 12 pages with `pypdf`.
- Verified the Beamer-compiled replacement PDF has 16 pages with `pypdf`.
- Verified `tectonic --version` works from the project shell.
- Verified the docs-aligned Beamer PDF has 17 pages with `pypdf`.
- Rendered sample pages from the regenerated PDF with `fitz` to check visual layout.
- Fetched V1/V2/V3 reward curves from SwanLab via the local `swanlab.Api` SDK.
- Regenerated and checked the final Beamer PDF after adding trace and ablation slides.
- Selected V1 rollout index 47 from `logs/rollout_selected_50/v1_final_rollouts_50.json` because it is a legal deal with balanced buyer/seller surplus and positive rewards:
  - deal price 1220
  - buyer/seller surplus split 50.8% / 49.2%
  - buyer/seller reward 56.9 / 56.0

## Follow-up

- The current main PDF is now the docs-aligned LaTeX/Beamer-compiled version.
