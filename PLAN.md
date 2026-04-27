# Plan

## Goal

Create a capstone presentation slide deck for the RL final project, following the provided rubric and template, and include a strong V1 rollout example.

## Assumptions

- The project root is the main working folder.
- The template zip contains a PowerPoint deck that should be reused when possible.
- Existing local rollout/evaluation logs are sufficient unless online SwanLab data is required.

## Files

- `RL_Final_Project_Rubric.pdf`
- `Capstone_project_presentation.zip`
- `logs/rollout_selected_50/v1_final_rollouts_50.json`
- New or generated presentation files in the project root.

## Steps

1. Extract and inspect the presentation template and rubric.
2. Inspect project docs, training logs, and V1 rollout results.
3. Select a high-quality V1 rollout example.
4. Generate the presentation deck from the template.
5. Validate the generated deck can be opened/read and record results.
6. Install a local LaTeX compiler and rebuild the presentation with the Beamer template to avoid PDF rendering/encoding issues.
7. Rewrite `presentation/slide.tex` so the narrative follows `docs/sft_grpo_training_plan.md` and `docs/grpo_engineering_log.md`, then recompile the Beamer PDF.
