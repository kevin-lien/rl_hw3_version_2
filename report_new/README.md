# Short report and result sources

Open `report_new.pdf`, or upload `report_new.tex`, `student.sty`, and
`mc_comparison.png` together to Overleaf. Select `report_new.tex` as the main document.
The course template's `student.sty` is included unchanged.

This revision uses short explanations and keeps only the figure required by Q2.
The two Astro Party figures in the request and the extra DQN learning curves are removed.
Q5 uses game counts in a table: the requirement asks for clean experiments, not W&B screenshots.

## Where the numbers came from

All paths below are relative to the repository root. The source snapshot was
`e0ef3105dc436bdded59d3a3a7b340f1616d6cb1`.

| Report item | Experiment script | Saved evidence |
| --- | --- | --- |
| Q1 MC method, Q2 scatter plot and gaps | `2048/report_experiments.py` (`mc`, `main`) | `report_work/results/mc_dqn.json`, `mc_ddqn.json`; each contains 12 states and all 64 returns per state |
| Q2 equal 40k-step budgets | `2048/report_experiments.py` | `report_work/results/2048_algorithm_comparison.json`, `comparison_dqn.zip`, `comparison_ddqn.zip` |
| Q3 architecture and recipe | `2048/train_2048.py`, `2048/envs/train_2048_env.py` | `report_work/results/2048_final.json`; checkpoint metadata has 360,000 steps |
| Q3 public score and best tile | original evaluator | `report_work/stock_2048.log`, and individual games in `2048_final.json` |
| Q4 controller, imitation and PPO settings | `astro_party/train_astro_party.py`, `b13901072/train_astro_party_env.py` | default defense configuration; `report_work/stock_astro_party.log` for the public result |
| Q5 PPO/A2C comparison | `astro_party/experiments/report_algorithm_comparison.py` | `report_work/results/astro_algorithm_comparison.json`: six runs, each with 60 game outcomes |

## What was verified in this revision

- Recalculated the MC means, 95% intervals and mean prediction gaps from individual saved returns.
- Checked that all 24 comparison states have 64 returns and no capped rollout.
- Recounted every Q5 win from individual games and checked the 30 evaluation seeds per opponent.
- Checked the 2048 mean score and best tile from its 100 saved games.
- Compared the report's settings with the experiment and training scripts.
- Redrew the required scatter plot directly from the checked JSON; no numbers were invented.

These checks establish consistency between the saved data, scripts and report.
Training and game evaluation were **not rerun** during this revision; PyTorch and
SB3 are not installed in this editing environment. Saved logs alone cannot independently
prove the historical execution of every run.

## Why the old graphs looked that way

`report_work/build_report.py` generated all four old figures with Matplotlib:

| Old figure | Actual input | Decision |
| --- | --- | --- |
| `mc_q.png` | `mc_dqn.json`, `mc_ddqn.json`, `mc_final_ddqn.json` | Replaced with two matched-budget panels; the longer-trained model is omitted |
| `dqn_learning.png` | `report_work/results/tb/` TensorBoard Q means and losses | Removed; these are not the fixed-state MC comparison |
| `astro_learning.png` | `astro_party/results/6o6gk5el/history.json` plus TensorBoard value loss | Removed; one PPO run does not compare algorithms |
| `astro_algorithms.png` | `astro_algorithm_comparison.json` | Removed; it repeats the Q5 table |

The old graph numbers are traceable to these saved files. They are not W&B screenshots.
The PPO/A2C script initially logs offline. A run ID in the old report is not, by itself,
verification that a live W&B page contains the same data; no live W&B page was checked here.
You do not need to retrain just to replace a graph with a W&B screenshot.

## Limits to understand before submission

- Q2 is one training seed and 12 states per policy. The policies visit different boards.
- MC uses shaped training rewards and the training illegal-move rule, not official score.
  The foul counter is restored for rollouts but is not part of the network input.
- The original 2048 run did not fix its training seed. Its checkpoint does not store the
  historical environment source; the report describes the current recipe, not proof of
  every historical reward version. Its training checks also reseed on reset.
- Q4 uses substantial hand-written controller logic at test time. The short wording does
  not mean the implementation is simple or that PPO learned all tactics by itself.
- Q5 is an existing short fine-tuning experiment from one shared BC checkpoint.
  PPO has more optimizer work. It supports the measured performance hypothesis, but
  cannot isolate clipping as the cause. A clipping/no-clipping ablation would be needed
  for that stronger claim. No new ablation result is claimed.
- The course slide warns about LLM-generated reports. The assistance disclosure remains.
  Read and revise the report yourself and follow your course's rules.

## Rebuild the figure and PDF

From the repository root (Python requires NumPy and Matplotlib):

```bash
python report_new/verify_and_plot.py
cd report_new
pdflatex -interaction=nonstopmode -halt-on-error report_new.tex
pdflatex -interaction=nonstopmode -halt-on-error report_new.tex
```

The verification script only reads saved results and redraws the figure; it does not
train, evaluate a policy, or upload data to W&B.

To repeat the original experiments, use a separate checkout with the repository's
dependencies and checkpoints. From `2048/`, run `python report_experiments.py`.
From `astro_party/`, run `PYTHONPATH=. WANDB_MODE=offline python experiments/report_algorithm_comparison.py`.
These commands train models and overwrite their existing result JSON files; save a copy first.
