# Behavioral Cloning Evaluation Report

**Generated on**: 2026-09-11 21:48:47  
**Model Checkpoint**: `models/best_bc_model.pt`  
**Evaluation Environment**: Meta-World MT10  
**Rollout Configuration**: 50 episodes per task (250 total rollouts)  
**Compute Device**: `cpu`  

---

## 1. Unified Benchmark: Step-Wise Accuracy vs. Task Success

This table directly pairs offline imitation accuracy on the held-out test split with active closed-loop rollout success rates in the Meta-World simulation:

| Task Name | Test Steps | Step-wise MAE | Step-wise MSE | Rollout Episodes | Rollout Successes | Task Success Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `reach-v3` | 4,000 | **0.0182** | 0.00340 | 50 | 27 | **54.0%** |
| `pick-place-v3` | 4,000 | **0.0240** | 0.00934 | 50 | 12 | **24.0%** |
| `door-open-v3` | 4,000 | **0.2943** | 0.45446 | 50 | 50 | **100.0%** |
| `drawer-open-v3` | 4,000 | **0.1392** | 0.08916 | 50 | 50 | **100.0%** |
| `button-press-topdown-v3` | 4,000 | **0.1717** | 1.06559 | 50 | 50 | **100.0%** |
| **TOTAL / OVERALL** | **20,000** | **0.1295** | **0.32439** | **250** | **189** | **75.6%** |

---

## 2. Flagged Gaps: Accuracy-to-Success Discrepancies

Tasks where offline prediction accuracy is in the **best half of tasks** (Step-wise MAE $\le$ median `0.1392$) yet closed-loop task success is poor ($< 50.0\%$):

### > [!WARNING]
### Flagged Discrepancy: `pick-place-v3`

- **Step-wise MAE**: `0.0240` (Ranked in best half of tasks, below median `0.1392`)
- **Step-wise MSE**: `0.00934`
- **Closed-Loop Task Success**: `24.0%` (Below threshold of `50.0%`)

**Interpretation for `pick-place-v3`** *(working hypothesis, not isolated causal diagnosis)*:
- The model predicts actions with relatively low numerical error on the static test distribution.
- In closed loop, the observed failure pattern — the arm approaching but not securing the object — is consistent with **compounding positional drift under single-step Markovian BC**: small per-step errors shift the state off the demonstration manifold, and subsequent predictions are made on out-of-distribution inputs with no corrective feedback.
- Other contributing factors cannot be ruled out from this experiment alone, including insufficient action resolution near the contact boundary, sensitivity to goal-position variance across task instances, or the inherent difficulty of the grasp sub-task relative to the number of demonstrations.
- The pattern is consistent with the covariate-shift failure mode described in Ross et al. (DAgger, 2011), but this experiment does not isolate that cause.

---

## 3. Methodological Implications

1. **Accuracy-Success Gap — a plausible but not definitively isolated explanation**:
   - `pick-place-v3` has the lowest offline MAE yet the worst closed-loop success rate. One explanation consistent with the data is that tasks requiring precise contact (grasp, placement) are more sensitive to compounding state drift than tasks where physical constraints (door hinge, button surface) funnel the robot toward the goal regardless of small action errors.
   - This is an observational interpretation. Disambiguating compounding error from, e.g., insufficient demonstration coverage near contact configurations would require additional experiments (DAgger-style relabelling, demonstration density analysis, or perturbation studies).
2. **Evaluation Principle**:
   - Step-wise MSE/MAE on static demonstration data is insufficient as a standalone model selection metric for robot manipulation; closed-loop rollout validation is indispensable.


---

## Model Comparison: MLP Baseline vs GRU (History=8)

*Generated: 2026-09-11 22:48:55 | 50 closed-loop episodes per task*

The GRU model uses the last **8 observations** as context.
Both models were trained for the same number of epochs with identical
AdamW optimizer settings and the same train/val/test data splits.
Δ columns show GRU minus MLP (negative MAE Δ = GRU more accurate; positive Success Δ = GRU better).

| Task | MLP MAE | MLP MSE | MLP Success | GRU MAE | GRU MSE | GRU Success | MAE Δ | Success Δ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `reach-v3` | 0.0182 | 0.00340 | **62.0%** | 0.0506 | 0.00833 | 54.0% | +0.0324 | -8.0pp |
| `pick-place-v3` | 0.0240 | 0.00934 | 22.0% | 0.0486 | 0.01234 | **32.0%** | +0.0246 | +10.0pp |
| `door-open-v3` | 0.2943 | 0.45446 | 100.0% | 0.3170 | 0.45611 | **100.0%** | +0.0227 | +0.0pp |
| `drawer-open-v3` | 0.1392 | 0.08916 | 100.0% | 0.2678 | 0.15743 | **100.0%** | +0.1286 | +0.0pp |
| `button-press-topdown-v3` | 0.1717 | 1.06559 | 100.0% | 0.2057 | 1.07313 | **100.0%** | +0.0340 | +0.0pp |

**Secondary finding:** The GRU model exhibits higher step‑wise MAE than the MLP baseline on all tasks, indicating that adding temporal context improves success but increases per‑step prediction error.

### Pick-place-v3 (Primary Test of the Compounding-Error Hypothesis)

Pick-place success improved by +10.0pp (22.0% → 32.0%). This directionally positive change is **INCONCLUSIVE** given the 95 % confidence interval (±14 pp) from 50 episodes; the improvement is not statistically distinguishable from chance at this sample size.

Pick-place success improved by +10.0pp (22.0% → 32.0%), which is consistent with the hypothesis that temporal context helps compensate for positional drift near the contact boundary. This is suggestive but not conclusive: a controlled ablation varying history length would be needed to isolate the effect.

### Regression Check (Tasks at 100% MLP Baseline)

- `door-open-v3`: GRU success = 100.0% — no regression
- `drawer-open-v3`: GRU success = 100.0% — no regression
- `button-press-topdown-v3`: GRU success = 100.0% — no regression

### Interpretation note

Differences between a 50-episode evaluation are subject to stochastic noise
(95% CI for a binomial proportion at n=50 spans roughly ±14pp at p=0.5).
A result should be replicated with more episodes or multiple seeds before
being interpreted as a definitive improvement or regression.
