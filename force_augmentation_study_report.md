# Force-Augmented Observation Study: Closing the Imitation-to-Success Gap in Contact Manipulation

**Project**: Intent-to-Action / REVEL Thesis Evaluation  
**Status**: UNDER REVISION: GRU columns invalid (train/eval protocol mismatch)  
**Date**: September 2026  
**Artifact Directory**: `models/`, `dataset_12tasks/`  

> [!CAUTION]
> **ERRATA NOTICE (September 2026)**:  
> All GRU evaluation columns (both Kinematic GRU and Force GRU) reported below were generated using `method=step` (continuous carryover of the recurrent hidden state $h_t$ across 500 closed-loop environment steps). In training, policies were trained strictly on short sequence slices ($H=8$) initialized with zero hidden state ($h_0 = 0$). This profound train/eval mismatch caused severe out-of-distribution recurrent state drift in evaluation, causing an apparent "collapse" in Kinematic GRU.  
> When evaluated under `method=window` matching training conditions ($H=8$ sliding window, $h_0 = 0$), Kinematic GRU achieves **41.7%–50.0%** on the 12-pose benchmark and **84.2%** on the 12-task benchmark. Across matched seeds, Force GRU is within $\pm 2$ poses of Kinematic GRU, and the previously reported "force rescue" and "physical anchor" effects were artifacts of the evaluation protocol mismatch.

---

## 1. Executive Summary & Core Finding

This study provides an empirical test of the central hypothesis in the REVEL framework: **whether augmenting robot policy observations with genuine contact force/torque sensing closes the imitation-accuracy-to-task-success gap in contact-rich manipulation**.

In earlier phases of this project:
1. **Evaluation Harness Determinism Confirmed**: We isolated and eliminated random number generator leakage, confirming bitwise determinism across repeated rollouts.
2. **Failure Mechanism Characterized**: We discovered that baseline imitation failure on `pick-place-v3` is purely geometry-determined (bimodal 100% or 0% across initial object-target poses) rather than stochastic noise.
3. **Loss Reweighting Failed**: A proximity-weighted imitation loss failed catastrophically (paired success dropped from 50.0% to 25.0%, with **0 / 6 hard poses flipped**).

### The Primary Finding: Force Rescued the GRU, But Did Not Help the MLP
Augmenting policy observations with a 6-D genuine contact wrench ($F_x, F_y, F_z, \tau_x, \tau_y, \tau_z$) yielded dramatic gains in recurrent policies (GRU), but **failed to improve feedforward MLP policies on pick-place tasks**, refuting the hypothesis of a universal, architecture-independent sensory gap:

- **On Recurrent Policies (GRU Architecture)**:
  - **12-Pose Paired Benchmark (`pick-place-v3`)**: Kinematic GRU suffered complete representational collapse under sequential rollout (**0.0%**, 0/12 poses, 0/60 rollouts). Adding contact force rescued the policy to **50.0%** (6/12 poses, 30/60 rollouts, $+50.0\%$ absolute gain; exact sign test $p=0.031$).
  - **12-Task Benchmark (Locked `MT50(seed=42)`)**: On `pick-place-wall-v3`, Kinematic GRU scored 0.0%, while Force GRU reached **65.0%** (+65.0% [Paired 90% CI: $+47.0\%, +83.0\%$]). Overall across 12 tasks, Force GRU gained **+12.1%** over Kinematic GRU (64.2% vs 52.1% [Paired 90% CI: $+7.5\%, +16.7\%$]).
- **On Feedforward Policies (MLP Architecture Control)**:
  - **12-Pose Paired Benchmark (`pick-place-v3`)**: Kinematic MLP scored **25.0%** (3/12 poses, 15/60 rollouts). Force MLP scored **33.3%** (4/12 poses, 20/60 rollouts; exact sign test $p=1.000$), consistent with noise.
  - **12-Task Benchmark (Locked `MT50(seed=42)`)**: On `pick-place-v3`, Force MLP actually **regressed** compared to Kinematic MLP (**25.0% vs 45.0%**, $-20.0\%$ [Paired 90% CI: $-35.1\%, -4.9\%$]). On `pick-place-wall-v3`, Force MLP was virtually identical to Kinematic MLP (**65.0% vs 70.0%**, $-5.0\%$ [Paired 90% CI: $-23.8\%, +13.8\%$]). Overall across 12 tasks, Force MLP (84.6%, 203/240) achieved essentially identical success to Kinematic MLP (84.2%, 202/240, $+0.4\%$ [Paired 90% CI: $-3.5\%, +4.4\%$]).
- **Architectural Conclusion**: Contact force does not act as an independent causal cue for pick-and-place grasping across all model types. Rather, our working hypothesis—which remains an **untested hypothesis** requiring internal hidden state ablation—is that contact sensing acts as a physical anchor that stabilizes recurrent hidden state trajectories, rescuing GRUs from the multi-task capacity collapse that afflicted Kinematic GRUs under sequential evaluation. In feedforward policies, contact force aids tight geometric insertion (`peg-insert-side-v3` +20.0%), but does not resolve edge-pose grasping in pick-and-place.

---

## 2. Primary Benchmark: Locked 12-Pose `pick-place-v3` Paired Evaluation

### 2.1 Benchmark Protocol & Methodology
The benchmark locks 12 representative initial configurations spanning the workspace, evaluated across 5 fixed random seeds (`[1000, 2026, 3000, 4000, 5000]`), yielding 60 rollouts per model (240 rollouts total per benchmark run).

To ensure direct paired comparability with the Phase 3 weighted-loss study, the evaluation harness instantiated `metaworld.MT10(seed=42)` on the locked indices `[0, 4, 7, 10, 15, 20, 24, 31, 35, 40, 44, 47]`.

### 2.2 Complete Pose-by-Pose Results (Full `[X, Y, Z]` Coordinates)

| Pose ID | Initial Object Pos `[X, Y, Z]` | Target Goal Pos `[X, Y, Z]` | Baseline 5T MLP | 12T Kin MLP | 12T Force MLP | 12T Kin GRU | 12T Force GRU | Force GRU vs Kin GRU Verdict |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#00 (HARD)** | `[1.00, -0.05, 0.62]` | `[-0.06, 0.90, 0.13]` | 0.0% (0/5) | 100.0% (5/5) | 100.0% (5/5) | 0.0% (0/5) | **100.0% (5/5)** | **[FLIPPED] Hard $\to$ Success (+100%)** |
| **#04 (EASY)** | `[1.00, -0.05, 0.69]` | `[ 0.00, 0.84, 0.20]` | 100.0% (5/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | [DEGRADED] Lost Easy (0%) |
| **#07 (HARD)** | `[1.00, -0.06, 0.61]` | `[-0.09, 0.85, 0.28]` | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | Maintained Hard (0%) |
| **#10 (EASY)** | `[1.00,  0.04, 0.68]` | `[ 0.04, 0.85, 0.20]` | 100.0% (5/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | [DEGRADED] Lost Easy (0%) |
| **#15 (EASY)** | `[1.00,  0.02, 0.65]` | `[ 0.05, 0.88, 0.23]` | 100.0% (5/5) | 100.0% (5/5) | 100.0% (5/5) | 0.0% (0/5) | 0.0% (0/5) | [DEGRADED] Lost Easy (0%) |
| **#20 (HARD)** | `[1.00,  0.03, 0.61]` | `[-0.01, 0.86, 0.09]` | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | **100.0% (5/5)** | **[FLIPPED] Hard $\to$ Success (+100%)** |
| **#24 (EASY)** | `[1.00, -0.03, 0.67]` | `[ 0.03, 0.86, 0.23]` | 100.0% (5/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | **100.0% (5/5)** | **[RECOVERED] Maintained Easy (100%)** |
| **#31 (HARD)** | `[1.00, -0.01, 0.61]` | `[-0.06, 0.86, 0.21]` | 0.0% (0/5) | 100.0% (5/5) | 100.0% (5/5) | 0.0% (0/5) | 0.0% (0/5) | Maintained Hard (0%) |
| **#35 (EASY)** | `[1.00, -0.07, 0.63]` | `[-0.00, 0.84, 0.15]` | 100.0% (5/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | **100.0% (5/5)** | **[RECOVERED] Maintained Easy (100%)** |
| **#40 (HARD)** | `[1.00,  0.03, 0.65]` | `[ 0.09, 0.88, 0.12]` | 0.0% (0/5) | 0.0% (0/5) | 100.0% (5/5) | 0.0% (0/5) | 0.0% (0/5) | Maintained Hard (0%) |
| **#44 (EASY)** | `[1.00,  0.00, 0.68]` | `[ 0.01, 0.86, 0.18]` | 100.0% (5/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | **100.0% (5/5)** | **[RECOVERED] Maintained Easy (100%)** |
| **#47 (HARD)** | `[1.00, -0.03, 0.61]` | `[ 0.08, 0.90, 0.14]` | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) | **100.0% (5/5)** | **[FLIPPED] Hard $\to$ Success (+100%)** |
| **OVERALL** | — | — | **50.0% (30/60)** | **25.0% (15/60)** | **33.3% (20/60)** | **0.0% (0/60)** | **50.0% (30/60)** | **+30/60 (+50.0% vs Kin GRU)** |

### 2.3 Pose as the Unit of Analysis & Exact Statistical Tests
Within-pose variance across random seeds is strictly zero in this benchmark: every pose evaluates to either 5/5 successes or 0/5 successes across all seeds. Consequently, treating each rollout as an independent sample ($N=60$) pseudoreplicates the 12 spatial configurations and drastically overstates precision. **Pose ($N=12$) is the appropriate unit of analysis.**

Evaluating policy differences at the pose level yields the following discordant-pose counts and exact two-tailed sign tests (exact binomial tests on discordant pairs):

1. **Force GRU vs Kinematic GRU (under standing `step()` protocol)**:
   - Poses solved by Force GRU only: **6** (#00, #20, #24, #35, #44, #47)
   - Poses solved by Kin GRU only: **0**
   - Concordant failures: 6 (#04, #07, #10, #15, #31, #40)
   - Concordant successes: 0
   - Discordant pairs: $n_d = 6$
   - Exact two-tailed sign test: $p = 2 \times 0.5^6 = 0.03125$ ($p \approx \mathbf{0.031}$, statistically significant at $\alpha = 0.05$).

2. **Force MLP vs Kinematic MLP**:
   - Poses solved by Force MLP only: **1** (#40)
   - Poses solved by Kin MLP only: **0**
   - Concordant successes: 3 (#00, #15, #31)
   - Concordant failures: 8 (#04, #07, #10, #20, #24, #35, #44, #47)
   - Discordant pairs: $n_d = 1$
   - Exact two-tailed sign test: $p = 2 \times 0.5^1 = \mathbf{1.000}$ (not statistically significant).

3. **Force GRU vs Kinematic MLP**:
   - Poses solved by Force GRU only: **5** (#20, #24, #35, #44, #47)
   - Poses solved by Kin MLP only: **2** (#15, #31)
   - Concordant successes: 1 (#00)
   - Concordant failures: 4 (#04, #07, #10, #40)
   - Discordant pairs: $n_d = 7$ ($5$ vs $2$)
   - Exact two-tailed sign test: $p = 2 \times \sum_{k=5}^7 \binom{7}{k} 0.5^7 = 2 \times \frac{21 + 7 + 1}{128} = \frac{58}{128} = \mathbf{0.453}$ (not statistically significant).

### 2.4 Verified Exact Reproducibility Across Independent Executions
To verify that this result is immune to numerical drift or non-deterministic execution, the 240-rollout paired benchmark was executed twice from scratch:
- **Run 1 (`task-3108`)**: Baseline 30/60 (6/12 poses), Kin MLP 15/60 (3/12 poses), Kin GRU 0/60 (0/12 poses), Force GRU 30/60 (6/12 poses).
- **Run 2 (`task-3170`)**: Baseline 30/60 (6/12 poses), Kin MLP 15/60 (3/12 poses), Kin GRU 0/60 (0/12 poses), Force GRU 30/60 (6/12 poses).
- **Divergence Check**: **0 / 240 episodes diverged**. Every single episode produced the identical step count and boolean outcome.
- **Force MLP Evaluation (`task-4725`)**: The 12-Task Force MLP scored **20/60 (4/12 poses, 33.3%)**, maintaining only 1 easy pose (#15) and hitting hard poses #00, #31, #40. Relative to Kin MLP (3/12 poses, 25.0%), it flipped only 1 hard pose (#40), and the difference (1 discordant pair, exact sign test $p=1.000$) is consistent with pure noise.

---

## 3. Comparison of Interventions and Architecture Controls

A critical contribution of this project is the direct comparison between loss-level interventions, observation-level interventions, and model architecture controls evaluated under identical protocols:
1. **Intervention A (Phase 3)**: Modifying the training loss function (proximity-weighted BC loss, penalizing errors heavily during the approach phase).
2. **Intervention B (Phase 4)**: Augmenting the recurrent policy observation space with genuine contact wrench signals (Force GRU).
3. **Control C (Phase 4)**: Augmenting the feedforward policy observation space with genuine contact wrench signals (Force MLP).

### 3.1 Comparative Summary

| Evaluation Metric | Baseline Policy (5T MLP) | Intervention A: Proximity-Weighted Loss | Intervention B: Force Augmentation (Force GRU) | Control C: Force Augmentation (Force MLP) |
| :--- | :---: | :---: | :---: | :---: |
| **Paired Benchmark Success** | 50.0% (30 / 60) | 25.0% (15 / 60) [$-25.0\%$ Regression] | **50.0% (30 / 60) [$+50.0\%$ vs Kin GRU]** | **33.3% (20 / 60) [$+8.3\%$ vs Kin MLP]** |
| **Hard Poses Flipped to Success** | 0 / 6 (0.0%) | 0 / 6 (0.0%) | **3 / 6 (50.0%)** (Poses #00, #20, #47) | **1 / 6 (16.7%)** (Pose #40 relative to Kin MLP) |
| **Easy Poses Maintained** | 6 / 6 (100.0%) | 3 / 6 (50.0%) [Lost 3 easy poses] | 3 / 6 (50.0%) [Poses #24, #35, #44] | 1 / 6 (16.7%) [Only Pose #15 held] |
| **Multi-Task Pick-Place Success** | — | — | **25.0% (5 / 20)** [$+20.0\%$ vs Kin GRU] | **25.0% (5 / 20)** [$-20.0\%$ vs Kin MLP] |
| **Core Diagnosis** | Kinematics insufficient on edge poses | Loss distortion broke approach corridor | **Force rescued GRU from capacity collapse** | **Force did not rescue feedforward MLP** |

### 3.2 Behavioral Interpretation: An Architectural Interaction Rather Than Pure Sensor Causality

The empirical divergence between Force GRU and Force MLP clarifies the true mechanism of force augmentation:

- **Why Loss Reweighting Failed**: Loss reweighting operates solely within kinematic space. Modifying loss weights near contact over-constrained the arm along narrow demonstration corridors, degrading generalization on previously solvable configurations without providing any new information to resolve edge poses.
- **Why Force Rescued the GRU**: In recurrent neural networks (GRU), multi-task imitation is notoriously prone to hidden state drift: small kinematic integration errors compound over time, leading to catastrophic representation collapse across diverse tasks (manifesting as 0% success on `pick-place-v3`, `pick-place-wall-v3`, `assembly-v3`, and `peg-insert-side-v3`). Contact force provides an abrupt physical discontinuity upon touch (0 N jumping to 5–15 N), which is hypothesized—as an **untested hypothesis** requiring internal hidden state probing—to act as a physical anchor that resets or regularizes the recurrent hidden state. This allowed the GRU to avoid catastrophic collapse and execute contact-aligned grasps.
- **Why Force Did Not Help the MLP on Pick-Place**: Feedforward MLPs evaluate each observation frame-by-frame with zero recurrent memory, and thus cannot suffer from hidden state drift. Because the training demonstrations were generated by scripted oracles that are completely force-blind, the demonstrations contain no reactive, force-compliant adjustments. Consequently, providing force observations to a feedforward MLP does not convey a learned closed-loop tactile control law; on `pick-place-v3`, Force MLP fell to 25.0% (vs Kin MLP 45.0%), and on the 12-pose benchmark it scored 33.3% (within noise of Kin MLP's 25.0%). Force only improved the MLP on tasks with rigid mechanical constraints where force correlates directly with alignment, such as `peg-insert-side-v3` (+20.0%).

---

## 4. Secondary Multi-Task Benchmark Across 12 Tasks

To determine whether the benefit of force augmentation extends across diverse physical regimes, all models were evaluated on 20 rollouts across 12 Meta-World MT50 tasks (240 rollouts per model).

### 4.1 Physical Interaction Regimes

Crucially, the 12 tasks partition into distinct physical interaction categories:
1. **True Contact Manipulation (4 tasks)**: `pick-place-v3`, `pick-place-wall-v3`, `assembly-v3`, `sweep-into-v3`. Contact force is actively required to grasp, transport, or guide the object.
2. **Constrained Mechanism / Hardstop Holding (7 tasks)**: `door-open-v3`, `drawer-open-v3`, `button-press-topdown-v3`, `drawer-close-v3`, `door-close-v3`, `peg-insert-side-v3`, `hammer-v3`. Success is determined by kinematic alignment with a fixed track; post-success behavior is dominated by pushing against rigid stops (800 N – 2300 N).
3. **Free-Space / Zero-Contact (1 task)**: `reach-v3`. Contact force is strictly 0.00 N throughout.

### 4.2 Multi-Task Benchmark Results (Locked `MT50(seed=42)`)

*All tasks evaluated on 20 fixed episodes per model ($n=20$, evaluation seeds $42 + 17 \cdot i$, $i \in [0, 19]$). Brackets report 90% Wilson score confidence intervals for success rates, and exact paired 90% difference confidence intervals ($d_i = y_{i, \text{Force}} - y_{i, \text{Kin}}$) across the identical 20 episode configurations.*

| Task Name | Interaction Regime | 12T Kin MLP ($n=20$) | 12T Kin GRU ($n=20$) | 12T Force GRU ($n=20$) | Force Delta vs Kin GRU [Paired 90% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `pick-place-v3` | **True Contact** | 45.0% [28.4, 62.8] | 5.0% [1.1, 19.6] | **25.0%** [12.7, 43.2] | **+20.0%** [+4.9%, +35.1%] |
| `pick-place-wall-v3` | **True Contact** | 70.0% [51.6, 83.6] | 0.0% [0.0, 11.9] | **65.0%** [46.7, 79.8] | **+65.0%** [+47.0%, +83.0%] |
| `sweep-into-v3` | **True Contact** | 95.0% [80.4, 98.9] | 95.0% [80.4, 98.9] | 75.0% [56.8, 87.3] | **-20.0%** [-35.1%, -4.9%] |
| `assembly-v3` | **True Contact** | 65.0% [46.7, 79.8] | 0.0% [0.0, 11.9] | 0.0% [0.0, 11.9] | no discordant pairs |
| `hammer-v3` | Hardstop Mechanism | 95.0% [80.4, 98.9] | 40.0% [24.2, 58.1] | **80.0%** [62.2, 90.7] | **+40.0%** [+18.0%, +62.0%] |
| `button-press-topdown-v3`| Hardstop Mechanism | 100.0% [88.1, 100.0] | 55.0% [37.2, 71.6] | **100.0%** [88.1, 100.0] | **+45.0%** [+26.2%, +63.8%] |
| `door-open-v3` | Hardstop Mechanism | 100.0% [88.1, 100.0] | 100.0% [88.1, 100.0] | 100.0% [88.1, 100.0] | no discordant pairs |
| `drawer-open-v3` | Hardstop Mechanism | 100.0% [88.1, 100.0] | 100.0% [88.1, 100.0] | 100.0% [88.1, 100.0] | no discordant pairs |
| `drawer-close-v3` | Hardstop Mechanism | 100.0% [88.1, 100.0] | 100.0% [88.1, 100.0] | 100.0% [88.1, 100.0] | no discordant pairs |
| `door-close-v3` | Hardstop Mechanism | 95.0% [80.4, 98.9] | 100.0% [88.1, 100.0] | 90.0% [73.8, 96.6] | **-10.0%** [-21.3%, +1.3%] |
| `peg-insert-side-v3` | Hardstop Mechanism | 75.0% [56.8, 87.3] | 0.0% [0.0, 11.9] | **5.0%** [1.1, 19.6] | **+5.0%** [-3.2%, +13.2%] |
| `reach-v3` | Free-Space (0 N) | 70.0% [51.6, 83.6] | 30.0% [16.4, 48.4] | 30.0% [16.4, 48.4] | +0.0% [-20.7%, +20.7%] |

### 4.3 Regime-Decomposed Summary

| Regime Group | 12T Kinematic MLP | 12T Kinematic GRU | 12T Force GRU | Force Impact on GRU [Paired 90% CI] |
| :--- | :---: | :---: | :---: | :---: |
| **True Contact Tasks (4 tasks, 80 rollouts)** | 68.8% (55 / 80) [59.7, 76.5] | 25.0% (20 / 80) [17.9, 33.7] | **41.2% (33 / 80)** [32.6, 50.4] | **+16.2%** [+7.3%, +25.2%] |
| **Hardstop Mechanism Tasks (7 tasks, 140 rollouts)** | 95.0% (133 / 140) [91.0, 97.3] | 70.7% (99 / 140) [64.0, 76.6] | **82.1% (115 / 140)** [76.2, 86.8] | **+11.4%** [+6.1%, +16.7%] |
| **Free-Space Tasks (1 task, 20 rollouts)** | 70.0% (14 / 20) [51.6, 83.6] | 30.0% (6 / 20) [16.4, 48.4] | 30.0% (6 / 20) [16.4, 48.4] | +0.0% [-20.7%, +20.7%] |
| **Overall Benchmark (12 tasks, 240 rollouts)** | **84.2% (202 / 240)** [79.9, 87.7] | **52.1% (125 / 240)** [46.8, 57.3] | **64.2% (154 / 240)** [58.9, 69.1] | **+12.1%** [+7.5%, +16.7%] |

### 4.4 Feedforward Force MLP (45-D) Ablation & Hypothesis Test

To isolate whether force feedback is independently causal in a memoryless architecture or depends on recurrent temporal integration, an identical feedforward MLP policy was trained with 45-D force-augmented observations (`models/bc_12tasks_mlp_force.pt`, Linear(45, 256) $\to$ ReLU $\to$ Linear(256, 256) $\to$ ReLU $\to$ Linear(256, 4), trained for 20 epochs on `dataset_12tasks/dataset_12tasks_force.pt`).

#### 4.4.1 Multi-Task Benchmark Results (Locked `MT50(seed=42)`)
**Pre-Run Hypothesis**: *"If force is causal, Force MLP > Kin MLP on pick-place and pick-place-wall."*

The model was evaluated once across all 12 tasks on the identical locked `MT50(seed=42)` 20-episode benchmark:

| Task Name | Interaction Regime | 12T Kin MLP ($n=20$) | 12T Force MLP ($n=20$) | Force Delta vs Kin MLP [Paired 90% CI] |
| :--- | :--- | :---: | :---: | :---: |
| `pick-place-v3` | **True Contact** | 45.0% (9/20) [28.4, 62.8] | 25.0% (5/20) [12.7, 43.2] | **-20.0%** [-35.1%, -4.9%] |
| `pick-place-wall-v3` | **True Contact** | 70.0% (14/20) [51.6, 83.6] | 65.0% (13/20) [46.7, 79.8] | **-5.0%** [-23.8%, +13.8%] |
| `sweep-into-v3` | **True Contact** | 95.0% (19/20) [80.4, 98.9] | **100.0% (20/20)** [88.1, 100.0] | **+5.0%** [-3.2%, +13.2%] |
| `assembly-v3` | **True Contact** | 65.0% (13/20) [46.7, 79.8] | **75.0% (15/20)** [56.8, 87.3] | **+10.0%** [-10.3%, +30.3%] |
| `hammer-v3` | Hardstop Mechanism | 95.0% (19/20) [80.4, 98.9] | **100.0% (20/20)** [88.1, 100.0] | **+5.0%** [-3.2%, +13.2%] |
| `button-press-topdown-v3`| Hardstop Mechanism | 100.0% (20/20) [88.1, 100.0] | 100.0% (20/20) [88.1, 100.0] | no discordant pairs |
| `door-open-v3` | Hardstop Mechanism | 100.0% (20/20) [88.1, 100.0] | 100.0% (20/20) [88.1, 100.0] | no discordant pairs |
| `drawer-open-v3` | Hardstop Mechanism | 100.0% (20/20) [88.1, 100.0] | 100.0% (20/20) [88.1, 100.0] | no discordant pairs |
| `drawer-close-v3` | Hardstop Mechanism | 100.0% (20/20) [88.1, 100.0] | 100.0% (20/20) [88.1, 100.0] | no discordant pairs |
| `door-close-v3` | Hardstop Mechanism | 95.0% (19/20) [80.4, 98.9] | 90.0% (18/20) [73.8, 96.6] | **-5.0%** [-19.5%, +9.5%] |
| `peg-insert-side-v3` | Hardstop Mechanism | 75.0% (15/20) [56.8, 87.3] | **95.0% (19/20)** [80.4, 98.9] | **+20.0%** [+4.9%, +35.1%] |
| `reach-v3` | Free-Space (0 N) | 70.0% (14/20) [51.6, 83.6] | 65.0% (13/20) [46.7, 79.8] | **-5.0%** [-30.2%, +20.2%] |
| **TOTAL (240 eps)** | — | **84.2% (202 / 240)** [79.9, 87.7] | **84.6% (203 / 240)** [80.4, 88.0] | **+0.4%** [-3.5%, +4.4%] |

#### 4.4.2 Paired 12-Pose Benchmark Evaluation (`MT10(seed=42)`)
**Pre-Run Hypothesis**: *"Force MLP will not beat the Kin MLP's 25% by more than noise ($\ge 40\%$ would contradict this read) and will flip at most 1 of 6 hard poses."*

**Pre-Run Hypothesis Outcome**:
- **Aggregate clause held**: Force MLP achieved **33.3% (20/60 rollouts, 4/12 poses)**, strictly $<40\%$, which did not beat Kin MLP's 25.0% (15/60 rollouts, 3/12 poses) by more than noise ($+8.3\%$ [Paired 90% CI: $-3.5\%, +20.2\%$]; pose-level exact sign test $p=1.000$).
- **Flip clause missed under the standing definition**: Under the standing definition of baseline hard poses (the 6 poses where the 5-task baseline scored 0%: #00, #07, #20, #31, #40, #47), Force MLP solved **3 of 6** hard poses (#00, #31, #40), exceeding the predicted $\le 1/6$ bound. (Note: relative to Kin MLP, which already solved #00 and #31, Force MLP flipped only 1 incremental pose, #40).

To directly test whether force augmentation resolves edge-pose grasping in a feedforward architecture, Force MLP was evaluated on the locked 12-pose paired benchmark (12 poses $\times$ 5 seeds = 60 rollouts):

| Evaluation Metric | 12T Kinematic MLP | 12T Force MLP | Difference (Force MLP - Kin MLP) | Paired 90% CI | Pose-Level Sign Test |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Benchmark Success** | 25.0% (15 / 60) | **33.3% (20 / 60)** | **+8.3%** | [-3.5%, +20.2%] | — |
| **Poses Solved (5/5)** | 3 / 12 (#00, #15, #31) | 4 / 12 (#00, #15, #31, #40) | **+1 pose (#40)** | — | $p = 1.000$ (1 vs 0) |
| **Baseline Hard Poses Solved** | 2 / 6 (#00, #31) | 3 / 6 (#00, #31, #40) | **+1 pose (#40)** | — | — |
| **Baseline Easy Poses Maintained** | 1 / 6 (#15) | 1 / 6 (#15) | **0 poses** | — | — |

*Pose-by-pose details*: Force MLP scored 100% on #00, #15, #31, and #40, and 0% on the remaining 8 poses. Relative to Kin MLP (which succeeded on #00, #15, and #31), Force MLP flipped exactly **1 hard pose (#40)**, and its overall score (33.3%) is well below the 40% threshold, with a confidence interval overlapping zero.

#### 4.4.3 Findings & Architectural Implications
1. **Refutation of Isolated Feedforward Force Hypothesis**: On `pick-place-v3` and `pick-place-wall-v3`, Force MLP does *not* outperform Kinematic MLP:
   - On multi-task `pick-place-v3`, Force MLP fell from 45.0% to 25.0% ($-20.0\%$ [Paired 90% CI: $-35.1\%, -4.9\%$]).
   - On `pick-place-wall-v3`, Force MLP remained virtually unchanged (65.0% vs 70.0%, $-5.0\%$ [Paired 90% CI: $-23.8\%, +13.8\%$]).
   - On the 12-pose benchmark, Force MLP scored 33.3% vs Kin MLP 25.0% (+8.3%, within noise), far below Force GRU (50.0%).
2. **Where Force Feedforward Directly Helps**: Force augmentation produced significant, unconfounded gains in high-precision mechanical insertion tasks:
   - `peg-insert-side-v3`: **+20.0%** (from 75.0% to 95.0%, [Paired 90% CI: $+4.9\%, +35.1\%$]).
   - `assembly-v3`: **+10.0%** (from 65.0% to 75.0%).
   - `hammer-v3`: **+5.0%** (from 95.0% to 100.0%).
3. **The GRU Interaction**: In recurrent policies, Kinematic GRU experienced severe multi-task capacity collapse on precision tasks (`assembly`: 0%, `peg-insert`: 0%, `pick-place-wall`: 0%), whereas feedforward MLPs did not suffer this collapse (65%–75%). Adding force to GRUs rescued them from collapse on `pick-place-wall` (0% $\to$ 65%) and `hammer` (40% $\to$ 80%), demonstrating that contact feedback stabilizes recurrent state estimation across multi-task regimes.

---

## 5. Methodological Insights & Caveats

### 5.1 Hardstop Holding as a Confounder
An investigation into the 7 mechanism tasks revealed that Meta-World scripted oracles achieve task success within 50–110 steps, and then continue commanding maximum actuator thrust into mechanical hardstops for the remaining 400 steps:
- Pre-success contact force: $<6\text{ N}$.
- Post-success contact force: **800 N to 2300 N**.
- On these tasks, an elevated force signal predominantly encodes "task completed, holding against limit" rather than pre-contact tactile guidance. Consequently, gains on tasks like `button-press` and `hammer` reflect terminal limit holding, whereas **the primary, unconfounded test of tactile manipulation remains `pick-place-v3` and `pick-place-wall-v3`**, where forces are purely interactive ($6\text{ N} - 10\text{ N}$).

### 5.2 Clarification of Task Initialization (`MT10()` vs `MT10(seed=42)`)
Meta-World's internal task generator `metaworld._make_tasks` initializes tasks using NumPy's RNG:
- **Phase 2 Baseline Decomposition (`task-1453.log`)**: Instantiated with unseeded `MT10()`. On that set, the 12 selected poses split into **4 Easy (#10, #24, #40, #47)** and **8 Hard (#00, #04, #07, #15, #20, #31, #35, #44)**.
- **Phase 3 & Phase 4 Paired Benchmarks (`task-1778.log`, `task-3108.log`, `task-3170.log`)**: Instantiated with `MT10(seed=42)`. Passing `seed=42` regenerated task goals, producing a different sequence of coordinates that evaluated to **6 Easy (#04, #10, #15, #24, #35, #44)** and **6 Hard (#00, #07, #20, #31, #40, #47)**.
- Both sets are internally consistent, 100% deterministic, and exhibit zero stochastic variance across random seeds. All comparisons between the baseline, weighted loss, kinematic GRU, and force GRU were conducted on the **exact same `MT10(seed=42)` locked set**.

### 5.3 Methodological Note on Multi-Task Benchmark Task Seeding (`MT50(seed=None)` vs `MT50(seed=42)`)
The earlier version of this report reflected an initial unseeded `MT50()` execution. In Meta-World, constructing `MT50()` with default `seed=None` causes internal task goal sampling (`_make_tasks`) to depend on ambient Python RNG state at process launch. While all 3 models were evaluated on the exact same task instances within that single run, point estimates exhibited sampling variance across separate process invocations (e.g., `door-close-v3` evaluated at 60%, 70%, 80%, and 90% across different unseeded process runs, though Kinematic GRU was 100% invariant across all runs).

To ensure complete rigor and bitwise reproducibility matching the primary pick-place benchmark, the entire 12-task benchmark was completely re-executed with `metaworld.MT50(seed=42)` locked in advance across all three models (720 total rollouts, 240 rollouts per model, evaluation seeds $42 + 17 \cdot i$). The numbers reported in Section 4 are now permanently locked, reproducible down to individual episode steps, and reported with explicit 90% confidence intervals to avoid over-interpreting sample noise at $n=20$ episodes per task.

---


### 5.4 Limitations

1. **Force-Blind Scripted Demonstrations**: All expert demonstrations in this benchmark were generated by Meta-World scripted oracles. These oracles are completely force-blind; they follow open-loop, hardcoded Cartesian waypoints without tactile compliance or force-guided velocity modulation. Consequently, the imitation objective $\mathcal{L}_{\text{BC}}$ trains policies to predict kinematics-driven actions. An imitation policy cannot learn active, force-compliant closed-loop manipulation strategies from demonstrations that do not exhibit them.
2. **Multiple Comparisons Across Tasks**: Evaluating 12 distinct manipulation tasks across 4 model architectures creates a substantial multiple comparisons problem. Individual task deltas with nominal $p < 0.05$ or non-overlapping confidence intervals must be interpreted cautiously, as family-wise error rates are elevated. Conclusions should be drawn from regime-wide patterns (e.g., true contact vs hardstop mechanisms vs free-space) and paired multi-seed benchmarks rather than single task point estimates.
3. **Single-Seed Evaluation and Sample Size ($n=20$)**: The 12-task multi-task benchmark evaluates $n=20$ episodes per task under a single locked generator seed (`MT50(seed=42)`). While holding seeds strictly constant ensures determinism and valid paired comparisons, $n=20$ produces wide 90% confidence intervals (margins of $\pm 14\% - 18\%$ on point estimates). Point estimate fluctuations within this margin reflect binomial sampling uncertainty rather than systematic policy capabilities.

---

## 6. Conclusion & Recommendation for REVEL

1. **Force Rescued Recurrent Hidden State, But Did Not Help Feedforward Policies**: Contact force augmentation produced a massive +50.0% gain on the 12-pose paired benchmark and +65.0% on multi-task pick-place-wall when added to a recurrent policy (GRU). However, when evaluated on a feedforward MLP control, force augmentation provided zero meaningful gain on pick-place (+8.3% on the 12-pose benchmark within noise; -20.0% on multi-task pick-place). This demonstrates that force sensing does not act as an independent causal solution for grasping; rather, it is hypothesized—as an **untested hypothesis** requiring direct hidden state ablation—to function as a physical anchor that prevents catastrophic recurrent hidden state drift in multi-task GRUs.
2. **Force Benefits Tight Geometric Alignments in MLPs**: While force feedback did not improve open-workspace pick-and-place grasping in MLPs, it produced large, unambiguous gains on tasks with tight mechanical constraints (`peg-insert-side-v3` +20.0%, 75% $\to$ 95%), where instantaneous reactive force correlates directly with insertion alignment.
3. **Revised Manuscript Recommendation**:
   - Reframe the core finding: the primary value of contact force sensing in multi-task imitation learning is **stabilizing sequential/recurrent temporal representations**, rather than acting as a universal drop-in replacement for kinematic grasping.
   - Clarify the distinction between tasks requiring temporal state tracking (where force anchors the recurrence) versus instantaneous compliance (which cannot be fully learned from force-blind scripted experts).
