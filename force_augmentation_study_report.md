# Force-Augmented Observation Study: Closing the Imitation-to-Success Gap in Contact Manipulation

**Project**: Intent-to-Action / REVEL Thesis Evaluation  
**Status**: Completed (Revised Empirical Findings)  
**Date**: September 2026  
**Artifact Directory**: `models/`, `dataset_12tasks/`, `logs/`  

---

## 1. Executive Summary & Core Finding

This study provides a rigorous empirical evaluation of the core hypothesis in the REVEL framework: **whether augmenting robot policy observations with genuine contact force/torque sensing closes the imitation-accuracy-to-task-success gap in contact-rich manipulation**.

### 1.1 The Primary Finding: No Reliable Force Benefit in this Simulator

When evaluated under valid protocols that match training conditions, **augmenting policy observations with genuine 6-D contact wrench sensing ($F_x, F_y, F_z, 	au_x, 	au_y, 	au_z$) yields no reliable, generalizable performance advantage over purely kinematic observations**.

Across both feedforward (MLP) and recurrent (GRU) architectures, contact force augmentation fails to resolve edge-pose grasping in pick-and-place manipulation, and overall multi-task success rates remain virtually indistinguishable from kinematic baselines:

1. **Recurrent Policies (GRU Architecture, Validated `method=window`)**:
   - **12-Pose Paired Benchmark (`pick-place-v3`, 3 Matched Training Seeds)**:
     - Seed 42: Kin GRU **5/12 poses (41.7%)** vs Force GRU **7/12 poses (58.3%)** (Delta: **+2 poses**)
     - Seed 43: Kin GRU **6/12 poses (50.0%)** vs Force GRU **5/12 poses (41.7%)** (Delta: **-1 pose**, Kinematic beats Force)
     - Seed 44: Kin GRU **6/12 poses (50.0%)** vs Force GRU **6/12 poses (50.0%)** (Delta: **0 poses**, bitwise identical)
     - **Result**: Force GRU is within $\pm 2$ poses of Kinematic GRU on every matched seed (+2, -1, 0), refuting the hypothesis that force sensing provides an independent causal advantage ($\ge 3$ poses).
   - **12-Task Multi-Task Benchmark (`MT50(seed=42)`, 3 Matched Training Seeds, 240 Rollouts/Seed)**:
     - Seed 42: Kin GRU 202/240 (84.2%) vs Force GRU 212/240 (88.3%) (Delta: +10 episodes)
     - Seed 43: Kin GRU 189/240 (78.8%) vs Force GRU 194/240 (80.8%) (Delta: +5 episodes)
     - Seed 44: Kin GRU 198/240 (82.5%) vs Force GRU 205/240 (85.4%) (Delta: +7 episodes)
     - **Mean Performance Across 3 Seeds**: Kinematic GRU **196.3 / 240 (81.8%)** vs Force GRU **203.7 / 240 (84.9%)**.
     - **Mean Difference**: **+7.3 episodes (+3.1%) out of 240**, strictly within the pre-stated $\pm 10$ episode noise margin of the benchmark.
     - On the primary contact manipulation tasks, Kinematic GRU slightly outperformed Force GRU: `pick-place-v3` (**40.0% vs 35.0%**) and `pick-place-wall-v3` (**91.7% vs 86.7%**).

2. **Feedforward Policies (MLP Architecture Control)**:
   - **12-Pose Paired Benchmark (`pick-place-v3`)**: Kinematic MLP scored **25.0%** (3/12 poses). Force MLP scored **33.3%** (4/12 poses; exact sign test $p=1.000$), consistent with noise.
   - **12-Task Benchmark (`MT50(seed=42)`)**: On `pick-place-v3`, Force MLP actually **regressed** compared to Kinematic MLP (**25.0% vs 45.0%**, $-20.0\%$). Overall across 12 tasks, Force MLP (84.6%, 203/240) was virtually identical to Kinematic MLP (84.2%, 202/240, $+0.4\%$).

---

## 2. Anatomy of the Recurrent Evaluation Bug & Artifact Dissection

An earlier version of this report presented an apparent "+50.0% pose gain" and "+65.0% pick-place-wall gain" for Force GRU over Kinematic GRU, suggesting that force acted as a "physical anchor" rescuing recurrent state. **Detailed investigation confirmed this was entirely an artifact of a severe train/eval protocol mismatch.**

### 2.1 How the Mismatch Arose

The bug was inherited from the original multi-task behavioral cloning implementation (`train_bc_gru.py` and `compare_models.py`):

1. **Training Protocol (`SequenceDataset` with $H=8$)**:
   During training, the policy was trained on short sequence slices of length 8 ($H=8$). In the training loop:
   ```python
   # train_12tasks_study.py / train_bc_gru.py
   pred, _ = model(obs_seq)  # hidden is None -> h_0 initialized to 0
   ```
   At every gradient step, the GRU was passed a batch of 8-step sequences with `hidden=None`, meaning **the hidden state was always initialized to zero ($h_0 = 0$) at the start of each 8-step window**. The GRU was never trained on sequences longer than 8 steps, and never learned to maintain stable recurrent state across hundreds of closed-loop environment steps.

2. **Evaluation Protocol (`method=step`)**:
   In the closed-loop evaluation harness (`eval_12tasks_benchmark.py`, `eval_12tasks_paired_pickplace.py`):
   ```python
   # Rollout evaluation in old benchmark
   model.reset_hidden(device)  # h_0 initialized to 0 once at t=0
   for step in range(500):
       feat_t = torch.from_numpy(norm_feat).float().unsqueeze(0).to(device)
       action = model.step(feat_t)  # carries self._h continuously for 500 steps!
   ```
   The hidden state $h_t$ was updated step-by-step and carried continuously across all 500 steps of the episode. By step 20, 50, and 200, the recurrent state accumulated compounding numerical and representational drift, entering regions of state space orders of magnitude outside anything seen during training.

### 2.2 Why Kinematic GRU Collapsed While Force GRU Appeared to Survive

Under continuous 500-step carryover (`method=step`), Kinematic GRU suffered catastrophic state drift, causing it to freeze or execute pathological actions (scoring 0.0% on `pick-place-v3`, `pick-place-wall-v3`, and `assembly-v3`).

In Force GRU, the additional 6 dimensions of contact wrench—which experience sharp changes when contacting surfaces—happened to perturb the recurrent dynamics in a way that partially disrupted runaway state drift on certain trajectories. This allowed Force GRU to complete some tasks under `step` (reaching 50%–65%), creating the powerful optical illusion of a "force rescue."

### 2.3 How the Artifact Was Discovered and Verified

1. **Input Tensor Inspection**: Comparing the input tensors fed to the GRU at rollout time versus training time revealed that `model.step()` was passing single-step slices with unconstrained 500-step recurrent history, completely violating the 8-step zero-initialized training formulation.
2. **Protocol Correction (`method=window`)**: When evaluation was aligned with training (maintaining an 8-step sliding observation buffer $[o_{t-7}, \dots, o_t]$ and resetting $h_0 = 0$ at each decision step, exactly matching training forward passes), the apparent Kinematic GRU "collapse" instantly vanished:
   - On the 12-pose benchmark, Kinematic GRU leaped from **0.0% to 41.7%–50.0%**.
   - On `pick-place-wall-v3`, Kinematic GRU leaped from **0.0% to 100.0%**.
   - On the 12-task benchmark, Kinematic GRU leaped from **52.1% to 84.2%**.
3. **Multi-Seed Matched Re-evaluation**: Re-training both models across 3 independent random seeds (s42, s43, s44) and evaluating under valid `method=window` demonstrated that Force GRU provided **no statistically or practically significant advantage** over Kinematic GRU.
4. **Deletion of 'Physical Anchor' Hypothesis**: The hypothesis that contact force provides a "physical anchor" stabilizing recurrent trajectories is deleted; it was an elaborate explanation of an evaluation bug.

---

## 3. Validated 12-Pose Paired Benchmark (`pick-place-v3`)

The 12-pose paired benchmark evaluates 12 fixed workspace configurations across 5 fixed seeds ($N=60$ rollouts per model) using `method=window`.

### 3.1 Pose-by-Pose Results Across Matched Seeds

| Pose ID | Initial Object `[X, Y, Z]` | Target Goal `[X, Y, Z]` | Kin s42 | Force s42 | Kin s43 | Force s43 | Kin s44 | Force s44 | Kin MLP | Force MLP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#00** | `[1.00, -0.05, 0.62]` | `[-0.06, 0.90, 0.13]` | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| **#04** | `[1.00, -0.05, 0.69]` | `[ 0.00, 0.84, 0.20]` | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| **#07** | `[1.00, -0.06, 0.61]` | `[-0.09, 0.85, 0.28]` | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| **#10** | `[1.00,  0.04, 0.68]` | `[ 0.04, 0.85, 0.20]` | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| **#15** | `[1.00,  0.02, 0.65]` | `[ 0.05, 0.88, 0.23]` | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| **#20** | `[1.00,  0.03, 0.61]` | `[-0.01, 0.86, 0.09]` | 0/5 | 5/5 | 5/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| **#24** | `[1.00, -0.03, 0.67]` | `[ 0.03, 0.86, 0.23]` | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| **#31** | `[1.00, -0.01, 0.61]` | `[-0.06, 0.86, 0.21]` | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| **#35** | `[1.00, -0.07, 0.63]` | `[-0.00, 0.84, 0.15]` | 0/5 | 5/5 | 0/5 | 0/5 | 5/5 | 5/5 | 0/5 | 0/5 |
| **#40** | `[1.00,  0.03, 0.65]` | `[ 0.09, 0.88, 0.12]` | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 5/5 |
| **#44** | `[1.00,  0.00, 0.68]` | `[ 0.01, 0.86, 0.18]` | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| **#47** | `[1.00, -0.03, 0.61]` | `[ 0.08, 0.90, 0.14]` | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 |
| **Poses (5/5)** | — | — | **5/12** | **7/12** | **6/12** | **5/12** | **6/12** | **6/12** | **3/12** | **4/12** |
| **Rollouts (%)** | — | — | **41.7%** | **58.3%** | **50.0%** | **41.7%** | **50.0%** | **50.0%** | **25.0%** | **33.3%** |

### 3.2 Matched-Seed Pairwise Analysis
- **Seed 42**: Force GRU solved 7 poses vs Kin GRU 5 poses (**+2 poses**, gaining #20 and #35).
- **Seed 43**: Kin GRU solved 6 poses vs Force GRU 5 poses (**-1 pose**, Kinematic beats Force on #20).
- **Seed 44**: Force GRU solved 6 poses vs Kin GRU 6 poses (**0 poses**, identical solutions on #00, #15, #31, #35, #40, #47).
- **Mean Poses Solved**: Kinematic GRU solves **5.67 / 12 poses** (47.2%); Force GRU solves **6.00 / 12 poses** (50.0%). The net mean difference is **+0.33 poses** out of 12, well within stochastic noise.

---

## 4. Validated 12-Task Multi-Task Benchmark (`MT50(seed=42)`)

All models were evaluated across 12 Meta-World tasks with 20 fixed episodes per task ($N=240$ rollouts per run) under `method=window`.

### 4.1 Multi-Seed Results Table (Seeds 42, 43, 44)

| Task Name | Interaction Regime | Kin s42 | Force s42 | Kin s43 | Force s43 | Kin s44 | Force s44 | Mean Kin | Mean Force | Mean Δ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pick-place-v3` | **True Contact** | 8/20 | 10/20 | 8/20 | 5/20 | 8/20 | 6/20 | **8.0/20 (40.0%)** | **7.0/20 (35.0%)** | **-1.0 (-5.0%)** |
| `pick-place-wall-v3` | **True Contact** | 20/20 | 18/20 | 17/20 | 17/20 | 19/20 | 17/20 | **18.7/20 (93.3%)** | **17.3/20 (86.7%)** | **-1.3 (-6.7%)** |
| `peg-insert-side-v3` | Hardstop Hold | 11/20 | 16/20 | 11/20 | 11/20 | 8/20 | 16/20 | 10.0/20 (50.0%) | 14.3/20 (71.7%) | +4.3 (+21.7%) |
| `assembly-v3` | **True Contact** | 20/20 | 20/20 | 15/20 | 20/20 | 20/20 | 20/20 | 18.3/20 (91.7%) | 20.0/20 (100.0%) | +1.7 (+8.3%) |
| `hammer-v3` | Hardstop Hold | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20.0/20 (100.0%)| 20.0/20 (100.0%) | +0.0 (0.0%) |
| `sweep-into-v3` | **True Contact** | 18/20 | 20/20 | 20/20 | 19/20 | 19/20 | 20/20 | 19.0/20 (95.0%) | 19.7/20 (98.3%) | +0.7 (+3.3%) |
| `reach-v3` | Free-Space (0 N) | 5/20 | 8/20 | 4/20 | 4/20 | 4/20 | 6/20 | 4.3/20 (21.7%) | 6.0/20 (30.0%) | +1.7 (+8.3%) |
| `door-open-v3` | Hardstop Hold | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20.0/20 (100.0%)| 20.0/20 (100.0%) | +0.0 (0.0%) |
| `drawer-open-v3` | Hardstop Hold | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20.0/20 (100.0%)| 20.0/20 (100.0%) | +0.0 (0.0%) |
| `button-press-topdown-v3`| Hardstop Hold | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20.0/20 (100.0%)| 20.0/20 (100.0%) | +0.0 (0.0%) |
| `drawer-close-v3` | Hardstop Hold | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20.0/20 (100.0%)| 20.0/20 (100.0%) | +0.0 (0.0%) |
| `door-close-v3` | Hardstop Hold | 20/20 | 20/20 | 14/20 | 18/20 | 20/20 | 20/20 | 18.0/20 (90.0%) | 19.3/20 (96.7%) | +1.3 (+6.7%) |
| **TOTAL (out of 240)** | — | **202** | **212** | **189** | **194** | **198** | **205** | **196.3 / 240** | **203.7 / 240** | **+7.3 episodes** |
| **SUCCESS RATE (%)** | — | **84.2%** | **88.3%** | **78.8%** | **80.8%** | **82.5%** | **85.4%** | **81.8%** | **84.9%** | **+3.1%** |

### 4.2 Evaluating the Pre-Stated Benchmark Hypothesis
- **Hypothesis Stated in Advance**: *"The mean Force−Kin difference across three seeds is within $\pm 10$ episodes of 240, using reach-v3 as the noise yardstick."*
- **Outcome**:
  - The mean Force minus Kinematic difference across all 12 tasks is **+7.33 episodes out of 240** (+3.1%), which strictly satisfies the pre-stated bound ($\le \pm 10$ episodes).
  - On `reach-v3`—a completely free-space task where contact forces are strictly 0.00 N and provide zero task information—the mean Force minus Kinematic difference was **+1.67 episodes**.

---

## 5. Methodological Takeaways & Limitations

1. **Protocol Symmetry in Recurrent Imitation Learning**:
   Training policies with truncated sequence windows and evaluating them with unreset step-by-step carryover creates severe out-of-distribution recurrent drift. Closed-loop rollouts must maintain the exact same history length and initial hidden state conditions ($h_0 = 0$) used during gradient optimization.
2. **Force-Blind Scripted Demonstrations Cannot Teach Tactile Control**:
   All training trajectories were generated by open-loop Meta-World scripted experts. Because these oracles follow fixed geometric waypoints without modulating velocity or compliance based on contact forces, the imitation objective $\mathcal{L}_{\text{BC}}$ offers no training signal for reactive tactile strategies. In an imitation learning setup, providing force feedback to a policy trained on force-blind demonstrations cannot induce tactile behaviors that are entirely absent from the dataset.
3. **Hardstop Holding Artifacts in Meta-World**:
   On mechanism tasks (`door-open`, `drawer-open`, `button-press`, `door-close`), the scripted oracle drives the actuator into mechanical limits for hundreds of steps after task completion, generating forces of 800 N–2300 N. In these tasks, force features function as an artifactual indicator of completion rather than closed-loop manipulation cues.

---

## 6. Final Conclusion & Recommendations

1. **Refutation of Sensory Insufficiency Hypothesis**: The hypothesis that kinematics-only imitation fails due to sensory insufficiency that can be resolved by feeding raw contact wrenches is refuted for this simulation environment.
2. **True Source of Pick-Place Failure**: Failure on edge poses in `pick-place-v3` is driven by demonstration coverage and high-dimensional regression ambiguity near contact boundaries, not by missing force observations.
3. **Recommendation for REVEL**: Rather than treating contact force as an automatic drop-in observation for standard behavioral cloning, genuine tactile benefits require demonstrations generated by force-reactive controllers or reinforcement learning objectives where policy actions actively leverage contact feedback.
