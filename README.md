# Emergent Language as an Approach to Conscious AI

Source code for the paper [**"Emergent Language as an Approach to Conscious AI"**](https://arxiv.org/abs/2606.06380).

## Overview

Two agents with fully independent parameters learn to coordinate via a narrow discrete message channel (7 tokens including SILENCE). We test three structural preconditions for self-referential communication:

- **P1 — Indexical Encoding**: messages encode the sender's own private state ($I(m; s_{\text{self}}) \gg I(m; s_{\text{other}})$), with partner-specific dialects and context-invariant encoding.
- **P2 — Persistent State Representation**: under POMDP masking ($s_i$ visible only at $t{=}0$), the GRU hidden state latches $s_{\text{self}}$ across the episode and acquires $s_{\text{other}}$ from incoming messages.
- **P3 — Behavioral Self-Monitoring**: an echo channel feeds back the agent's own (possibly corrupted) message; agents detect echo–intention mismatches and adjust subsequent communication, exhibiting a closed-loop self-monitoring circuit.

## Files

### Training

| Script | Description |
|---|---|
| `train_independent.py` | **With-echo training.** Two independent GRU agents (d=128), echo channel active, POMDP masking, 4-phase performance-gated curriculum (60k updates). Produces the models for all P1/P2/P3 evaluations. |
| `train_independent_noecho.py` | **No-echo training (ablation control).** Identical setup but echo permanently silenced (`NO_ECHO=True`). Communication is preserved; echo-dependent P3 self-monitoring signatures are selectively abolished. |

### Evaluation

| Script | Description |
|---|---|
| `probe_independent.py` | **Main probe battery** on with-echo models. Includes: trigger contrast (corruption → silence-breaking), sender–receiver dissociation, echo vs. receive channel split, linear probes ($h_t \to$ intended token / actual token / $s_{\text{self}}$ / $s_{\text{other}}$), corruption detection (same-step & lag-1), test-time echo ablation, MI analysis, downstream repair benefit, and cross-pair controls (A-vs-A, B-vs-B). |
| `probe_independent_noecho.py` | **No-echo probe battery.** Runs the no-echo ablation diagnostics on models trained with echo permanently silenced and evaluated with echo kept at SILENCE. This corresponds to the paper’s train-time no-echo control: ordinary communication is preserved (`Δ_comm = 0.283 ± 0.011` in the paper), while the echo-dependent trigger contrast drops to approximately zero. |
| `probe_independent_secondorder.py` | **Second-order analysis.** B1: corruption decodability from `h_t`, orthogonality between first-order and second-order probe directions, causal mediation analysis, and no-echo thermometer baseline. B2: counterfactual intention perturbation under CLEAN, FORCE, CORRUPT, and FORCE_NOECHO conditions. |

## Environment

Abstract coordination task: each agent holds a private state $s_i \in \{0,1,2\}$ and context $c_i \in \{0,\dots,5\}$ during training, with held-out contexts $\{6,7,8\}$ for generalization tests. Episodes last $T{=}10$ steps. The correct action is $a^*_{i \to j} = (s_j + c_i + t) \bmod 3$, requiring communication of private states. The channel is narrow (one token per step), creating bandwidth pressure for efficient encoding. Each agent outputs both an other-targeting action and a self-targeting action, with targets computed by the same modular rule.

## Usage

Scripts are designed for Google Colab with an NVIDIA A100 (40 GB). Each training run (+ full probe battery) takes ~7 hours.

```bash
# Train with-echo agents (10 seeds)
python train_independent.py

# Train no-echo ablation control
python train_independent_noecho.py

# Run probe battery
python probe_independent.py
python probe_independent_noecho.py
python probe_independent_secondorder.py
```
