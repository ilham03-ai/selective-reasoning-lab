# Project Summary

## High-Level Summary

`selective-reasoning-lab` studies selective reasoning in a compact partially observable environment where a model must decide whether to act, inspect for more evidence, or abstain. The core setup is a three-state diagnosis task with noisy observations, optional inspection at a cost, and a meaningful abstention penalty. A small GRU-based model is trained on offline trajectories labeled by an exact Bayesian oracle, then evaluated both for prediction quality and for downstream decision quality. The main result is that uncertainty-aware selective behavior can produce positive utility even when raw hidden-state prediction accuracy is only moderate. The project is positioned as a controlled testbed, not as a new uncertainty-estimation method.

## What Question The Project Studies

The project asks whether a lightweight model can learn not only to predict a hidden state from partial evidence, but also to detect when its internal belief is too uncertain for confident action and therefore to inspect further or abstain.

## What Makes The Setup Interesting

The environment is small enough that the correct posterior and the optimal meta-decision can be computed exactly, which makes interpretation straightforward. At the same time, the problem is not trivial: some prefixes clearly justify acting, some justify another inspection, and some remain uncertain enough that abstention is rational. That makes uncertainty operational rather than cosmetic.

## What Results Matter Most

- Hidden-state prediction reaches about `66%` accuracy with low calibration error (`ECE ≈ 0.019`).
- The learned selective policy reaches about `84.5%` accuracy on the cases where it chooses to act.
- The selective policy acts on `74%` of episodes, abstains on `26%`, and achieves positive average reward (`0.122`).
- Naive baselines remain negative in utility:
  - always act: `-0.272`
  - fixed inspect then act: `-0.043`
  - random inspect: `-0.331`
- The threshold sweep shows a clear act/inspect/abstain frontier rather than a single dominant behavior.

## What Limitations Remain

- The environment is intentionally small and analytically convenient.
- The oracle is exact, so the supervision target is cleaner than most real problems.
- The model effectively matches the oracle meta-decision on this task, which is useful for the prototype but also means the benchmark is still relatively easy.
- The project uses only Monte Carlo dropout for uncertainty and does not benchmark alternative estimators.

## How To Describe The Project In An Application Or Interview

### One-Line Version

I built a small research prototype that studies when an AI system should act, ask for more information, or abstain under partial observability.

### Three-Sentence Version

The project is a compact selective-reasoning lab built in PyTorch around a partially observable diagnosis environment. A GRU-based model predicts hidden state from noisy observation histories, estimates uncertainty with Monte Carlo dropout, and learns a meta-decision policy over acting, inspecting, or abstaining. The main result is that uncertainty-aware selective behavior substantially improves utility over always-acting and fixed-inspection baselines.

### 30-Second Spoken Version

I built a compact Python research prototype called `selective-reasoning-lab` to study uncertainty-aware decision making. The environment has a hidden state and only noisy partial observations, so the model has to decide whether to commit to an action, gather another observation at a small cost, or abstain. I trained a small GRU model on oracle-labeled offline trajectories and showed that even with only moderate raw classification accuracy, a selective policy that uses uncertainty can achieve much better decision quality and positive utility compared with naive baselines.
