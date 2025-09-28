# ECO Decoding — Entropy-Based Control for Controllable Dialogue Generation

> **One-line**: A decoding-time method that *dynamically* balances **controllability** and **fluency** by scaling attribute logits with **entropy based control strength**.

This repository contains a implementation of **ECO Decoding** for controllable dialogue generation (CDG). It augments weighted decoding methods (e.g., **Director**, **DASC**) with **dynamic, per-step control strength** computed from the entropy of the language model and attribute classifier distributions.

---

## Highlights

- **Plug-in to existing weighted decoding** (FUDGE/Director/DASC-style) — no extra training required; control strength is chosen *on-the-fly* at each step.
- **Better controllability without hurting grammar/fluency** (vs. static coefficients).
- **Multi-attribute friendly** — alleviates probability-interpolation issues; naturally extends via product over attributes.
- Minimal runtime overhead in our tests (per-token latency +~1%).

> For method details, ablations, and full results, see **Citation** at the bottom of this README.

---

## Repository Structure

```
.
├─ model/
│  ├─ modeling_gpt2_director.py   # GPT-2 backend with Director head (+ ECO)
│  └─ modeling_gpt2_dasc.py       # GPT-2 backend with DASC-style attribute space (+ ECO)
├─ datamodule.py                  # DataModule for DailyDialog / MultiWOZ style corpora
├─ learner.py                     # LightningModule: training loop + ECO decoding hooks
├─ main.py                        # Hydra entrypoint (train/test)
├─ perplexity.py                  # Perplexity metric helper
├─ run.sh                         # Example training command(s)
└─ test.sh                        # Example test/inference commands with ECO knobs
```

> **Note**: The `modeling_*.py` files live under `model/` and implement the GPT-2 backends with the necessary heads for Director/DASC, plus the ECO integration points.

---

## Datasets

- **DailyDialog**: open-domain multi-turn dialog with *emotion* and *dialog-act* attributes.
- **MultiWOZ**: multi-domain task-oriented dialogs; attributes obtained via an evaluator/labeler.

The provided `datamodule.py` prepares loaders for the above settings. If you maintain data in custom folders, adjust the paths/logic in `datamodule.py`.

---

## Quick Start (Training)

You can use **Hydra** overrides via CLI. The provided `run.sh` shows a minimal training recipe with **Director** on the **emotion** attribute:

```bash
# Single-attribute (emotion) training with Director
CUDA_VISIBLE_DEVICES=0 python main.py epochs=30 \
  learning_rate=1e-5 gradient_accumulation_steps=1 method=gpt2_director \
  datamodule.data_name=emo seed=616 wandb=False \
  learner.use_prompt=True learner.num_attribute=1 learner.pre_seq_len=200
```

**Tips**

- `method` can be switched to your preferred backend (e.g., a DASC variant) if implemented.
- Hydra’s default config path is `../conf/config.yaml` from `main.py`; CLI overrides allow you to run without editing config files.

---

## Evaluation / Inference (with ECO)

`test.sh` demonstrates how to turn on ECO decoding and sweep key knobs:

```bash
# Evaluate with ECO decoding enabled
CUDA_VISIBLE_DEVICES=0 python main.py epochs=30 \
  learning_rate=1e-5 gradient_accumulation_steps=8 method=gpt2_director \
  datamodule.data_name=emo seed=616 wandb=False \
  learner.smoothing_factor=1.0 learner.condition_lambda=1.0 \
  learner.use_prompt=True learner.num_attribute=1 learner.pre_seq_len=200 test=True
```


- `learner.condition_lambda (λ)`: **global control scale** — larger values bias more toward the attribute classifier.
- `learner.smoothing_factor (τ)`: softmax temperature used when turning raw logits into entropy (stabilizes entropy measurement).

---
