# Plan: from LLM autoresearch to tabular oracle autoresearch

This document captures the plan for adapting the autoresearch loop from its original
LLM pretraining target (lower `val_bpb` on a 5-minute budget) to a tabular regression
target: an oracle for a hardware compiler that predicts `bottleneck_fps` from compile
context features.

The basic pipe stays:
- single GPU, single file editable by the agent,
- fixed 5-minute wall-clock training budget,
- branch-per-run, `results.tsv` log of keep / discard / crash, autonomous loop.

What goes away: the GPT model, tokenizer, FA3 kernels, BPE, HuggingFace data download,
`val_bpb`.

---

## 1. Data inventory

File: `all_data.pkl.gz` at the repo root.

- pandas DataFrame, shape `(1_165_935, 222)`, ~1.2 GB in memory.
- Target: `bottleneck_fps` (float32, range ~26 to ~3.1e7, six orders of magnitude — a
  relative-scale loss is essential).
- Auxiliary target also present: `bottleneck_fps_norm_io`.
- Categoricals:
  - `model_name` — 63 unique.
  - `context_name` — 17,349 unique (used for the train/val split, dropped from features).
  - `hw_arch` — 1 value (drop).
  - `group_id` — 144,835 unique (drop, leakage-prone).
- Numerics: float32. Several columns are constant (`hw_n_*`, `agg_dilation_h_*`,
  `agg_groups_min`, `fraction_sparse`, …) and must be dropped during prep.
- Engineered features present: `compute_density`, `memory_pressure`,
  `connector_ratio`, `bn_*` (bottleneck-layer features), `agg_*` (whole-network
  aggregates), etc.

---

## 2. Loss / metric

- Training loss: `MSE(log_pred, log_true)` where target is standardized log-FPS.
- Fixed evaluation metric (the analogue of `evaluate_bpb`, lives in the read-only
  `prepare.py`): **RMSLE** — `sqrt(mean((log_pred - log_true)^2))`.
  - Aligned with the training loss.
  - Symmetric in relative terms (2x over- and under-prediction cost the same).
  - Reads as approximately the standard deviation of relative error.
- Alternative considered: Median Absolute Percentage Error
  (`median(|pred - true| / true)`) — more robust to outliers, slightly less aligned
  with the loss. RMSLE wins on alignment.

The metric column in `results.tsv` is renamed `val_rmsle`. Lower is better.

---

## 3. Train / val split

Group-aware split by `context_name`:

- Hold out ~15% of contexts as val.
- Forces the oracle to generalize to compile contexts it has never seen, which is the
  realistic deployment scenario.
- `context_name` is therefore used to define the split but is **not** a feature.

Open question (default chosen, easy to revisit): drop `context_name` entirely as a
feature. The alternative — integer-encode and embed it — risks context memorization.

---

## 4. Environment changes (priority 1, done)

The original docker image was set up for an LLM workflow (Flash Attention 3 kernels
downloaded at runtime, BPE tokenizer, HuggingFace shard download). All of that is
removed.

### dockerfile

- Base image `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` — kept.
- Removed `HF_HOME` env (no HuggingFace dependency anymore).
- Everything else — apt deps, `uv` install, user setup, `git clone` of the repo as
  in-image scaffold, `CUDA_VISIBLE_DEVICES=0` — unchanged.

### pyproject.toml

Dependency diff:

- Removed: `kernels`, `rustbpe`, `tiktoken`.
- Added: `scikit-learn`, `lightgbm`.
- Kept: `torch==2.9.1` (from the cu128 index), `numpy`, `pandas`, `pyarrow`,
  `matplotlib`, `requests`.

`scikit-learn` covers splits, scalers, metrics, baselines. `lightgbm` gives a strong
tabular baseline and a second model the agent can iterate on.

### uv.lock

Regenerated. Net package change: −many small LLM-stack transitives, +`joblib`,
`scikit-learn`, `scipy`, `threadpoolctl`, `lightgbm`.

### Rebuild note

The dockerfile builds the in-image `.venv` by cloning the GitHub repo and running
`uv sync`. For that `.venv` to reflect the new dependency set, the new
`pyproject.toml` + `uv.lock` must be pushed to the GitHub repo before rebuild. The
runtime `.venv` actually used inside the container comes from the mounted host repo,
which is already in sync.

---

## 5. Code changes (priority 2, after rebuild)

### prepare.py (read-only, rewritten)

Same role it had before: fixed constants, one-time data prep, runtime data utilities,
and the fixed evaluation function.

- Load `all_data.pkl.gz`.
- Drop constant columns, drop `hw_arch`, drop `group_id`, drop `context_name`.
- Encode `model_name` as integer ids; persist the vocab.
- Standardize numeric features (per-column z-score); persist the scaler.
- Target: `y = log(bottleneck_fps)`, then standardize for training.
- Group-aware split by `context_name`, ~15% of contexts to val.
- Persist train/val tensors and metadata under `~/.cache/autoresearch/oracle/`.
- Provide:
  - `TIME_BUDGET = 300`,
  - `make_dataloader(split, batch_size)` (shuffled for train),
  - `evaluate_rmsle(model, val_loader)` — the fixed metric.

### train.py (the agent's edit surface, rewritten)

Keep the structural skeleton: time-budget loop, prefetch, fast-fail on NaN/explosion,
GC freeze trick, summary printout. Swap the model and loss.

- Model: small MLP. Default `Linear(in, 256) -> GELU -> Dropout -> Linear(256, 256)
  -> GELU -> Linear(256, 1)`, plus `nn.Embedding` for `model_name` concatenated with
  numeric features.
- Loss: MSE on standardized log-target.
- Optimizer: AdamW. Muon is dropped from the baseline; it can come back as an agent
  experiment if useful.
- Summary block prints `val_rmsle` in place of `val_bpb`, plus the same training
  seconds / peak VRAM / num params / etc.

### program.md (the human-edited agent instructions, rewritten)

- Keep: branch ritual, `results.tsv` schema and protocol, keep/discard/crash logic,
  the autonomous "NEVER STOP" loop.
- Rename the metric to `val_rmsle`.
- Replace LLM-specific guidance (model size, vocab, `MAX_SEQ_LEN`, `val_bpb`) with
  tabular guidance: network depth/width, dropout, embedding dim, feature engineering
  (log/sqrt transforms, dropping noisy features, interactions), regularization,
  ensembling, lightgbm-style alternatives.

`results.tsv` columns:

```
commit  val_rmsle  memory_gb  status  description
```

### README.md

Light retitle and intro update. Non-blocking; comes last.

---

## 6. Sequence of work

1. Environment edits to `pyproject.toml` / `dockerfile` / `uv.lock` / host `.venv` —
   done.
2. Container rebuild — user.
3. Rewrite `prepare.py` (data prep + fixed eval).
4. Rewrite `train.py` (small MLP + log-MSE baseline).
5. Rewrite `program.md` for the tabular oracle context.
6. Establish baseline on the new pipe, log to `results.tsv`, hand off to the
   autonomous experiment loop.

---

## 7. Decisions locked (overridable)

- Metric: **RMSLE** on `bottleneck_fps`.
- Split: **group split by `context_name`**, ~15% held out.
- `context_name` is **not** a feature.
- Categorical input: `model_name` only, via `nn.Embedding`.
- Baseline model: small MLP, AdamW.
- Additional baseline available to the agent: LightGBM.
