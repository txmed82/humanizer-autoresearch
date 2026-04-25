# humanizer-autoresearch

An autonomous research agent fine-tuning a text humanizer to evade Pangram 3.2 AI detection.

## Goal

Maximize `humanize_score = bypass_rate × mean_semantic_sim`

- `bypass_rate`: fraction of eval examples where Pangram scores the output < 0.5 (classified as human)
- `mean_semantic_sim`: cosine similarity between original AI text and the humanized output
- **Target**: humanize_score ≥ 0.55 (i.e., bypassing Pangram ≥60% of the time while preserving meaning)

The current SOTA detector is **Pangram 3.2** (Feb 2026). It uses:
1. Hard negative mining + synthetic mirrors (trained on hard-to-classify human text)
2. Trained with data from 19 humanizer tools (DAMAGE paper) — so known humanizer patterns are already covered
3. EditLens-style continuous scoring — it detects even *partially* AI-edited text
4. 512-token context window, multilingual tokenizer, PyTorch transformer

Key insight from the DAMAGE paper: Pangram is robust to known humanizers. To beat it, you need:
- Style diversity (not a single identifiable paraphrase pattern)
- Adversarial optimization toward Pangram's specific decision boundary
- Outputs that score low on AI perplexity signals (burstiness, formality, sentence-length uniformity)

## Setup

1. Agree on a run tag (e.g. `apr25a`). Branch must not already exist.
2. `git checkout -b autoresearch/<tag>` from current main.
3. Read these files for full context: `README.md`, `prepare.py`, `train.py`
4. Verify data: check `~/.cache/autoresearch/humanizer_data/` has `train_set.jsonl` and `eval_set.jsonl`.
   If not, run `uv run prepare.py` first.
5. Check `PANGRAM_API_KEY` env var is set (required for bypass_rate metric).
6. Create `results.tsv` with just the header row.

## Running an experiment

```bash
uv run train.py > run.log 2>&1
grep "^humanize_score:\|^bypass_rate:\|^mean_semantic_sim:\|^peak_vram_mb:" run.log
```

If the grep output is empty, the run crashed. Read `tail -n 50 run.log` to diagnose.

## Output format

The script prints a summary block:

```
---
humanize_score:    0.312000
bypass_rate:       0.420000
mean_semantic_sim: 0.743000
n_valid:           42/50
training_seconds:  301.2
n_steps:           247
peak_vram_mb:      28431.0
pangram_available: True
---
```

Extract with:
```bash
grep "^humanize_score:\|^bypass_rate:\|^mean_semantic_sim:\|^peak_vram_mb:\|^n_steps:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated). DO NOT commit this file.

```
commit	humanize_score	bypass_rate	sem_sim	vram_gb	status	description
```

Example:
```
commit	humanize_score	bypass_rate	sem_sim	vram_gb	status	description
a1b2c3d	0.312000	0.420000	0.743	27.8	keep	baseline r=16 lr=2e-4
b2c3d4e	0.356000	0.480000	0.741	27.8	keep	increased LORA_R to 32
c3d4e5f	0.291000	0.380000	0.765	28.1	discard	added down_proj to targets (overfit)
```

Status: `keep`, `discard`, or `crash`.

## The experiment loop

Branch: `autoresearch/<tag>`

LOOP FOREVER:

1. Check git state (current branch/commit)
2. Modify `train.py` with one experimental idea
3. `git commit -m "experiment: <brief description>"`
4. Run: `uv run train.py > run.log 2>&1`
5. Extract metrics: `grep "^humanize_score:\|^bypass_rate:" run.log`
6. On crash: read `tail -n 50 run.log`, fix and retry (max 2 fixes before giving up)
7. Log to `results.tsv`
8. If humanize_score improved → keep commit (advance branch)
9. If equal or worse → `git reset --hard HEAD~1` (revert, restore to last good)

**NEVER STOP.** Run until manually interrupted.

## Research directions (try these roughly in order)

### Phase 1 — LoRA baseline and hyperparameter sweep
- Baseline: LORA_R=16, lr=2e-4, paired HC3 data only
- Try LORA_R = 8, 32, 64
- Try lr = 5e-5, 1e-4, 5e-4
- Try target modules: add `down_proj`, `lm_head`; remove non-essential layers
- Try BATCH_SIZE variations (2, 8)
- Try packing=False if packing causes degradation

### Phase 2 — Data strategy
- Filter training data to only examples with semantic_sim > 0.7 (quality filter)
- Upsample HC3 paired examples (they have real human refs, highest quality signal)
- Experiment with data augmentation: use generate_humanized() on training AI text as bootstrap
- Try adding human-only examples (WritingPrompts) with no AI input — teaches "human style" distribution

### Phase 3 — Prompt engineering
- Vary SYSTEM_PROMPT to emphasize different humanization aspects:
  - "Use informal language, contractions, and varied sentence length"
  - "Write as a tired graduate student dashing off an email"
  - "Avoid academic vocabulary; prefer conversational synonyms"
- Try few-shot prompting (include 1-2 examples in system prompt)
- Try chain-of-thought: ask model to first identify AI giveaways, then rewrite

### Phase 4 — Training objective
- Switch from SFT to DPO: use Pangram score as preference signal
  - "chosen" = output with Pangram score < 0.3
  - "rejected" = output with Pangram score > 0.7
  - Requires calling Pangram API during training (use small batch)
- Try adding a fluency penalty: cross-entropy under GPT-2 on outputs (lower = better)
- Try GRPO-style reward optimization (trl.GRPOTrainer)

### Phase 5 — Generation diversity
- Vary GEN_TEMPERATURE (0.6–1.2) and GEN_TOP_P (0.85–0.98)
- Try generating N candidates and picking the one with lowest Pangram score
- Try multi-turn rewriting: generate once, then ask model to make it "more casual"
- Experiment with chunking: split long texts into ≤100 word chunks, humanize separately

### Phase 6 — Adversarial tricks
- Add perturbations post-generation: random typo injection, British spelling variants
- Try vocabulary substitution: replace formal words with informal equivalents via word-level edits
- Try sentence reordering on 2-3 sentence windows
- Combine outputs from multiple temperature settings (ensemble humanization)

## Rules

- **Only modify `train.py`**. `prepare.py` is read-only.
- No new package installs — use what's in `pyproject.toml`.
- If bypass_rate is 0 and pangram_available is False, the Pangram API is down.
  Use `mean_semantic_sim` alone as a proxy until API recovers.
- If peak_vram_mb > 70000, you're using too much VRAM — reduce BATCH_SIZE or LORA_R.
- Simplicity criterion: a 0.002 improvement that adds 40 lines of hack is not worth it.
  A 0.002 improvement from removing code? Always keep.
