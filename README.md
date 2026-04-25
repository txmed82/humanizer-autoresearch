# humanizer-autoresearch

Karpathy's autoresearch loop adapted to autonomously fine-tune a text humanizer targeting Pangram 3.2.

## Goal

Beat Pangram 3.2 AI detection ≥60% of the time while preserving semantic meaning.

`humanize_score = bypass_rate × mean_semantic_sim ≥ 0.55`

## How it works

Identical loop to [karpathy/autoresearch](https://github.com/karpathy/autoresearch):

- Agent reads `program.md` and modifies `train.py`
- Each run: 5-min LoRA fine-tuning of Llama-3.1-8B-Instruct (QLoRA, 4-bit)
- Eval: generate humanized outputs on 50 AI text examples → call Pangram API → compute score
- Keep if improved, revert if not, loop forever

Three files:
- `prepare.py` — fixed. Data download, eval harness, Pangram API wrapper. DO NOT MODIFY.
- `train.py` — agent modifies this. LoRA config, training objective, data strategy, generation params.
- `program.md` — research directions for the agent.

## Quick start

```bash
# Requirements: single H100 (80GB), Python 3.10+, uv
# Set PANGRAM_API_KEY env var

# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (includes flash-attn for H100)
pip install flash-attn --no-build-isolation -q
uv sync

# 3. Download data and pre-cache model weights (one-time, ~10 min)
uv run prepare.py

# 4. Run a single baseline experiment
uv run train.py > run.log 2>&1
grep "^humanize_score:\|^bypass_rate:\|^mean_semantic_sim:" run.log
```

Then point an agent at `program.md` and let it run overnight.

## Pangram 3.2 context

- **Architecture**: Transformer classifier, 512-token context, multilingual tokenizer
- **Training**: Hard negative mining + synthetic mirrors; trained against 19 humanizers (DAMAGE paper)
- **Key difficulty**: Detects partially AI-edited text (EditLens technology); cross-humanizer generalization

See `program.md` for the full research roadmap.

## Cloud setup (RunPod)

```bash
bash setup.sh <YOUR_PANGRAM_API_KEY>
```

H100 80GB HBM3 at ~$2.99/hr. ~12 experiments/hour. 100 experiments overnight.
