"""
prepare.py — fixed constants, one-time data prep, and runtime evaluation utilities.
DO NOT MODIFY. The agent only modifies train.py.

Provides:
  - Constants: paths, model name, time budget
  - Data prep: download_data() builds training/eval splits
  - Eval: evaluate_bypass() calls Pangram API on eval set, returns humanize_score
"""

import os
import json
import time
import hashlib
import requests
import numpy as np
from pathlib import Path
from datasets import load_dataset, Dataset

# ── Constants ────────────────────────────────────────────────────────────────

CACHE_DIR         = Path(os.environ.get("HF_HOME", "/workspace/hf_cache"))
DATA_DIR          = CACHE_DIR / "humanizer_data"
EVAL_FILE         = DATA_DIR / "eval_set.jsonl"
TRAIN_FILE        = DATA_DIR / "train_set.jsonl"
BASE_MODEL        = "meta-llama/Llama-3.1-8B-Instruct"
TRAINING_SECONDS  = 300          # 5-minute wall-clock training budget
EVAL_N            = 50           # number of examples to eval against Pangram API
SEMANTIC_SIM_MIN  = 0.60         # minimum semantic similarity to count as valid
MAX_SEQ_LEN       = 512          # token limit per example

PANGRAM_API_KEY   = os.environ.get("PANGRAM_API_KEY", "")
PANGRAM_API_URL   = "https://api.pangram.com/v1/text"

# ── Data preparation ─────────────────────────────────────────────────────────

def _make_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def download_data():
    """
    Download and prepare training + eval data.
    Creates DATA_DIR/train_set.jsonl and eval_set.jsonl.

    Training format: {"ai_text": str, "human_ref": str}
      ai_text   — AI-generated passage
      human_ref — reference human rewrite (for SFT; agent can ignore and use its own generation strategy)

    Eval format: {"ai_text": str, "source": str}
      eval set has NO human reference — we only measure Pangram bypass rate on generated outputs.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if EVAL_FILE.exists() and TRAIN_FILE.exists():
        print(f"Data already prepared at {DATA_DIR}. Skipping.")
        _print_data_stats()
        return

    print("Downloading datasets...")

    train_examples = []
    eval_examples  = []

    # ── Source 1: RAID benchmark (diverse LLM + domain coverage) ─────────────
    try:
        raid = load_dataset("liamdugan/raid", split="train", trust_remote_code=True)
        raid_ai = [r for r in raid if r.get("label") == "ai" and len(r.get("generation", "")) > 100]
        for r in raid_ai[:3000]:
            train_examples.append({"ai_text": r["generation"], "human_ref": "", "source": "raid"})
        print(f"  RAID: {len(raid_ai[:3000])} AI examples")
    except Exception as e:
        print(f"  RAID unavailable: {e}")

    # ── Source 2: HC3 (human/ChatGPT Q&A pairs) ──────────────────────────────
    try:
        hc3 = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
        for row in hc3:
            ai_answers = row.get("chatgpt_answers", [])
            human_answers = row.get("human_answers", [])
            if ai_answers and human_answers:
                ai_text = ai_answers[0]
                human_ref = human_answers[0]
                if len(ai_text) > 80 and len(human_ref) > 80:
                    train_examples.append({
                        "ai_text": ai_text,
                        "human_ref": human_ref,
                        "source": "hc3"
                    })
        print(f"  HC3: {sum(1 for e in train_examples if e['source']=='hc3')} paired examples")
    except Exception as e:
        print(f"  HC3 unavailable: {e}")

    # ── Source 3: WritingPrompts (human creative writing for style reference) ─
    try:
        wp = load_dataset("euclaise/writingprompts", split="train")
        for row in wp[:2000]:
            story = row.get("story", "")
            if len(story) > 150:
                train_examples.append({"ai_text": "", "human_ref": story, "source": "writingprompts"})
        print(f"  WritingPrompts: {sum(1 for e in train_examples if e['source']=='writingprompts')} human examples")
    except Exception as e:
        print(f"  WritingPrompts unavailable: {e}")

    # ── Build eval set (AI text only, no human ref) ───────────────────────────
    ai_examples = [e for e in train_examples if e["ai_text"] and e["source"] != "writingprompts"]
    np.random.seed(42)
    indices = np.random.choice(len(ai_examples), min(EVAL_N, len(ai_examples)), replace=False)
    eval_set = [ai_examples[i] for i in indices]

    # Remove eval examples from training to prevent leakage
    eval_texts = {e["ai_text"] for e in eval_set}
    train_set = [e for e in train_examples if e["ai_text"] not in eval_texts]

    with open(TRAIN_FILE, "w") as f:
        for ex in train_set:
            f.write(json.dumps(ex) + "\n")

    with open(EVAL_FILE, "w") as f:
        for ex in eval_set:
            f.write(json.dumps(ex) + "\n")

    print(f"\nData ready: {len(train_set)} train, {len(eval_set)} eval")
    _print_data_stats()


def _print_data_stats():
    train = [json.loads(l) for l in open(TRAIN_FILE)]
    eval_ = [json.loads(l) for l in open(EVAL_FILE)]
    by_source = {}
    for ex in train:
        by_source[ex["source"]] = by_source.get(ex["source"], 0) + 1
    print(f"Train: {len(train)} | Eval: {len(eval_)}")
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")


def load_train_data() -> list[dict]:
    assert TRAIN_FILE.exists(), "Run prepare.py first to download data."
    return [json.loads(l) for l in open(TRAIN_FILE)]


def load_eval_data() -> list[dict]:
    assert EVAL_FILE.exists(), "Run prepare.py first to download data."
    return [json.loads(l) for l in open(EVAL_FILE)]


# ── Evaluation harness ────────────────────────────────────────────────────────

def _call_pangram(text: str) -> float | None:
    """Call Pangram API. Returns AI probability score (1.0 = definitely AI, 0.0 = definitely human)."""
    if not PANGRAM_API_KEY:
        return None
    try:
        resp = requests.post(
            PANGRAM_API_URL,
            headers={
                "Authorization": f"Bearer {PANGRAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"text": text},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Pangram returns {"score": float, "label": "ai"|"human"}
            return float(data.get("score", 0.5))
    except Exception as e:
        print(f"  Pangram API error: {e}")
    return None


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Fast sentence-level cosine similarity using a cached embedding model.
    Loaded once globally to avoid reload overhead during eval.
    """
    global _SIM_MODEL
    if _SIM_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SIM_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embs = _SIM_MODEL.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))

_SIM_MODEL = None


def evaluate_bypass(generated_texts: list[str]) -> dict:
    """
    Primary eval function. Called by train.py after the training run.

    Args:
        generated_texts: list of humanized outputs, one per eval example.
                         Must be same length as load_eval_data().

    Returns dict with:
        humanize_score   — composite metric (higher is better, range 0–1)
                           = bypass_rate * mean_semantic_sim (only counting valid outputs)
        bypass_rate      — fraction where Pangram score < 0.5 (classified as human)
        mean_semantic_sim — mean cosine sim between original AI text and humanized output
        n_valid          — count of outputs with semantic_sim >= SEMANTIC_SIM_MIN
        pangram_scores   — list of raw Pangram scores (None if API unavailable)
        sim_scores       — list of cosine similarity scores
    """
    eval_data = load_eval_data()
    assert len(generated_texts) == len(eval_data), (
        f"Expected {len(eval_data)} generated texts, got {len(generated_texts)}"
    )

    pangram_scores = []
    sim_scores     = []

    for i, (ex, gen) in enumerate(zip(eval_data, generated_texts)):
        ai_text = ex["ai_text"]
        sim = _semantic_similarity(ai_text, gen)
        sim_scores.append(sim)

        score = _call_pangram(gen)
        pangram_scores.append(score)

        if (i + 1) % 10 == 0:
            valid_pg = [s for s in pangram_scores if s is not None]
            if valid_pg:
                bypass_so_far = sum(1 for s in valid_pg if s < 0.5) / len(valid_pg)
                print(f"  [{i+1}/{len(eval_data)}] bypass_rate_so_far={bypass_so_far:.3f}, "
                      f"mean_sim={np.mean(sim_scores):.3f}")

    valid_sim    = [sim_scores[i] for i in range(len(sim_scores))
                    if sim_scores[i] >= SEMANTIC_SIM_MIN]
    n_valid      = len(valid_sim)
    mean_sim     = float(np.mean(sim_scores)) if sim_scores else 0.0

    valid_pg     = [s for s in pangram_scores if s is not None]
    bypass_rate  = (sum(1 for s in valid_pg if s < 0.5) / len(valid_pg)) if valid_pg else 0.0

    humanize_score = bypass_rate * max(0.0, mean_sim) if valid_pg else 0.0

    return {
        "humanize_score":    round(humanize_score, 6),
        "bypass_rate":       round(bypass_rate, 6),
        "mean_semantic_sim": round(mean_sim, 6),
        "n_valid":           n_valid,
        "n_eval":            len(eval_data),
        "pangram_available": len(valid_pg) > 0,
        "pangram_scores":    [round(s, 4) if s is not None else None for s in pangram_scores],
        "sim_scores":        [round(s, 4) for s in sim_scores],
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_data()
    print("\nVerifying eval harness (no Pangram call — just semantic sim check)...")
    eval_data = load_eval_data()
    dummy_outputs = [ex["ai_text"] for ex in eval_data]  # identical = should score 1.0 sim
    sims = [_semantic_similarity(ex["ai_text"], ex["ai_text"]) for ex in eval_data[:3]]
    print(f"Self-similarity check (should be ~1.0): {sims}")
    print("prepare.py OK.")
