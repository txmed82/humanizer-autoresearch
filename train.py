"""
train.py — humanizer LoRA fine-tuning. THIS FILE IS MODIFIED BY THE AGENT.

Goal: produce the highest humanize_score = bypass_rate × semantic_sim
      where bypass_rate = fraction of eval examples Pangram scores < 0.5 (human).

Architecture:
  Base model  : Llama-3.1-8B-Instruct (4-bit NF4 QLoRA)
  Adapter     : LoRA on q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj
  Task        : Instruction-tuned text rewriting (AI → human-sounding)
  Time budget : TRAINING_SECONDS seconds wall-clock (from prepare.py)

After training, generates outputs on eval set and calls evaluate_bypass().
"""

import os, sys, time, json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from prepare import (
    BASE_MODEL, TRAINING_SECONDS, MAX_SEQ_LEN, CACHE_DIR,
    load_train_data, load_eval_data, evaluate_bypass,
)

# ── Hyperparameters (agent modifies this section) ─────────────────────────────

LORA_R           = 16          # LoRA rank — try 8, 16, 32, 64
LORA_ALPHA       = 32          # LoRA alpha — usually 2× rank
LORA_DROPOUT     = 0.05
LORA_TARGET_MODS = [           # which projection layers to adapt
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj",
]

LEARNING_RATE    = 2e-4        # try 1e-4, 2e-4, 5e-4
BATCH_SIZE       = 4           # per-device train batch
GRAD_ACCUM       = 4           # effective batch = BATCH_SIZE * GRAD_ACCUM = 16
WARMUP_RATIO     = 0.05
LR_SCHEDULER     = "cosine"    # "linear", "cosine", "constant"
MAX_GRAD_NORM    = 1.0

# Prompt template — controls how the model learns to rewrite
SYSTEM_PROMPT = (
    "You are a skilled human writer. Rewrite the following AI-generated text so it "
    "sounds naturally human: vary sentence length and structure, use informal connectors, "
    "add minor imperfections, avoid formulaic transitions. Preserve all meaning exactly."
)

# Data strategy: which training sources to use
USE_PAIRED_DATA  = True        # use (ai_text, human_ref) pairs when available
USE_AI_ONLY_DATA = True        # include ai-only examples with synthetic human target
MAX_TRAIN_EXAMPLES = 2000      # cap training size for time budget

# Generation strategy during eval
GEN_TEMPERATURE  = 0.8
GEN_TOP_P        = 0.92
GEN_MAX_NEW_TOKENS = 512
GEN_REPETITION_PENALTY = 1.1

# ── End hyperparameters ───────────────────────────────────────────────────────


def build_prompt(ai_text: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{ai_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def build_training_example(ai_text: str, human_ref: str, tokenizer) -> dict:
    prompt = build_prompt(ai_text)
    full   = prompt + human_ref + "<|eot_id|>"
    return {"text": full}


def build_dataset(tokenizer) -> Dataset:
    raw = load_train_data()
    examples = []

    for ex in raw[:MAX_TRAIN_EXAMPLES]:
        ai   = ex.get("ai_text", "").strip()
        ref  = ex.get("human_ref", "").strip()

        if USE_PAIRED_DATA and ai and ref:
            examples.append(build_training_example(ai, ref, tokenizer))
        elif USE_AI_ONLY_DATA and ai and not ref:
            # Without a human reference, use the AI text itself as target.
            # This teaches the model the rewriting pattern even without a gold label.
            # The agent may want to replace this with a better synthetic strategy.
            examples.append(build_training_example(ai, ai, tokenizer))

    print(f"Training on {len(examples)} examples")
    return Dataset.from_list(examples)


def load_model_and_tokenizer():
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        cache_dir=CACHE_DIR,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODS,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def run_training(model, tokenizer, train_dataset, output_dir: Path):
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=100,               # time budget will cut this off
        max_steps=-1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        dataloader_num_workers=2,
        report_to="none",
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=True,                       # pack examples for throughput
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Timed training loop — respects TRAINING_SECONDS budget
    t0 = time.time()
    print(f"\nStarting training (budget: {TRAINING_SECONDS}s)...")

    class TimedCallback:
        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - t0 >= TRAINING_SECONDS:
                control.should_training_stop = True

    from transformers import TrainerCallback
    class _CB(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - t0 >= TRAINING_SECONDS:
                print(f"\n  Time budget reached at step {state.global_step}. Stopping.")
                control.should_training_stop = True

    trainer.add_callback(_CB())
    trainer.train()

    training_seconds = time.time() - t0
    print(f"Training done: {training_seconds:.1f}s, {trainer.state.global_step} steps")
    return trainer.state.global_step, training_seconds


@torch.no_grad()
def generate_humanized(model, tokenizer, eval_data: list[dict]) -> list[str]:
    model.eval()
    outputs = []
    for ex in eval_data:
        prompt = build_prompt(ex["ai_text"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            repetition_penalty=GEN_REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(text)
    return outputs


def main():
    output_dir = CACHE_DIR / "lora_adapter"

    print("Loading model and tokenizer...")
    t_load = time.time()
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model loaded in {time.time()-t_load:.1f}s")

    print("Building dataset...")
    train_dataset = build_dataset(tokenizer)

    print("Running training...")
    n_steps, training_seconds = run_training(model, tokenizer, train_dataset, output_dir)

    print("\nGenerating humanized outputs on eval set...")
    eval_data = load_eval_data()
    t_gen = time.time()
    generated = generate_humanized(model, tokenizer, eval_data)
    gen_seconds = time.time() - t_gen
    print(f"Generated {len(generated)} outputs in {gen_seconds:.1f}s")

    print("\nEvaluating against Pangram API...")
    results = evaluate_bypass(generated)

    # ── Print summary (agent reads this) ─────────────────────────────────────
    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print("\n---")
    print(f"humanize_score:    {results['humanize_score']:.6f}")
    print(f"bypass_rate:       {results['bypass_rate']:.6f}")
    print(f"mean_semantic_sim: {results['mean_semantic_sim']:.6f}")
    print(f"n_valid:           {results['n_valid']}/{results['n_eval']}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"n_steps:           {n_steps}")
    print(f"peak_vram_mb:      {vram_mb:.1f}")
    print(f"pangram_available: {results['pangram_available']}")
    print("---")


if __name__ == "__main__":
    main()
