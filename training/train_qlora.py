"""
QLoRA supervised fine-tuning (instruction tuning) of the base model.

    python training/train_qlora.py
    python training/train_qlora.py --max-steps 10     # quick smoke test

What is trained
---------------
* The base model (config.MODEL_NAME) is loaded in 4-bit NF4 and FROZEN.
* LoRA adapters (rank config.LORA_R) are inserted into the attention and MLP
  projection layers (config.LORA_TARGET_MODULES). ONLY these adapters are
  trained - typically ~1% of the parameters.
* Training data: instruction -> response pairs (data/train.jsonl) in the
  model's chat format. The loss is computed only on the response tokens.
* Output: the LoRA adapter in models/edu_tune_lora/ (NOT merged into the base
  model), plus logs in results/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from transformers.trainer_callback import PrinterCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

import config  # noqa: E402
from training.format_dataset import to_prompt_completion  # noqa: E402
from utils.device import OOM_HELP, can_use_4bit, get_compute_dtype, is_oom_error, print_device_info  # noqa: E402
from utils.model_loader import ModelLoadError, load_base_model  # noqa: E402


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
class HistoryCallback(TrainerCallback):
    """Prints epoch / step / train loss / val loss / LR and keeps a history."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step, "epoch": round(state.epoch or 0.0, 3)}
        if "loss" in logs:
            row.update(train_loss=logs["loss"], learning_rate=logs.get("learning_rate"))
            print(f"Epoch {row['epoch']:>5.2f} | Step {row['step']:>4}/{state.max_steps} | "
                  f"Training Loss {logs['loss']:.4f} | LR {logs.get('learning_rate', 0):.2e}")
        elif "eval_loss" in logs:
            row.update(val_loss=logs["eval_loss"])
            print(f"Epoch {row['epoch']:>5.2f} | Step {row['step']:>4}/{state.max_steps} | "
                  f"Validation Loss {logs['eval_loss']:.4f} | "
                  f"Perplexity {math.exp(min(logs['eval_loss'], 20)):.2f}")
        else:
            return
        self.rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        cols = ["step", "epoch", "train_loss", "val_loss", "learning_rate"]
        if not self.rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(self.rows)
        df = df.groupby("step", as_index=False).agg("last")   # merge train + eval rows per step
        for c in cols:
            if c not in df:
                df[c] = None
        return df[cols]


# ---------------------------------------------------------------------------
def count_parameters(model) -> tuple[int, int]:
    """(trainable, total). PEFT's helper correctly counts packed 4-bit weights."""
    trainable, total = model.get_nb_trainable_parameters()
    return int(trainable), int(total)


def load_data():
    for path in (config.TRAIN_FILE, config.VALIDATION_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/generate_dataset.py` and "
                f"`python scripts/prepare_dataset.py` first."
            )
    ds = load_dataset("json", data_files={"train": str(config.TRAIN_FILE),
                                          "validation": str(config.VALIDATION_FILE)})
    # Convert every record into a (prompt, completion) conversation.
    return ds.map(to_prompt_completion, remove_columns=ds["train"].column_names)


def build_model(use_4bit: bool):
    model, tokenizer = load_base_model()
    if use_4bit:
        # Casts norms to fp32, enables input gradients + gradient checkpointing
        # so a frozen 4-bit model can pass gradients to the LoRA adapters.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Freezes all base weights and inserts trainable low-rank matrices A, B.
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def plot_history(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    tr = df.dropna(subset=["train_loss"])
    va = df.dropna(subset=["val_loss"])
    ax.plot(tr["step"], tr["train_loss"], label="Training loss", color="#2563eb", linewidth=2)
    if not va.empty:
        ax.plot(va["step"], va["val_loss"], label="Validation loss", color="#ea580c",
                marker="o", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("EduTune AI - QLoRA fine-tuning")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2, 2)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for EduTune AI")
    parser.add_argument("--epochs", type=float, default=config.NUM_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N steps (smoke test)")
    args = parser.parse_args()

    info = print_device_info()
    use_4bit = config.USE_4BIT and can_use_4bit()
    if config.USE_4BIT and not use_4bit:
        print("WARNING: USE_4BIT=True but 4-bit needs a CUDA GPU + bitsandbytes. "
              "Falling back to regular LoRA (full-precision frozen base model). "
              "Training on CPU will be very slow.")
    if not info.cuda_available:
        print("WARNING: Training on CPU. Consider Qwen/Qwen2.5-0.5B-Instruct and --max-steps for a demo.")

    try:
        ds = load_data()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    n_train, n_val = len(ds["train"]), len(ds["validation"])

    print(f"\nLoading base model: {config.MODEL_NAME} "
          f"({'4-bit NF4 - QLoRA' if use_4bit else 'no quantisation - LoRA'})")
    try:
        model, tokenizer = build_model(use_4bit)
    except ModelLoadError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    trainable, total = count_parameters(model)
    pct = 100 * trainable / total
    print("\n" + "=" * 60)
    print("PARAMETER-EFFICIENT FINE-TUNING (PEFT / LoRA)")
    print("=" * 60)
    print(f"Total parameters:      {total:,}")
    print(f"Trainable parameters:  {trainable:,}")
    print(f"Trainable percentage:  {pct:.4f}%")
    print(f"Frozen parameters:     {total - trainable:,}")
    print("=" * 60 + "\n")

    effective_batch = config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS
    total_steps = (args.max_steps if args.max_steps > 0
                   else math.ceil(n_train / effective_batch) * math.ceil(args.epochs))
    bf16 = info.cuda_available and info.bf16_supported
    sft_config = SFTConfig(
        output_dir=str(config.CHECKPOINT_DIR),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_steps=max(1, int(total_steps * config.WARMUP_RATIO)),
        weight_decay=config.WEIGHT_DECAY,
        max_length=config.MAX_SEQ_LENGTH,
        completion_only_loss=True,          # learn the answers, not the questions
        bf16=bf16,
        fp16=info.cuda_available and not bf16,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",   # QLoRA paged optimiser
        logging_steps=config.LOGGING_STEPS,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="no",                 # we save the adapter explicitly at the end
        report_to="none",
        seed=config.SEED,
        disable_tqdm=True,
    )
    history = HistoryCallback()
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        callbacks=[history],
    )
    trainer.remove_callback(PrinterCallback)   # HistoryCallback prints a cleaner log

    print(f"Training on {n_train} examples, validating on {n_val}. "
          f"Effective batch size {effective_batch}, ~{total_steps} optimiser steps.\n")
    if info.cuda_available:
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    try:
        train_result = trainer.train()
        final_eval = trainer.evaluate()
    except Exception as exc:  # noqa: BLE001
        if is_oom_error(exc):
            print("\nERROR: " + OOM_HELP)
            sys.exit(1)
        raise
    runtime = time.time() - start

    # ---- Save the adapter (only the LoRA weights, a few tens of MB) -------
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(config.OUTPUT_DIR))
    tokenizer.save_pretrained(str(config.OUTPUT_DIR))

    # ---- Logs --------------------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = history.to_frame()
    df.to_csv(config.TRAINING_HISTORY_FILE, index=False)
    plot_history(df, config.TRAINING_PLOT_FILE)

    eval_loss = final_eval.get("eval_loss")
    summary = {
        "model_name": config.MODEL_NAME,
        "fine_tuning": "Supervised Fine-Tuning (instruction tuning)",
        "peft_method": "LoRA",
        "quantization": "4-bit NF4 (QLoRA)" if use_4bit else "None (LoRA)",
        "lora_r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT,
        "lora_target_modules": config.LORA_TARGET_MODULES,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": round(pct, 4),
        "train_examples": n_train,
        "validation_examples": n_val,
        "epochs": args.epochs if args.max_steps <= 0 else round(trainer.state.epoch, 3),
        "optimizer_steps": trainer.state.global_step,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": effective_batch,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "final_train_loss": round(train_result.training_loss, 4),
        "final_validation_loss": round(eval_loss, 4) if eval_loss is not None else None,
        "final_validation_perplexity": round(math.exp(eval_loss), 3) if eval_loss is not None else None,
        "runtime_minutes": round(runtime / 60, 2),
        "gpu": info.gpu_name or "CPU",
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if info.cuda_available else None,
        "compute_dtype": str(get_compute_dtype()).replace("torch.", ""),
        "adapter_path": str(config.OUTPUT_DIR.relative_to(config.ROOT_DIR)),
        "adapter_size_mb": dir_size_mb(config.OUTPUT_DIR),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    config.TRAINING_SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for k in ("final_train_loss", "final_validation_loss", "final_validation_perplexity",
              "runtime_minutes", "peak_gpu_memory_gb", "adapter_size_mb"):
        print(f"{k:<30}{summary[k]}")
    print(f"\nLoRA adapter saved to:   {config.OUTPUT_DIR}")
    print(f"Training history:        {config.TRAINING_HISTORY_FILE}")
    print(f"Loss curve:              {config.TRAINING_PLOT_FILE}")
    print(f"Summary:                 {config.TRAINING_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
