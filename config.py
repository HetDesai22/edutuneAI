"""
Central configuration for EduTune AI.

Every script (dataset generation, preprocessing, training, evaluation, app)
reads its settings from here, so the model, paths and hyper-parameters can be
changed in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_FILE = DATA_DIR / "instruction_dataset.jsonl"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALIDATION_FILE = DATA_DIR / "validation.jsonl"
KNOWLEDGE_BASE_FILE = RAW_DATA_DIR / "knowledge_base.json"

MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = MODELS_DIR / "edu_tune_lora"          # LoRA adapter (NOT a merged model)
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"         # intermediate trainer checkpoints

RESULTS_DIR = ROOT_DIR / "results"
TRAINING_HISTORY_FILE = RESULTS_DIR / "training_history.csv"
TRAINING_SUMMARY_FILE = RESULTS_DIR / "training_summary.json"
TRAINING_PLOT_FILE = RESULTS_DIR / "training_loss.png"
EVAL_METRICS_FILE = RESULTS_DIR / "eval_metrics.json"
COMPARISON_FILE = RESULTS_DIR / "comparison.json"

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
# A small instruction-capable causal LM that fits a consumer laptop GPU.
# Alternatives that work with the same code (adjust LORA_TARGET_MODULES if needed):
#   "Qwen/Qwen2.5-0.5B-Instruct"          (faster, weaker; good for CPU / <6 GB GPUs)
#   "Qwen/Qwen2.5-3B-Instruct"            (stronger; needs ~8 GB+ with 4-bit)
#   "meta-llama/Llama-3.2-1B-Instruct"    (gated on Hugging Face)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SEED = 42
VALIDATION_SPLIT = 0.10          # 90% train / 10% validation
REQUIRED_FIELDS = ("instruction", "input", "output")

# System message used for BOTH training and inference, so the only difference
# between the "base" and "fine-tuned" model at demo time is the LoRA adapter.
SYSTEM_PROMPT = (
    "You are EduTune AI, a patient and accurate teaching assistant for "
    "Artificial Intelligence, Machine Learning, Deep Learning and Generative AI."
)

# ---------------------------------------------------------------------------
# Training hyper-parameters (tuned for an 8 GB laptop GPU)
# ---------------------------------------------------------------------------
NUM_EPOCHS = 3
BATCH_SIZE = 2                       # per-device micro-batch
GRADIENT_ACCUMULATION_STEPS = 4      # effective batch = BATCH_SIZE * GRAD_ACCUM = 8
LEARNING_RATE = 2e-4                 # typical for LoRA (higher than full fine-tuning)
MAX_SEQ_LENGTH = 1024                # tokens; reduce to 512 if you run out of memory
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 5
EVAL_STEPS = 25
GRADIENT_CHECKPOINTING = True        # trades compute for a large memory saving

# ---------------------------------------------------------------------------
# LoRA / QLoRA
# ---------------------------------------------------------------------------
LORA_R = 16                # rank of the low-rank update matrices A (r x k) and B (d x r)
LORA_ALPHA = 32            # scaling: the update is multiplied by LORA_ALPHA / LORA_R
LORA_DROPOUT = 0.05
# Attention + MLP projection layers of Qwen2 / Llama-style decoders.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

USE_4BIT = True                      # QLoRA: load the frozen base model in 4-bit
BNB_4BIT_QUANT_TYPE = "nf4"          # NormalFloat4, the data type introduced by QLoRA
BNB_4BIT_USE_DOUBLE_QUANT = True     # also quantise the quantisation constants

# ---------------------------------------------------------------------------
# Generation (inference)
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
EVAL_MAX_NEW_TOKENS = 400            # used by evaluation/evaluate.py (greedy decoding)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
APP_TITLE = "EduTune AI"
APP_SUBTITLE = "Your Instruction-Tuned AI Teaching Assistant"
DISCLAIMER = (
    "EduTune AI is an educational assistant. Its answers may contain mistakes. "
    "Verify important academic information with textbooks, lecturers, or "
    "authoritative sources."
)
