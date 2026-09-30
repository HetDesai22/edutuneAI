# 🎓 EduTune AI — Instruction-Tuned AI Teaching Assistant

EduTune AI takes a general-purpose open-source language model (**Qwen2.5-1.5B-Instruct**) and
**specialises it into an AI/ML teaching assistant** with **instruction tuning**. It uses
**supervised fine-tuning (SFT)** with **PEFT/LoRA**, and **QLoRA** (a 4-bit quantised base model)
so training fits on a consumer laptop GPU.

The project covers the full pipeline: dataset generation → cleaning → QLoRA training → evaluation →
an interactive Streamlit teaching app. The app shows the **base model** and the **fine-tuned model**
answering side by side.

> ⚠️ EduTune AI is an educational assistant. Its answers may contain mistakes. Verify important
> academic information with textbooks, lecturers, or authoritative sources.

---

## Table of contents
1. [Problem statement](#problem-statement)
2. [Objective](#objective)
3. [Architecture](#architecture)
4. [What exactly is trained](#what-exactly-is-trained)
5. [Technologies](#technologies)
6. [Project structure](#project-structure)
7. [Installation](#installation)
8. [Dataset generation & preparation](#dataset-generation--preparation)
9. [Training](#training)
10. [Evaluation](#evaluation)
11. [Run the application](#run-the-application)
12. [Testing](#testing)
13. [Results](#results)
14. [Presentation / demo guide](#presentation--demo-guide)
15. [Concepts explained](#concepts-explained)
16. [Hardware limits & troubleshooting](#hardware-limits--troubleshooting)
17. [Limitations & honest notes](#limitations--honest-notes)

---

## Problem statement
General-purpose LLMs are trained to be broadly helpful, but they are not optimised for a particular
**teaching style** or **domain**. Ask one to "explain LSTM to a beginner" or "act as a viva
examiner" and the style, structure and depth can vary a lot. They also rarely follow a fixed format
that an app can rely on, such as MCQs with a parseable answer key.

## Objective
Build an **instruction-tuned educational assistant** for AI, ML, DL and Generative AI that:
* explains concepts in different **teaching modes** (simple, detailed, "explain like I'm 10",
  example-based, interview, viva, MCQ, numerical, revision);
* uses a consistent teaching structure: definitions, analogies, worked examples, key takeaways;
* can **quiz**, **examine (viva)** and **interview** a student and give feedback;
* is produced by **actual fine-tuning** (not prompt engineering) that runs on a laptop GPU.

## Architecture

```text
Educational knowledge base (37 topics, structured fields)
        ↓  scripts/generate_dataset.py
Instruction dataset  (instruction, input, output, category)   840 examples, 14 categories
        ↓  scripts/prepare_dataset.py
Cleaning (invalid / empty / duplicates) → 90% train / 10% validation (stratified)
        ↓  training/format_dataset.py  (model chat template, loss on response only)
Base LLM (Qwen2.5-1.5B-Instruct) loaded in 4-bit NF4 and FROZEN
        ↓  training/train_qlora.py   (TRL SFTTrainer + PEFT LoRA)
Fine-tuned LoRA adapter  → models/edu_tune_lora/   (~46 MB, not merged)
        ↓  evaluation/evaluate.py
Evaluation (validation loss, perplexity, side-by-side answers)
        ↓  utils/model_loader.py  (base + adapter; adapter can be switched off)
Teaching assistant  →  app.py (Streamlit web interface)
```

At inference time the app loads the base weights **once** and attaches the adapter. "Base Model"
answers come from the same model with the adapter **temporarily disabled** (`disable_adapter()`).
Both answers use the same prompt, so **the only difference between them is the trained adapter**.

## What exactly is trained

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (1.54 B parameters), set in `config.py` → `MODEL_NAME` |
| Method | Supervised fine-tuning (instruction tuning) with LoRA on a 4-bit base model = **QLoRA** |
| Frozen | **All** original model weights, stored in 4-bit NF4 with double quantisation |
| Trained | LoRA matrices A and B (rank 16, α = 32, dropout 0.05) on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` in all 28 decoder layers |
| Trainable parameters | **18,464,768 of 1,562,179,072 (1.18%)** |
| Training data | 756 instruction→response pairs (84 held out for validation) |
| Loss | Next-token cross-entropy on the **response tokens only** (the prompt is masked) |
| Optimiser | paged AdamW 8-bit, LR 2e-4, cosine schedule with warm-up, 3 epochs, effective batch 8 |
| Output | LoRA adapter only (`models/edu_tune_lora/`), **not merged** into the base model |

The model is trained to respond with a consistent teaching structure (headings, analogies, worked
examples, "key idea to remember"), to use the right format for each mode (viva feedback verdicts,
interview "what you got right / what you missed", MCQs with `**Answer:**` and `**Explanation:**`),
and to ask single viva/interview questions.

## Technologies

| Technology | Role in this project |
|---|---|
| **Python 3.11** | Language for the whole pipeline |
| **PyTorch** | Tensor computation and GPU execution (CUDA 12.8 build, needed for RTX 50-series GPUs) |
| **Hugging Face Transformers** | Loads the pre-trained model and tokenizer, chat templates, generation |
| **Hugging Face Datasets** | Loads the JSONL train/validation files for the trainer |
| **PEFT** | `LoraConfig`, `get_peft_model` (adds LoRA adapters, freezes the base), `PeftModel` (loads the adapter for inference) |
| **TRL** | `SFTTrainer` / `SFTConfig`: supervised fine-tuning with completion-only loss |
| **bitsandbytes** | 4-bit NF4 quantisation (QLoRA) and the paged 8-bit AdamW optimiser |
| **Accelerate** | Device placement and mixed precision under the hood of the Trainer |
| **Streamlit** | The web interface (chat, comparison, quiz, viva, interview, dashboard) |
| **Pandas / NumPy** | Training history, dataset statistics |
| **scikit-learn** | Stratified train/validation split |
| **Matplotlib** | Saves the loss curve (`results/training_loss.png`) |

## Project structure

```text
AI_Teaching_Assistant/            (= edu_tune_ai/)
├── app.py                        Streamlit application (6 pages)
├── config.py                     ALL settings: model, paths, hyper-parameters, LoRA, 4-bit
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml        UI theme
│
├── data/
│   ├── raw/knowledge_base.json   structured source content (exported by the generator)
│   ├── instruction_dataset.jsonl 840 generated examples
│   ├── train.jsonl               756 (90%)
│   └── validation.jsonl          84 (10%)
│
├── models/
│   └── edu_tune_lora/            trained LoRA adapter (adapter_config.json + adapter_model.safetensors)
│
├── training/
│   ├── train_qlora.py            QLoRA supervised fine-tuning
│   └── format_dataset.py         format_instruction(): the single formatting function
│
├── evaluation/
│   ├── evaluate.py               loss / perplexity + side-by-side generation
│   └── comparison.py             observable, non-judgemental response metrics
│
├── scripts/
│   ├── generate_dataset.py       builds the instruction dataset from the knowledge base
│   ├── prepare_dataset.py        cleaning, validation, stats, 90/10 split
│   ├── test_model.py             PASS/FAIL system tests
│   └── knowledge/                structured educational content (AI, ML, DL, GenAI, comparisons)
│
├── utils/
│   ├── device.py                 CUDA detection, dtype selection, OOM help
│   ├── model_loader.py           load_base_model(), load_finetuned_model(), TeachingModel
│   ├── prompts.py                teaching modes, build_prompt(), viva/interview/MCQ prompts
│   └── mcq.py                    MCQ parser + curated question bank
│
└── results/
    ├── training_history.csv      step, epoch, train loss, val loss, learning rate
    ├── training_summary.json     parameter counts, losses, runtime, GPU
    ├── training_loss.png         loss curve
    ├── eval_metrics.json         base vs fine-tuned loss & perplexity
    └── comparison.json           base vs fine-tuned answers to fixed questions
```

## Installation

Requirements: Python 3.11, an NVIDIA GPU with ≥ 6 GB VRAM (8 GB recommended) and a recent driver.
A CPU-only install works for inference, but slowly, and cannot do 4-bit QLoRA training.

```bash
# 1. Create and activate a virtual environment
python -m venv venv                  # on Windows with several Pythons: py -3.11 -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```
Linux / macOS:
```bash
source venv/bin/activate
```

```bash
# 2. Install PyTorch with CUDA (cu128 supports RTX 20xx–50xx GPUs)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Install the rest
pip install -r requirements.txt

# 4. Check the GPU is detected
python utils/device.py
```

Expected output:
```text
CUDA Available: True
GPU: NVIDIA GeForce RTX 5050 Laptop GPU
```

The base model (~3 GB) downloads automatically from Hugging Face on first use.

## Dataset generation & preparation

```bash
python scripts/generate_dataset.py     # → data/instruction_dataset.jsonl (840 examples)
python scripts/prepare_dataset.py      # → data/train.jsonl, data/validation.jsonl + statistics
```

**How the dataset is built.** Writing 800+ examples by hand would give inconsistent quality, so each
of the **37 topics** (AI fundamentals, agents, search, knowledge representation, supervised and
unsupervised learning, regression, classification, clustering, decision trees, random forests, SVM,
KNN, gradient descent, overfitting, underfitting, ANN, backpropagation, activation functions,
vanishing/exploding gradients, CNN, RNN, LSTM, GRU, GAN, VAE, transformers, attention, LLM, prompt
engineering, fine-tuning, LoRA, QLoRA, PEFT, instruction tuning, RLHF, DPO) is written **once** as
structured, hand-written content in `scripts/knowledge/`. The fields are: definition, beginner
explanation, ELI10 explanation, analogy, how it works, worked example, applications, pros/cons, key
points, viva Q&A, interview Q&A, MCQs, and for many topics a numerical problem and a troubleshooting
case. There are also **18 comparison tables** (RNN vs LSTM, LoRA vs QLoRA, RLHF vs DPO, …).

The generator turns each field into instruction→response pairs in **14 categories**
(`definition, simple_explanation, detailed_explanation, analogy, example, comparison,
advantages_disadvantages, application, numerical, mcq, viva, interview, revision, troubleshooting`).
It varies the phrasing with a seeded RNG and uses the **same instruction templates as the app**
(`utils/prompts.py`). Some examples use the `input` field, for example "evaluate this student's viva
answer" or "simplify this explanation".

Record format:
```json
{"instruction": "Explain LSTM using a simple real-world analogy.", "input": "", "output": "**Analogy:** Imagine you are reading a long story...", "category": "analogy", "topic": "lstm", "domain": "Deep Learning"}
```

`prepare_dataset.py` removes malformed JSON, invalid records, empty instructions and outputs, and
duplicates (case- and whitespace-insensitive). It validates the required fields, prints statistics
and makes a **stratified** 90/10 split.

## Training

```bash
python training/train_qlora.py                # full training (~17 min on an RTX 5050 Laptop GPU)
python training/train_qlora.py --max-steps 10  # quick smoke test
```

The script prints the hardware, then the **PEFT parameter report**:
```text
Total parameters:      1,562,179,072
Trainable parameters:  18,464,768
Trainable percentage:  1.1820%
```
During training it logs **epoch, step, training loss, validation loss (+ perplexity) and learning
rate**. It saves:
* `models/edu_tune_lora/` — the LoRA adapter (+ tokenizer files)
* `results/training_history.csv`, `results/training_loss.png`, `results/training_summary.json`

All hyper-parameters are in `config.py`:
`MODEL_NAME, OUTPUT_DIR, TRAIN_FILE, VALIDATION_FILE, NUM_EPOCHS, BATCH_SIZE,
GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, MAX_SEQ_LENGTH, LORA_R, LORA_ALPHA, LORA_DROPOUT,
LORA_TARGET_MODULES, USE_4BIT, …`

## Evaluation

```bash
python evaluation/evaluate.py                   # metrics + side-by-side answers
python evaluation/evaluate.py --skip-generation # metrics only (faster)
```

* **Quantitative** (`results/eval_metrics.json`): validation **loss** and **perplexity** for the
  base and fine-tuned models on the 84 held-out examples. Only **response tokens** are counted,
  with a per-category breakdown. Classification accuracy is deliberately **not** used, because it
  is not meaningful for free-form generated explanations.
* **Qualitative** (`results/comparison.json`): both models answer 14 fixed questions with greedy
  decoding. These include *Explain RNN, Explain LSTM, What is vanishing gradient?, Compare RNN and
  LSTM, Explain CNN to a beginner, What is overfitting?, Explain LoRA, Explain QLoRA*, several
  teaching modes, and 3 topics that are **not in the training data** (batch normalization, dropout,
  word embeddings) to check whether the style transfers. For each pair the file records observable
  differences (length, structure, analogy/example use, MCQ format compliance). **No automatic
  "winner" is declared.**

## Run the application

```bash
streamlit run app.py
```
Open http://localhost:8501. The first load takes ~10–20 s (model + adapter); after that the model is
cached.

| Page | What it does |
|---|---|
| 💬 **AI Tutor** | Chat with the fine-tuned model. The sidebar selects the **teaching mode** and **topic**. Quick actions: 📚 Explain · 💡 Give Example · 🧠 Give Analogy · 📝 Quiz Me · 🎯 Viva Questions · 💼 Interview Questions · 🔄 Simplify Answer |
| ⚖️ **Base vs Fine-Tuned** | The same question to both models, side by side, with observable differences; also browses saved evaluation results |
| 📝 **Quiz Mode** | One MCQ at a time, A–D answer, *Correct!* / explanation, running **Score: x / y**. Questions are model-generated (parsed) or from the curated bank |
| 🎤 **Viva Mode** | The AI examiner asks one question, you answer, and it gives feedback (verdict, what you got right, what to add, model answer) |
| 💼 **Interview Mode** | Interview question → your answer → *What you got right / What you missed / Suggested improved answer* |
| ❓ **MCQ Generator** | Topic, number of questions, Easy/Medium/Hard → MCQs with answers and explanations |

**Inference from Python** (without the UI):
```python
from utils.model_loader import TeachingModel
from utils.prompts import build_prompt

tm = TeachingModel()                                    # base + LoRA adapter, loaded once
q = build_prompt("What is an LSTM?", "Explain Like I'm 10")
print(tm.generate(q, use_adapter=True))                 # fine-tuned
print(tm.generate(q, use_adapter=False))                # base model
```

## Testing

```bash
python scripts/test_model.py          # full test (loads the model)
python scripts/test_model.py --quick  # no model loading
```
```text
[PASS] GPU detection
[PASS] Dataset loading
[PASS] Dataset formatting
[PASS] Prompt building
[PASS] MCQ parser
[PASS] Base model loading
[PASS] Fine-tuned model loading
[PASS] Inference
```

## Results

Measured on an **NVIDIA GeForce RTX 5050 Laptop GPU (8 GB)**. All numbers come from `results/`.

**Training** (`results/training_summary.json`, `results/training_history.csv`)

| Metric | Value |
|---|---|
| Trainable / total parameters | 18,464,768 / 1,562,179,072 (**1.18%**) |
| Optimiser steps / epochs | 285 / 3 |
| Training time | **16.5 min** |
| Peak GPU memory | **2.81 GB** |
| Adapter size | **46 MB** (vs ~3 GB for a full 16-bit copy of the model) |
| Validation loss (trainer): first eval at step 25 → final | 1.55 → **0.86** (perplexity 4.70 → **2.37**) |

See `results/training_loss.png`. The validation loss plateaus from about step 175 (0.867 → 0.866)
while the training loss keeps falling to about 0.4. The widening gap in epoch 3 is the **start of
memorisation**. Validation loss does not rise, so the model is not yet harmed, but 2 epochs would
give almost the same validation loss. That is a good talking point on the overfitting/early-stopping
trade-off.

**Evaluation** (`results/eval_metrics.json`): 84 held-out examples, response tokens only.

| | Base model | Fine-tuned model |
|---|---|---|
| Validation loss | 2.153 | **0.909** |
| Perplexity | 8.61 | **2.48** (−71%) |

**Qualitative** (`results/comparison.json`): typical observed differences.
* **Format following:** for *"Create 2 multiple-choice questions on RNN"* the base model wrote
  questions **without an answer key** (0/2 parseable). The fine-tuned model produced 2/2 in the
  exact `Question / A–D / Answer / Explanation` format that Quiz Mode needs.
* **ELI10:** both use an analogy, but the fine-tuned answer uses shorter sentences (12.5 vs 21
  words per sentence) and ends with a "grown-up version" line, as in training.
* **Teaching structure:** fine-tuned answers are shorter and structured (bold labels such as
  *Key idea to remember*, *Exam-ready definition*). The base model writes longer, more
  conversational paragraphs, which sometimes contain more detail.
* **Unseen topics** (batch normalization, dropout, word embeddings): the fine-tuned model applies
  the same teaching structure to topics that were not in the training data.
* **Not always better:** for *"Explain CNN to a beginner"* the base answer included an example and
  an analogy that the shorter fine-tuned answer did not. Fine-tuned MCQs can also contain mistakes
  (overlapping options, or a wrong premise such as an "RL loop" in DPO). That is why the app offers
  a curated question bank and shows the disclaimer.

## Presentation / demo guide

1. **Show the dataset** — open `data/instruction_dataset.jsonl` (or run `python scripts/prepare_dataset.py` to print the statistics and a formatted example such as "Explain RNN to a beginner").
2. **"This is an instruction dataset"** — each record is *Instruction → Response*, and the loss is computed only on the response.
3. **Show the base model** — *⚖️ Base vs Fine-Tuned*: ask *Explain RNN to a beginner.* and point at the BASE MODEL column.
4. **QLoRA training** — show `training/train_qlora.py` and the parameter counts in `results/training_summary.json` (do not re-run training during the demo — it overwrites the adapter): "The base model is quantised to 4-bit and frozen. LoRA adapters are trained instead of updating the entire model: 18.5 M of 1.56 B parameters (1.18%)."
5. **Load the fine-tuned model** — point at the FINE-TUNED MODEL column for the same question.
6. **BASE vs FINE-TUNED** — discuss the observable differences shown below the two answers.
7. **Explain Like I'm 10** — *💬 AI Tutor*: select the mode and ask *What is an LSTM?*
8. **Viva Mode** — let the examiner ask, give a partial answer, and show the feedback.
9. **MCQ Mode** — *📝 Quiz Mode*: answer 3–5 questions and show the score. *❓ MCQ Generator*: 5 medium questions on RNN.
10. **Training statistics** — show `results/training_loss.png`, `results/training_summary.json` and `results/eval_metrics.json` (trainable %, loss curve, perplexity before vs after, adapter size 46 MB vs ~3 GB model).

**Tip:** start the app ~1 minute before presenting so the model is already loaded.

**Summary sentence:** *"I took a pretrained language model and specialised it into an AI Teaching
Assistant using instruction tuning. I used PEFT with LoRA, and QLoRA to reduce GPU memory
requirements. The model was trained on educational instruction-response examples and can now
perform different teaching tasks such as simple explanations, analogies, viva preparation,
interview preparation, MCQs, and quizzes."*

## Concepts explained

**Fine-tuning** — a pretrained model is adapted to a specific task or domain by continuing training
on a smaller, focused dataset (transfer learning).

**Instruction tuning** — fine-tuning on *Instruction → Desired Response* pairs so the model learns to
follow requests in a particular way. It mainly teaches **behaviour and style**, not new facts.

**Supervised fine-tuning (SFT)** — the model learns from examples that contain the desired
responses. The loss is next-token cross-entropy, computed here **only on the response tokens**
(`completion_only_loss=True`).

**PEFT** — instead of updating every parameter, only a small number of additional parameters are
trained. Here that is 1.18%, which cuts gradient and optimiser memory and produces a 46 MB adapter
instead of a 3 GB model copy.

**LoRA** — for a frozen weight matrix `W (d×k)`, LoRA learns `ΔW = B·A` with `B (d×r)`, `A (r×k)` and
small rank `r` (16 here). The layer computes `h = Wx + (α/r)·BAx`. `B` starts at zero, so training
begins exactly from the original model. For one 1536×1536 projection, LoRA trains
16·(1536+1536) = 49,152 parameters instead of 2,359,296.

**QLoRA** — the frozen base model is stored in **4-bit NF4** (NormalFloat4, with double quantisation
of the scaling constants) and dequantised to bf16 on the fly for each matmul. Gradients flow through
it into LoRA adapters kept in bf16. A **paged** 8-bit optimiser avoids memory spikes. In this
project the model's weights take ~1 GB instead of ~3 GB, and training peaked at under 3 GB of GPU
memory.

## Hardware limits & troubleshooting

| Problem | What to do |
|---|---|
| `CUDA Available: False` | Install the CUDA build of PyTorch (step 2 of the installation) and update the NVIDIA driver. The app still runs on CPU, but slowly; QLoRA training needs a GPU. |
| **CUDA out of memory** during training | In `config.py`: `BATCH_SIZE = 1` (raise `GRADIENT_ACCUMULATION_STEPS`), `MAX_SEQ_LENGTH = 512`, keep `GRADIENT_CHECKPOINTING = True` and `USE_4BIT = True`, lower `LORA_R` to 8, or use `MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"`. The scripts print these suggestions automatically on an OOM error. |
| OOM during generation | Lower *Max new tokens* in the sidebar and close other GPU programs. |
| "No LoRA adapter found" | Run `python training/train_qlora.py`. The app still works with the Base Model. |
| Missing dataset | Run `generate_dataset.py`, then `prepare_dataset.py`. |
| Model download fails | Check the internet connection, or pre-download with `huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct`. |
| Changing the model | Edit `MODEL_NAME` (and `LORA_TARGET_MODULES` if the architecture differs), then retrain. An adapter only works with the base model it was trained on. |

## Limitations & honest notes

* **Validation examples share topics with training examples.** The split is by example, not by
  topic, so perplexity measures how well the model learned the target style and content, not
  knowledge of unseen topics. The 3 unseen-topic questions in `comparison.json` give a qualitative
  view of generalisation.
* The dataset is generated from hand-written structured content, so different categories for the
  same topic share facts (for example, the definition appears in several outputs). This is
  intentional, to keep the facts consistent, but it is less diverse than a crowd-sourced dataset.
* Instruction tuning on ~750 examples mainly changes **style and format**. The 1.5B model's factual
  knowledge is largely unchanged, so it can still make mistakes. Model-generated MCQs and feedback
  should be checked.
* Loss and perplexity show that the fine-tuned model assigns much higher probability to the
  reference teaching answers. They are not a direct measure of correctness or teaching quality.
  That is why side-by-side qualitative comparison is included and no automatic "better" score is
  claimed.
