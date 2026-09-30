"""
End-to-end checks with clear PASS/FAIL output.

    python scripts/test_model.py              # all tests (loads the model)
    python scripts/test_model.py --quick      # skip model loading / inference
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
_STATE: dict = {}


def check(name: str, fn: Callable[[], str | None]) -> bool:
    start = time.time()
    try:
        detail = fn() or ""
        ok = True
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        if "--verbose" in sys.argv:
            traceback.print_exc()
        ok = False
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {name:<32} {detail} ({time.time() - start:.1f}s)")
    RESULTS.append((name, ok, detail))
    return ok


# ---------------------------------------------------------------------------
def test_gpu() -> str:
    from utils.device import get_device_info
    info = get_device_info()
    if info.cuda_available:
        return f"CUDA available - {info.gpu_name} ({info.gpu_memory_gb} GB)"
    return "CUDA NOT available - running on CPU (slow, no 4-bit QLoRA)"


def test_dataset_loading() -> str:
    counts = {}
    for path in (config.DATASET_FILE, config.TRAIN_FILE, config.VALIDATION_FILE):
        assert path.exists(), f"{path.name} missing - run generate_dataset.py / prepare_dataset.py"
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        assert rows, f"{path.name} is empty"
        for r in rows:
            assert all(isinstance(r.get(f), str) for f in config.REQUIRED_FIELDS), f"bad record in {path.name}"
        counts[path.stem] = len(rows)
    _STATE["example"] = rows[0]
    return ", ".join(f"{k}={v}" for k, v in counts.items())


def test_formatting() -> str:
    from training.format_dataset import format_instruction, to_prompt_completion
    ex = {"instruction": "Explain RNN to a beginner.", "input": "", "output": "An RNN is ..."}
    text = format_instruction(ex)
    assert text.startswith("### Instruction:") and "### Response:\nAn RNN is ..." in text
    prompt_only = format_instruction(ex, include_response=False)
    assert prompt_only.endswith("### Response:\n")
    pc = to_prompt_completion(ex)
    assert [m["role"] for m in pc["prompt"]] == ["system", "user"]
    assert pc["completion"][0]["content"] == "An RNN is ..."
    return "Alpaca + chat prompt/completion formats OK"


def test_prompts() -> str:
    from utils.prompts import TEACHING_MODES, build_prompt
    for mode in TEACHING_MODES:
        assert "RNN" in build_prompt("What is RNN?", mode)
    try:
        build_prompt("   ", "Simple Explanation")
        raise AssertionError("empty question was accepted")
    except ValueError:
        pass
    return f"{len(TEACHING_MODES)} teaching modes, empty question rejected"


def test_mcq_parser() -> str:
    from utils.mcq import parse_mcqs
    sample = ("**Question 1:** Which network is used for sequences?\nA. CNN\nB. RNN\nC. GAN\nD. SVM\n"
              "**Answer:** B\n**Explanation:** RNNs keep a hidden state.")
    parsed = parse_mcqs(sample)
    assert len(parsed) == 1 and parsed[0]["answer"] == "B" and len(parsed[0]["options"]) == 4
    return "MCQ format parsed"


def test_base_model() -> str:
    from utils.model_loader import load_base_model
    model, tokenizer = load_base_model()
    _STATE["model"], _STATE["tokenizer"] = model, tokenizer
    quant = "4-bit" if getattr(model, "is_loaded_in_4bit", False) else str(model.dtype)
    return f"{config.MODEL_NAME} loaded ({quant}) on {next(model.parameters()).device}"


def test_adapter() -> str:
    from utils.model_loader import adapter_exists, attach_adapter
    assert "model" in _STATE, "base model not loaded"
    assert adapter_exists(), f"no adapter in {config.OUTPUT_DIR} - run training/train_qlora.py"
    _STATE["model"] = attach_adapter(_STATE["model"])
    trainable = sum(p.numel() for n, p in _STATE["model"].named_parameters() if "lora_" in n)
    return f"LoRA adapter attached ({trainable:,} adapter parameters)"


def test_inference() -> str:
    import torch
    from training.format_dataset import format_instruction
    assert "model" in _STATE, "model not loaded"
    model, tok = _STATE["model"], _STATE["tokenizer"]
    prompt = format_instruction({"instruction": "Explain RNN to a beginner.", "input": "", "output": ""},
                                tok, include_response=False)
    inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    outputs = {}
    for label, ctx in (("base", getattr(model, "disable_adapter", None)), ("fine-tuned", None)):
        with (ctx() if ctx else torch.no_grad()), torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outputs[label] = tok.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        assert outputs[label].strip(), f"{label} model produced empty output"
    return "base: " + repr(outputs["base"][:50]) + " | fine-tuned: " + repr(outputs["fine-tuned"][:50])


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip model loading and inference")
    parser.add_argument("--verbose", action="store_true", help="print tracebacks")
    args = parser.parse_args()

    print("=" * 70 + "\nEduTune AI - system tests\n" + "=" * 70)
    check("GPU detection", test_gpu)
    check("Dataset loading", test_dataset_loading)
    check("Dataset formatting", test_formatting)
    check("Prompt building", test_prompts)
    check("MCQ parser", test_mcq_parser)
    if not args.quick:
        if check("Base model loading", test_base_model):
            check("Fine-tuned model loading", test_adapter)
            check("Inference", test_inference)

    passed = sum(ok for _, ok, _ in RESULTS)
    print("=" * 70 + f"\n{passed}/{len(RESULTS)} tests passed")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
