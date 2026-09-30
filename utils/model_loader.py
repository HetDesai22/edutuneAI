"""
Model loading and text generation.

Key idea for the demo: the BASE model and the FINE-TUNED model share the same
frozen weights. The fine-tuned model is simply base model + LoRA adapter
(models/edu_tune_lora/). So we load the base weights once, attach the adapter,
and temporarily disable the adapter to get "base model" answers. This halves
GPU memory compared with loading two separate models, and guarantees the ONLY
difference between the two answers is the trained adapter.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TextIteratorStreamer)

import config
from training.format_dataset import format_instruction
from utils.device import OOM_HELP, can_use_4bit, get_compute_dtype, is_oom_error


class ModelLoadError(RuntimeError):
    """Raised with a user-friendly message when a model cannot be loaded."""


class AdapterNotFoundError(ModelLoadError):
    pass


class GenerationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def adapter_exists(path: Path = config.OUTPUT_DIR) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def get_quantization_config() -> BitsAndBytesConfig | None:
    """
    QLoRA 4-bit config: weights stored as 4-bit NormalFloat (NF4), with double
    quantisation of the scaling constants; matrix multiplications are done in
    bf16/fp16 after on-the-fly dequantisation.
    """
    if not (config.USE_4BIT and can_use_4bit()):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=get_compute_dtype(),
    )


def load_tokenizer(model_name: str = config.MODEL_NAME):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"Could not load the tokenizer for '{model_name}'. Check MODEL_NAME in config.py "
            f"and your internet connection (the model is downloaded on first use).\nDetails: {exc}"
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str = config.MODEL_NAME):
    """Load the pre-trained model (no adapter). Returns (model, tokenizer)."""
    tokenizer = load_tokenizer(model_name)
    quant = get_quantization_config()
    kwargs: dict = {"dtype": get_compute_dtype()}
    if quant is not None:
        kwargs["quantization_config"] = quant
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if is_oom_error(exc):
            raise ModelLoadError(OOM_HELP) from exc
        raise ModelLoadError(
            f"Could not load model '{model_name}'. Check MODEL_NAME in config.py, your "
            f"internet connection and free disk space.\nDetails: {exc}"
        ) from exc
    model.eval()
    return model, tokenizer


def attach_adapter(model, adapter_path: Path = config.OUTPUT_DIR):
    """Wrap a base model with the trained LoRA adapter (not merged)."""
    from peft import PeftModel

    if not adapter_exists(adapter_path):
        raise AdapterNotFoundError(
            f"No LoRA adapter found in '{adapter_path}'. Train one first with "
            f"`python training/train_qlora.py`."
        )
    try:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"The adapter in '{adapter_path}' could not be loaded. It may have been trained "
            f"for a different MODEL_NAME.\nDetails: {exc}"
        ) from exc
    model.eval()
    return model


def load_finetuned_model(model_name: str = config.MODEL_NAME, adapter_path: Path = config.OUTPUT_DIR):
    """Load base model + LoRA adapter. Returns (peft_model, tokenizer)."""
    if not adapter_exists(adapter_path):
        # Fail fast before spending time loading the base model.
        attach_adapter(None, adapter_path)
    model, tokenizer = load_base_model(model_name)
    return attach_adapter(model, adapter_path), tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _generation_kwargs(tokenizer, max_new_tokens: int, temperature: float) -> dict:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": config.REPETITION_PENALTY,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=config.TOP_P)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return kwargs


class TeachingModel:
    """
    One loaded model that can answer as the BASE model or as the FINE-TUNED
    model (adapter enabled). Cached once by the Streamlit app.
    """

    def __init__(self, load_adapter: bool = True):
        self.model, self.tokenizer = load_base_model()
        self.adapter_loaded = False
        self.adapter_error: str | None = None
        if load_adapter:
            try:
                self.model = attach_adapter(self.model)
                self.adapter_loaded = True
            except ModelLoadError as exc:
                self.adapter_error = str(exc)
        self._lock = threading.Lock()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _adapter_context(self, use_adapter: bool):
        if use_adapter and not self.adapter_loaded:
            raise AdapterNotFoundError(self.adapter_error or "The LoRA adapter is not loaded.")
        if self.adapter_loaded and not use_adapter:
            return self.model.disable_adapter()   # temporarily = pure base model
        return nullcontext()

    def _encode(self, instruction: str, input_text: str):
        prompt = format_instruction(
            {"instruction": instruction, "input": input_text, "output": ""},
            self.tokenizer, include_response=False,
        )
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def generate(self, instruction: str, input_text: str = "", use_adapter: bool = True,
                 max_new_tokens: int = config.MAX_NEW_TOKENS,
                 temperature: float = config.TEMPERATURE) -> str:
        """Return the full response as a string."""
        return "".join(self.stream(instruction, input_text, use_adapter, max_new_tokens, temperature))

    def stream(self, instruction: str, input_text: str = "", use_adapter: bool = True,
               max_new_tokens: int = config.MAX_NEW_TOKENS,
               temperature: float = config.TEMPERATURE) -> Iterator[str]:
        """Yield the response piece by piece (for a live typing effect in the UI)."""
        if not instruction or not instruction.strip():
            raise GenerationError("The question is empty.")
        inputs = self._encode(instruction, input_text)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, streamer=streamer,
                      **_generation_kwargs(self.tokenizer, max_new_tokens, temperature))
        errors: list[BaseException] = []

        def _run() -> None:
            try:
                with self._adapter_context(use_adapter), torch.inference_mode():
                    self.model.generate(**kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised in caller thread
                errors.append(exc)
                streamer.end()

        with self._lock:  # the adapter on/off switch is global model state
            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            for piece in streamer:
                yield piece
            thread.join()

        if errors:
            exc = errors[0]
            if is_oom_error(exc):
                torch.cuda.empty_cache()
                raise GenerationError(OOM_HELP) from exc
            if isinstance(exc, AdapterNotFoundError):
                raise exc
            raise GenerationError(f"Generation failed: {exc}") from exc
