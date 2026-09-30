"""
Instruction formatting - the ONLY place where a dataset record is turned into
model input. Training, evaluation, the test script and the app all use it.

A record looks like:
    {"instruction": "Explain RNN to a beginner.", "input": "", "output": "An RNN is ..."}

Two equivalent views are provided:

1. Chat format (used for training/inference). The record becomes
       system : config.SYSTEM_PROMPT
       user   : instruction (+ input)
       assistant : output
   and the tokenizer's own chat template turns it into text. Using the model's
   native template means the fine-tuned model sees prompts in the format it
   was pre-trained with.

2. Plain "### Instruction / ### Response" (Alpaca-style) text, used when a
   tokenizer has no chat template and for human-readable display.
"""

from __future__ import annotations

from typing import Any

from utils.prompts import build_messages

ALPACA_TEMPLATE = "### Instruction:\n{instruction}\n\n{input_block}### Response:\n"


def build_conversation(example: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Split a record into (prompt messages, completion messages)."""
    prompt = build_messages(example["instruction"], example.get("input", ""))
    completion = [{"role": "assistant", "content": example["output"]}]
    return prompt, completion


def to_prompt_completion(example: dict[str, Any]) -> dict[str, list[dict]]:
    """
    Conversational prompt-completion format understood by TRL's SFTTrainer.
    With this format TRL computes the loss ONLY on the completion (the
    assistant's answer), not on the system prompt or the question.
    """
    prompt, completion = build_conversation(example)
    return {"prompt": prompt, "completion": completion}


def format_alpaca(example: dict[str, Any], include_response: bool = True) -> str:
    inp = (example.get("input") or "").strip()
    input_block = f"### Input:\n{inp}\n\n" if inp else ""
    text = ALPACA_TEMPLATE.format(instruction=example["instruction"].strip(), input_block=input_block)
    if include_response:
        text += example["output"].strip()
    return text


def format_instruction(example: dict[str, Any], tokenizer=None, include_response: bool = True) -> str:
    """
    Format one record as model-ready text.

    - With a tokenizer that has a chat template: the model's native chat format.
    - Otherwise: the plain "### Instruction / ### Response" format.

    include_response=False returns only the prompt (ending where the model
    should start generating), which is what inference and loss-masking need.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        prompt, completion = build_conversation(example)
        messages = prompt + completion if include_response else prompt
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not include_response,
        )
    return format_alpaca(example, include_response)
