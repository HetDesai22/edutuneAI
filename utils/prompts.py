"""
Prompt templates.

This module is the single source of truth for how instructions are phrased.
The dataset generator (scripts/generate_dataset.py) builds training
instructions with these same templates, and the Streamlit app builds its
prompts with them too. The fine-tuned model is therefore trained on exactly
the instruction styles it will see at inference time.
"""

from __future__ import annotations

import config

# ---------------------------------------------------------------------------
# Teaching modes (sidebar). Each mode turns a free-form student question into
# an instruction for the model.
# ---------------------------------------------------------------------------
TEACHING_MODES: dict[str, dict[str, str]] = {
    "Simple Explanation": {
        "template": "Explain in simple, beginner-friendly language: {question}",
        "category": "simple_explanation",
    },
    "Detailed Explanation": {
        "template": "Give a detailed, well-structured explanation: {question}",
        "category": "detailed_explanation",
    },
    "Explain Like I'm 10": {
        "template": "Explain like I'm 10 years old: {question}",
        "category": "simple_explanation",
    },
    "Example-Based Learning": {
        "template": "Teach this using a concrete worked example: {question}",
        "category": "example",
    },
    "Interview Preparation": {
        "template": "Answer this as a strong candidate would in a technical interview: {question}",
        "category": "interview",
    },
    "Viva Preparation": {
        "template": "Answer this viva question clearly and concisely, as a student would to an examiner: {question}",
        "category": "viva",
    },
    "MCQ Practice": {
        "template": "Create 3 multiple-choice questions to test my understanding of: {question}",
        "category": "mcq",
    },
    "Numerical Problem": {
        "template": "Give a numerical problem with a step-by-step solution on: {question}",
        "category": "numerical",
    },
    "Revision": {
        "template": "Give quick revision notes on: {question}",
        "category": "revision",
    },
}

DEFAULT_MODE = "Simple Explanation"

# ---------------------------------------------------------------------------
# Quick-action buttons in the chat. {topic} is the current topic/question.
# ---------------------------------------------------------------------------
ACTIONS: dict[str, str] = {
    "📚 Explain": "Explain {topic}.",
    "💡 Give Example": "Give a real-world example of {topic}.",
    "🧠 Give Analogy": "Explain {topic} using a simple real-world analogy.",
    "📝 Quiz Me": "Create 3 multiple-choice questions on {topic}.",
    "🎯 Viva Questions": "Give 5 viva questions on {topic} with short answers.",
    "💼 Interview Questions": "Give interview questions on {topic} with model answers.",
}

# "Simplify Answer" is special: the previous answer is passed as the `input`.
SIMPLIFY_INSTRUCTION = "Simplify this explanation so that a beginner can understand it."
SIMPLIFY_ACTION = "🔄 Simplify Answer"

# Topics offered in the sidebar (display name -> short name used in prompts).
TOPICS: dict[str, list[str]] = {
    "Artificial Intelligence": [
        "AI fundamentals", "Intelligent agents", "Search algorithms",
        "Knowledge representation",
    ],
    "Machine Learning": [
        "Supervised learning", "Unsupervised learning", "Regression",
        "Classification", "Clustering", "Decision trees", "Random forests",
        "SVM", "KNN", "Gradient descent", "Overfitting", "Underfitting",
    ],
    "Deep Learning": [
        "Neural networks (ANN)", "Backpropagation", "Activation functions",
        "Vanishing and exploding gradients", "CNN", "RNN", "LSTM", "GRU",
    ],
    "Generative AI": [
        "GAN", "VAE", "Transformers", "Attention", "LLM",
        "Prompt engineering", "Fine-tuning", "LoRA", "QLoRA", "PEFT",
        "Instruction tuning", "RLHF", "DPO",
    ],
}

ALL_TOPICS: list[str] = [t for group in TOPICS.values() for t in group]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_prompt(question: str, mode: str = DEFAULT_MODE) -> str:
    """Turn a student's question + selected teaching mode into an instruction."""
    question = question.strip()
    if not question:
        raise ValueError("The question is empty.")
    template = TEACHING_MODES.get(mode, TEACHING_MODES[DEFAULT_MODE])["template"]
    return template.format(question=question)


def build_action_prompt(action: str, topic: str) -> str:
    return ACTIONS[action].format(topic=topic.strip())


def build_messages(instruction: str, input_text: str = "",
                   system_prompt: str | None = None) -> list[dict[str, str]]:
    """
    Chat messages for one instruction. Used identically by training
    (training/format_dataset.py) and inference (utils/model_loader.py).
    """
    user = instruction.strip()
    if input_text and input_text.strip():
        user = f"{user}\n\n{input_text.strip()}"
    return [
        {"role": "system", "content": system_prompt or config.SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --- Special modes: viva / interview / MCQ ---------------------------------
def build_viva_question_prompt(topic: str) -> str:
    return f"Ask me one viva question about {topic}. Only ask the question."


def build_viva_feedback_prompt(question: str) -> str:
    """The student's answer is passed separately as the `input` field."""
    return (
        f"You are a viva examiner. Evaluate the student's answer to the viva "
        f"question: \"{question}\" Give educational feedback."
    )


def build_interview_question_prompt(topic: str) -> str:
    return f"Ask me one technical interview question about {topic}. Only ask the question."


def build_interview_feedback_prompt(question: str) -> str:
    return (
        f"You are a technical interviewer. Evaluate the candidate's answer to the "
        f"interview question: \"{question}\" Say what they got right, what they "
        f"missed, and give a suggested improved answer."
    )


def build_mcq_prompt(topic: str, n: int = 5, difficulty: str | None = None) -> str:
    plural = "question" if n == 1 else "questions"
    if difficulty:
        return f"Create {n} {difficulty.lower()} multiple-choice {plural} on {topic}."
    return f"Create {n} multiple-choice {plural} on {topic}."


def format_student_answer(answer: str) -> str:
    return f"Student's answer: {answer.strip()}"
