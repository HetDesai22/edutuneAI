"""
EduTune AI - Streamlit interface.

    streamlit run app.py

Pages: AI Tutor chat, Base vs Fine-Tuned comparison, Quiz, Viva, Interview
and MCQ generator. All tutoring pages use the fine-tuned model.

The model is loaded ONCE (st.cache_resource). "Base Model" answers are produced
by the same model with the LoRA adapter temporarily disabled, so the only
difference between the two is the fine-tuned adapter.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="EduTune AI", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

import config  # noqa: E402
from evaluation.comparison import compare_responses  # noqa: E402
from utils.mcq import parse_mcqs, random_bank_question  # noqa: E402
from utils.prompts import (  # noqa: E402
    ACTIONS, ALL_TOPICS, SIMPLIFY_ACTION, SIMPLIFY_INSTRUCTION, TEACHING_MODES,
    build_action_prompt, build_interview_feedback_prompt, build_interview_question_prompt,
    build_mcq_prompt, build_prompt, build_viva_feedback_prompt, build_viva_question_prompt,
    format_student_answer,
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg,#4338ca 0%,#7c3aed 55%,#db2777 100%);
       border-radius: 18px; padding: 1.4rem 1.8rem; color: #fff; margin-bottom: 1.1rem;
       box-shadow: 0 10px 30px rgba(79,70,229,.25);}
.hero h1 {color:#fff; font-size: 2.1rem; margin: 0; padding: 0;}
.hero p {color: rgba(255,255,255,.88); margin: .25rem 0 0 0; font-size: 1.05rem;}
.hero .tag {display:inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
            padding: .15rem .7rem; font-size: .8rem; margin: .6rem .35rem 0 0;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1rem 1.2rem;
       background: rgba(127,127,127,.04); margin-bottom: .8rem;}
.card h4 {margin-top: 0;}
.model-head {border-radius: 12px; padding: .55rem 1rem; font-weight: 700; letter-spacing: .04em;
             margin-bottom: .6rem; text-align: center;}
.model-base {background: #e2e8f0; color: #1e293b;}
.model-ft {background: linear-gradient(90deg,#4f46e5,#7c3aed); color: #fff;}
.examiner {border-left: 4px solid #7c3aed; background: rgba(124,58,237,.07); border-radius: 8px;
           padding: .8rem 1rem; margin: .5rem 0 .9rem 0; font-size: 1.05rem;}
.disclaimer {font-size: .78rem; opacity: .8; border-top: 1px solid rgba(128,128,128,.3);
             padding-top: .6rem; margin-top: .6rem;}
</style>
""", unsafe_allow_html=True)

PAGES = [
    "💬 AI Tutor",
    "⚖️ Base vs Fine-Tuned",
    "📝 Quiz Mode",
    "🎤 Viva Mode",
    "💼 Interview Mode",
    "❓ MCQ Generator",
]
ANY_TOPIC = "(Any — use my question)"
# All tutoring pages use the fine-tuned model (base + LoRA adapter); the
# "Base vs Fine-Tuned" page still compares both side by side.
use_adapter = True
model_choice = "Fine-Tuned Model"


# ---------------------------------------------------------------------------
# Cached resources / data
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_teaching_model():
    from utils.model_loader import TeachingModel
    return TeachingModel(load_adapter=True)


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except json.JSONDecodeError:
        return None


def ss(key: str, default):
    """Session-state getter with a default."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ---------------------------------------------------------------------------
# Model loading (once, with a friendly error instead of a traceback)
# ---------------------------------------------------------------------------
def load_model():
    if "model_error" in st.session_state and st.session_state.model_error:
        return None
    try:
        with st.spinner(f"Loading {config.MODEL_NAME} and the LoRA adapter (first run only)..."):
            return get_teaching_model()
    except Exception as exc:  # noqa: BLE001
        st.session_state.model_error = str(exc)
        return None


tm = load_model()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 EduTune AI")
    page = st.radio("NAVIGATE", PAGES, label_visibility="collapsed")

    st.markdown("#### TEACHING MODE")
    mode = st.radio("TEACHING MODE", list(TEACHING_MODES), label_visibility="collapsed")

    st.markdown("#### TOPIC")
    topic_choice = st.selectbox("TOPIC", [ANY_TOPIC] + ALL_TOPICS, label_visibility="collapsed")
    sidebar_topic = None if topic_choice == ANY_TOPIC else topic_choice

    with st.expander("⚙️ Generation settings"):
        max_new_tokens = st.slider("Max new tokens", 64, 1024, config.MAX_NEW_TOKENS, 32)
        temperature = st.slider("Temperature", 0.0, 1.5, config.TEMPERATURE, 0.05,
                                help="0 = deterministic (greedy). Higher = more varied.")

    if tm is None and st.button("🔄 Retry loading model", width="stretch"):
        st.session_state.model_error = None
        st.rerun()
    st.markdown(f'<div class="disclaimer">⚠️ {config.DISCLAIMER}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def hero(title: str, subtitle: str, tags: list[str] | None = None) -> None:
    tag_html = "".join(f'<span class="tag">{t}</span>' for t in (tags or []))
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{tag_html}</div>',
                unsafe_allow_html=True)


def model_ready(adapter: bool | None = None) -> bool:
    """Show a friendly message and return False if the requested model can't be used."""
    if tm is None:
        st.error("❌ The model could not be loaded (use **Retry loading** in the sidebar).\n\n"
                 + (st.session_state.get("model_error") or ""))
        return False
    adapter = use_adapter if adapter is None else adapter
    if adapter and not tm.adapter_loaded:
        st.warning("⚠️ The fine-tuned LoRA adapter was not found, so only the **Base Model** is "
                   "available. Train it with `python training/train_qlora.py`, then restart the app."
                   f"\n\n{tm.adapter_error or ''}")
        return False
    return True


def stream_answer(instruction: str, input_text: str = "", adapter: bool | None = None,
                  temp: float | None = None, tokens: int | None = None) -> str | None:
    """Stream a response into the page. Returns the text, or None on error."""
    from utils.model_loader import GenerationError, ModelLoadError
    adapter = use_adapter if adapter is None else adapter
    try:
        return st.write_stream(tm.stream(
            instruction, input_text, use_adapter=adapter,
            max_new_tokens=tokens or max_new_tokens,
            temperature=temperature if temp is None else temp,
        ))
    except (GenerationError, ModelLoadError) as exc:
        st.error(f"❌ {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Unexpected generation error: {exc}")
    return None


def generate_text(instruction: str, input_text: str = "", adapter: bool | None = None,
                  temp: float | None = None, tokens: int | None = None) -> str | None:
    from utils.model_loader import GenerationError, ModelLoadError
    adapter = use_adapter if adapter is None else adapter
    try:
        return tm.generate(instruction, input_text, use_adapter=adapter,
                           max_new_tokens=tokens or max_new_tokens,
                           temperature=temperature if temp is None else temp)
    except (GenerationError, ModelLoadError) as exc:
        st.error(f"❌ {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Unexpected generation error: {exc}")
    return None


def clean_question(text: str) -> str:
    """Keep only the question the examiner/interviewer asked."""
    text = re.sub(r"\*\*|^(question|viva question|interview question)\s*:\s*", "", text.strip(),
                  flags=re.IGNORECASE)
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), text)
    return first.strip().strip('"')


def current_topic() -> str:
    return sidebar_topic or random.choice(ALL_TOPICS)


# ---------------------------------------------------------------------------
# Page: AI Tutor chat
# ---------------------------------------------------------------------------
def page_chat() -> None:
    hero("🎓 EduTune AI", config.APP_SUBTITLE,
         [f"Model: {model_choice}", f"Mode: {mode}", "QLoRA · LoRA · PEFT"])
    messages = ss("messages", [])
    pending: tuple[str, str, str] | None = None     # (display text, instruction, input)

    # Read the chat box first (it is always pinned to the bottom) so the quick
    # actions below already know about a question typed in this run.
    question = st.chat_input("Ask your question...")
    if question is not None:
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            pending = (question, build_prompt(question, mode), "")
            st.session_state.last_question = question.strip()

    if not messages:
        st.markdown('<div class="card"><h4>👋 Welcome!</h4>Ask any question about AI, Machine '
                    'Learning, Deep Learning or Generative AI. Choose a <b>teaching mode</b> in '
                    'the sidebar to change how the tutor explains. Try one of these:</div>',
                    unsafe_allow_html=True)
        examples = ["Explain LSTM in simple language.", "What is overfitting?",
                    "How does LoRA work?", "Compare CNN and RNN."]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex, width="stretch"):
                pending = (ex, build_prompt(ex, mode), "")

    for msg in messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant":
                st.caption(f"AI Tutor · {msg['model']} · {msg['mode']}")
            st.markdown(msg["content"])

    # Quick-action buttons act on the sidebar topic, or on the last question.
    last_q = ss("last_question", None)
    target = sidebar_topic or last_q
    st.markdown("**Quick actions** " + (f"— *{target}*" if target else "— ask a question or pick a topic first"))
    labels = list(ACTIONS) + [SIMPLIFY_ACTION]
    cols = st.columns(len(labels))
    for col, label in zip(cols, labels):
        if col.button(label, width="stretch", disabled=target is None and label != SIMPLIFY_ACTION):
            if label == SIMPLIFY_ACTION:
                last_answer = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), None)
                if last_answer:
                    pending = (f"{label}", SIMPLIFY_INSTRUCTION, last_answer)
                else:
                    st.warning("There is no answer to simplify yet.")
            else:
                pending = (f"{label} — {target}", build_action_prompt(label, target), "")

    _, right = st.columns([6, 1])
    if right.button("🗑️ Clear chat", width="stretch"):
        st.session_state.messages, st.session_state.last_question = [], None
        st.rerun()

    if pending and model_ready():
        display, instruction, input_text = pending
        messages.append({"role": "user", "content": display})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(display)
        with st.chat_message("assistant", avatar="🎓"):
            st.caption(f"AI Tutor · {model_choice} · {mode}")
            answer = stream_answer(instruction, input_text)
        if answer:
            messages.append({"role": "assistant", "content": answer, "model": model_choice, "mode": mode})
        with st.expander("🔍 Instruction sent to the model"):
            st.code(instruction + (f"\n\n{input_text}" if input_text else ""), language=None)


# ---------------------------------------------------------------------------
# Page: Base vs Fine-Tuned
# ---------------------------------------------------------------------------
def render_observations(instruction: str, base: str, ft: str) -> None:
    result = compare_responses(instruction, base, ft)
    st.markdown("#### 🔎 Observable differences")
    st.caption("Measured properties only — no automatic 'winner' is declared. "
               "Judge correctness and teaching quality yourself.")
    for note in result["observations"]:
        st.markdown(f"- {note}")
    rows = {k: [result["base_metrics"][k], result["finetuned_metrics"][k]] for k in result["base_metrics"]}
    df = pd.DataFrame(rows, index=["Base Model", "Fine-Tuned Model"]).T.astype(str)
    st.dataframe(df, width="stretch")


def page_compare() -> None:
    hero("⚖️ Base Model vs Fine-Tuned Model", "Same question, same prompt, same weights — "
         "the only difference is the trained LoRA adapter.", ["Side-by-side", "Greedy decoding"])
    c1, c2 = st.columns([3, 2])
    question = c1.text_input("Question", value=ss("cmp_q", "Explain LSTM to a beginner."))
    mode_opts = ["(No mode — send the question as-is)"] + list(TEACHING_MODES)
    cmp_mode = c2.selectbox("Teaching mode", mode_opts, index=0)
    greedy = st.checkbox("Deterministic decoding (recommended for a fair comparison)", value=True)

    if st.button("⚖️ Compare", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif model_ready(adapter=True):
            instruction = question.strip() if cmp_mode == mode_opts[0] else build_prompt(question, cmp_mode)
            st.session_state.cmp_q = question
            temp = 0.0 if greedy else temperature
            left, right = st.columns(2)
            with left:
                st.markdown('<div class="model-head model-base">BASE MODEL</div>', unsafe_allow_html=True)
                base = stream_answer(instruction, adapter=False, temp=temp)
            with right:
                st.markdown('<div class="model-head model-ft">FINE-TUNED MODEL</div>', unsafe_allow_html=True)
                ft = stream_answer(instruction, adapter=True, temp=temp)
            if base and ft:
                st.session_state.cmp_result = (instruction, base, ft)
                render_observations(instruction, base, ft)
    elif "cmp_result" in st.session_state:
        instruction, base, ft = st.session_state.cmp_result
        st.markdown(f"**Question:** {instruction}")
        left, right = st.columns(2)
        left.markdown('<div class="model-head model-base">BASE MODEL</div>', unsafe_allow_html=True)
        left.markdown(base)
        right.markdown('<div class="model-head model-ft">FINE-TUNED MODEL</div>', unsafe_allow_html=True)
        right.markdown(ft)
        render_observations(instruction, base, ft)

    st.divider()
    st.markdown("### 📁 Saved evaluation results")
    comp = load_json(config.COMPARISON_FILE)
    if not comp:
        st.info("No saved comparison yet. Run `python evaluation/evaluate.py`.")
        return
    items = comp["comparisons"]
    idx = st.selectbox("Evaluation question", range(len(items)),
                       format_func=lambda i: f"{items[i]['instruction']}  [{items[i]['mode']}]")
    item = items[idx]
    left, right = st.columns(2)
    left.markdown('<div class="model-head model-base">BASE MODEL</div>', unsafe_allow_html=True)
    left.markdown(item["base_model_response"])
    right.markdown('<div class="model-head model-ft">FINE-TUNED MODEL</div>', unsafe_allow_html=True)
    right.markdown(item["finetuned_model_response"])
    st.markdown("#### 🔎 Observable differences")
    for note in item["observations"]:
        st.markdown(f"- {note}")


# ---------------------------------------------------------------------------
# Page: Quiz
# ---------------------------------------------------------------------------
def page_quiz() -> None:
    hero("📝 Quiz Mode", "Answer multiple-choice questions and track your score.",
         [f"Topic: {sidebar_topic or 'Random'}", f"Model: {model_choice}"])
    quiz = ss("quiz", {"score": 0, "total": 0, "current": None, "answered": False, "asked": []})

    c1, c2, c3 = st.columns([2, 2, 1])
    source = c1.radio("Question source", ["Generated by the model", "Curated question bank"],
                      horizontal=True)
    c2.metric("Score", f"{quiz['score']} / {quiz['total']}")
    if c3.button("↺ Reset", width="stretch"):
        st.session_state.quiz = {"score": 0, "total": 0, "current": None, "answered": False, "asked": []}
        st.rerun()

    if st.button("🎲 New Question", type="primary"):
        topic = current_topic()
        q, note = None, ""
        if source == "Generated by the model" and model_ready():
            with st.spinner(f"The {model_choice.lower()} is writing a question on {topic}..."):
                raw = generate_text(build_mcq_prompt(topic, 1), temp=0.7, tokens=300)
            parsed = parse_mcqs(raw or "")
            if parsed:
                q, note = parsed[0], f"Generated by the {model_choice} on *{topic}*."
            else:
                note = ("The model's output could not be parsed as an A–D question, so this one "
                        "comes from the curated question bank.")
        if q is None:
            q = random_bank_question(topic, set(quiz["asked"]))
            note = note or f"From the curated question bank ({topic})."
        if q:
            quiz.update(current={**q, "note": note, "topic": topic}, answered=False)
            quiz["asked"].append(q["q"])

    q = quiz["current"]
    if not q:
        st.info("Click **New Question** to start. Choose a topic in the sidebar, or leave it on "
                "*Any* for random topics.")
        return
    st.markdown(f'<div class="card"><h4>Question</h4>{q["q"]}</div>', unsafe_allow_html=True)
    st.caption(q["note"])
    options = [f"{letter}. {opt}" for letter, opt in zip("ABCD", q["options"])]
    choice = st.radio("Your answer", options, index=None, key=f"quiz_choice_{len(quiz['asked'])}",
                      disabled=quiz["answered"])
    if st.button("✅ Submit", disabled=quiz["answered"]):
        if choice is None:
            st.warning("Select an option first.")
        else:
            quiz["answered"] = True
            quiz["total"] += 1
            quiz["last_correct"] = choice[0] == q["answer"]
            quiz["score"] += int(quiz["last_correct"])
            st.rerun()
    if quiz["answered"]:
        correct_text = options["ABCD".index(q["answer"])]
        if quiz.get("last_correct"):
            st.success(f"🎉 **Correct!**\n\n{q['explanation']}")
        else:
            st.error(f"❌ **Not quite.** The correct answer is **{correct_text}**.\n\n{q['explanation']}")
        st.caption("Model-generated questions may contain mistakes — verify with your notes.")


# ---------------------------------------------------------------------------
# Pages: Viva & Interview (same flow, different prompts)
# ---------------------------------------------------------------------------
def examiner_page(kind: str) -> None:
    viva = kind == "viva"
    role, icon = ("AI Examiner", "🎤") if viva else ("Interviewer", "💼")
    hero(f"{icon} {'Viva' if viva else 'Interview'} Mode",
         "One question at a time. Answer in your own words and get educational feedback."
         if viva else "Practise technical interview questions. Get feedback on what you got right, "
         "what you missed and an improved answer.",
         [f"Topic: {sidebar_topic or 'Random'}", f"Model: {model_choice}"])
    state = ss(kind, {"question": None, "topic": None, "history": []})

    if st.button(f"{icon} {'Ask me a question' if not state['question'] else 'Next question'}",
                 type="primary"):
        if model_ready():
            topic = current_topic()
            prompt = build_viva_question_prompt(topic) if viva else build_interview_question_prompt(topic)
            with st.spinner(f"{role} is thinking of a question on {topic}..."):
                raw = generate_text(prompt, temp=0.8, tokens=80)
            if raw:
                state.update(question=clean_question(raw), topic=topic)
                st.rerun()

    if not state["question"]:
        st.info(f"Click the button above. The {role.lower()} will ask a question about the sidebar "
                "topic (or a random topic).")
    else:
        st.markdown(f'<div class="examiner"><b>{role}</b> · <i>{state["topic"]}</i><br>{state["question"]}</div>',
                    unsafe_allow_html=True)
        answer = st.text_area("Your answer", key=f"{kind}_answer_{len(state['history'])}", height=130,
                              placeholder="Type your answer as you would say it...")
        if st.button("📨 Submit answer"):
            if not answer.strip():
                st.warning("Please type an answer first.")
            elif model_ready():
                instruction = (build_viva_feedback_prompt(state["question"]) if viva
                               else build_interview_feedback_prompt(state["question"]))
                st.markdown(f"**{role}'s feedback:**")
                feedback = stream_answer(instruction, format_student_answer(answer), temp=0.3)
                if feedback:
                    state["history"].append({"topic": state["topic"], "question": state["question"],
                                             "answer": answer, "feedback": feedback})
                    state["question"] = None
                    st.success("Click the button at the top for the next question.")

    if state["history"]:
        st.divider()
        st.markdown(f"### 🗂️ Session history ({len(state['history'])} answered)")
        for i, h in enumerate(reversed(state["history"]), 1):
            with st.expander(f"Q{len(state['history']) - i + 1}. {h['question']}"):
                st.markdown(f"**Your answer:** {h['answer']}")
                st.markdown(h["feedback"])


# ---------------------------------------------------------------------------
# Page: MCQ generator
# ---------------------------------------------------------------------------
def page_mcq() -> None:
    hero("❓ MCQ Generator", "Generate practice multiple-choice questions with answers and explanations.",
         [f"Model: {model_choice}"])
    c1, c2, c3 = st.columns([2, 1, 2])
    topics = ALL_TOPICS
    topic = c1.selectbox("Topic", topics, index=topics.index(sidebar_topic) if sidebar_topic else topics.index("RNN"))
    n = c2.number_input("Number of questions", 1, 10, 5)
    difficulty = c3.radio("Difficulty", ["Easy", "Medium", "Hard"], index=1, horizontal=True)

    if st.button("✨ Generate MCQs", type="primary") and model_ready():
        with st.spinner(f"Writing {n} {difficulty.lower()} questions on {topic}..."):
            raw = generate_text(build_mcq_prompt(topic, int(n), difficulty),
                                temp=0.7, tokens=min(170 * int(n) + 100, 1800))
        if raw:
            st.session_state.mcq_result = (topic, int(n), raw)

    if "mcq_result" not in st.session_state:
        return
    topic, n, raw = st.session_state.mcq_result
    mcqs = parse_mcqs(raw)
    if not mcqs:
        st.warning("The output could not be parsed into the A–D format. Raw output:")
        st.markdown(raw)
        return
    if len(mcqs) < n:
        st.info(f"Parsed {len(mcqs)} of the {n} requested questions.")
    for i, m in enumerate(mcqs, 1):
        opts = "<br>".join(f"<b>{l}.</b> {o}" for l, o in zip("ABCD", m["options"]))
        st.markdown(f'<div class="card"><h4>Q{i}. {m["q"]}</h4>{opts}</div>', unsafe_allow_html=True)
        with st.expander("Show answer & explanation"):
            st.markdown(f"**Answer: {m['answer']}** — {m['explanation']}")
    with st.expander("Raw model output"):
        st.markdown(raw)


# ---------------------------------------------------------------------------
ROUTES = {
    "💬 AI Tutor": page_chat,
    "⚖️ Base vs Fine-Tuned": page_compare,
    "📝 Quiz Mode": page_quiz,
    "🎤 Viva Mode": lambda: examiner_page("viva"),
    "💼 Interview Mode": lambda: examiner_page("interview"),
    "❓ MCQ Generator": page_mcq,
}
ROUTES[page]()
