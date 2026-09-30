"""Builds docs/EduTune_AI_Project_Report.pdf from the project's real data and results."""
from pdf_style import *  # noqa: F401,F403
from pdf_style import ROOT, config  # noqa: F401
import csv
import json
from collections import Counter

# ---------------------------------------------------------------- data
summary = json.loads((ROOT / "results/training_summary.json").read_text())
ev = json.loads((ROOT / "results/eval_metrics.json").read_text())
comp = json.loads((ROOT / "results/comparison.json").read_text(encoding="utf-8"))["comparisons"]
history = list(csv.DictReader(open(ROOT / "results/training_history.csv")))
dataset = [json.loads(l) for l in open(ROOT / "data/instruction_dataset.jsonl", encoding="utf-8")]
train = [json.loads(l) for l in open(ROOT / "data/train.jsonl", encoding="utf-8")]
val = [json.loads(l) for l in open(ROOT / "data/validation.jsonl", encoding="utf-8")]
kb = json.loads((ROOT / "data/raw/knowledge_base.json").read_text(encoding="utf-8"))
cat_counts = Counter(r["category"] for r in dataset)
dom_counts = Counter(r["domain"] for r in dataset)
val_cat = Counter(r["category"] for r in val)
lengths = [len(r["output"].split()) for r in dataset]

from utils.prompts import ACTIONS, SIMPLIFY_ACTION, TEACHING_MODES  # noqa: E402

try:
    from transformers import AutoTokenizer
    from training.format_dataset import format_instruction
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME, local_files_only=True)
    CHAT_EXAMPLE = format_instruction({"instruction": "Explain LSTM using a simple real-world analogy.",
                                       "input": "", "output": "**Analogy:** Imagine you are reading a long story..."},
                                      tok)
except Exception:  # noqa: BLE001
    CHAT_EXAMPLE = None


def fmt(n):
    return f"{n:,}"


TP, TOT = summary["trainable_parameters"], summary["total_parameters"]
BASE_PPL, FT_PPL = ev["base_model"]["perplexity"], ev["finetuned_model"]["perplexity"]
BASE_L, FT_L = ev["base_model"]["validation_loss"], ev["finetuned_model"]["validation_loss"]


# ---------------------------------------------------------------- diagrams
def arrow(d, x1, y1, x2, y2, color=VIOLET):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.4))
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    s = 5
    p1 = (x2 - s * math.cos(ang - 0.45), y2 - s * math.sin(ang - 0.45))
    p2 = (x2 - s * math.cos(ang + 0.45), y2 - s * math.sin(ang + 0.45))
    d.add(Polygon([x2, y2, p1[0], p1[1], p2[0], p2[1]], fillColor=color, strokeColor=color))


def box(d, x, y, w, h, title, sub="", fill=SOFT, edge=INDIGO):
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=edge, strokeWidth=1))
    if sub:
        d.add(String(x + w / 2, y + h - 14, title, fontName="Arial-Bold", fontSize=8.8, fillColor=INK,
                     textAnchor="middle"))
        for i, line in enumerate(sub.split("\n")):
            d.add(String(x + w / 2, y + h - 26 - 10 * i, line, fontName="Arial", fontSize=7.4,
                         fillColor=MUTED, textAnchor="middle"))
    else:
        d.add(String(x + w / 2, y + h / 2 - 3, title, fontName="Arial-Bold", fontSize=8.8, fillColor=INK,
                     textAnchor="middle"))


def pipeline_diagram():
    W, H = 470, 262
    d = Drawing(W, H)
    bw, bh = 132, 44
    col = [20, 169, 318]
    rows = [212, 122, 32]
    steps = [
        ("Knowledge base", "37 topics, 18 comparisons\n(hand-written)", SOFT, INDIGO),
        ("Dataset generator", "14 categories\nsame templates as app", SOFT, INDIGO),
        ("Instruction dataset", f"{len(dataset)} examples\ninstruction → response", SOFT, INDIGO),
        ("Cleaning + split", "validate, de-duplicate\n90% / 10% stratified", SOFT, INDIGO),
        ("Base LLM (frozen)", "Qwen2.5-1.5B-Instruct\n4-bit NF4", GREY_BG, MUTED),
        ("QLoRA SFT", "TRL SFTTrainer + PEFT\nloss on response only", PINK_BG, VIOLET),
        ("LoRA adapter", f"{TP / 1e6:.1f} M params, {summary['adapter_size_mb']:.0f} MB\nnot merged", GREEN_BG, GREEN),
        ("Evaluation", "loss, perplexity\nside-by-side answers", SOFT, INDIGO),
        ("Teaching app", "Streamlit: tutor, quiz,\nviva, interview, MCQ", AMBER_BG, AMBER),
    ]
    pos = [(col[0], rows[0]), (col[1], rows[0]), (col[2], rows[0]),
           (col[2], rows[1]), (col[0], rows[1]), (col[1], rows[1]),
           (col[0], rows[2]), (col[1], rows[2]), (col[2], rows[2])]
    for (x, y), (t, s, f, e) in zip(pos, steps):
        box(d, x, y, bw, bh, t, s, f, e)
    arrow(d, col[0] + bw, rows[0] + bh / 2, col[1], rows[0] + bh / 2)
    arrow(d, col[1] + bw, rows[0] + bh / 2, col[2], rows[0] + bh / 2)
    arrow(d, col[2] + bw / 2, rows[0], col[2] + bw / 2, rows[1] + bh)
    arrow(d, col[2], rows[1] + bh / 2, col[1] + bw, rows[1] + bh / 2)
    arrow(d, col[0] + bw, rows[1] + bh / 2, col[1], rows[1] + bh / 2)
    arrow(d, col[1] + bw / 2, rows[1], col[0] + bw / 2, rows[2] + bh)
    arrow(d, col[0] + bw, rows[2] + bh / 2, col[1], rows[2] + bh / 2)
    arrow(d, col[1] + bw, rows[2] + bh / 2, col[2], rows[2] + bh / 2)
    for i, (x, y) in enumerate(pos, 1):
        d.add(Rect(x - 6, y + bh - 8, 14, 14, rx=7, ry=7, fillColor=INDIGO, strokeColor=INDIGO))
        d.add(String(x + 1, y + bh - 4.5, str(i), fontName="Arial-Bold", fontSize=7.5, fillColor=colors.white,
                     textAnchor="middle"))
    d.add(String(W / 2, 6, "Data preparation (1-4)  ·  Training (5-7)  ·  Evaluation & use (8-9)",
                 fontName="Arial-Italic", fontSize=8, fillColor=MUTED, textAnchor="middle"))
    return d


def lora_diagram():
    W, H = 470, 210
    d = Drawing(W, H)
    box(d, 190, 12, 90, 28, "input  x", fill=GREY_BG, edge=MUTED)
    box(d, 60, 80, 150, 50, "W  (d × k)", "pre-trained weight\nFROZEN, stored in 4-bit", GREY_BG, MUTED)
    box(d, 265, 62, 150, 36, "A  (r × k)", "trainable, random init", PINK_BG, VIOLET)
    box(d, 265, 112, 150, 36, "B  (d × r)", "trainable, init = 0", PINK_BG, VIOLET)
    box(d, 175, 168, 120, 30, "h = Wx + (α/r)·BAx", fill=GREEN_BG, edge=GREEN)
    arrow(d, 215, 40, 135, 80, MUTED)
    arrow(d, 255, 40, 340, 62)
    arrow(d, 340, 98, 340, 112)
    arrow(d, 135, 130, 205, 168, MUTED)
    arrow(d, 340, 148, 265, 168)
    d.add(String(430, 100, "rank r = 16", fontName="Arial-Bold", fontSize=8, fillColor=VIOLET))
    d.add(String(430, 88, "α = 32", fontName="Arial-Bold", fontSize=8, fillColor=VIOLET))
    return d


def inference_diagram():
    W, H = 470, 150
    d = Drawing(W, H)
    box(d, 15, 55, 130, 50, "Base weights", "loaded ONCE\n4-bit NF4 on GPU", GREY_BG, MUTED)
    box(d, 170, 55, 130, 50, "+ LoRA adapter", "models/edu_tune_lora\nattached with PEFT", PINK_BG, VIOLET)
    box(d, 325, 95, 135, 42, "Fine-Tuned answer", "adapter ENABLED", GREEN_BG, GREEN)
    box(d, 325, 15, 135, 42, "Base answer", "with disable_adapter()", GREY_BG, MUTED)
    arrow(d, 145, 80, 170, 80)
    arrow(d, 300, 88, 325, 115, GREEN)
    arrow(d, 300, 72, 325, 37, MUTED)
    return d


def memory_chart():
    W, H = 470, 150
    d = Drawing(W, H)
    items = [("Full fine-tuning (fp16 + Adam), approx.", 24.7, colors.HexColor("#ef4444")),
             ("Model weights in 16-bit", 3.1, BASE_C),
             ("Model weights in 4-bit NF4", 1.0, VIOLET),
             ("Peak GPU memory of our QLoRA training", summary["peak_gpu_memory_gb"], GREEN)]
    x0, maxw = 210, 230
    for i, (label, gb, c) in enumerate(items):
        y = H - 30 - i * 32
        d.add(String(x0 - 8, y + 5, label, fontName="Arial", fontSize=8, fillColor=INK, textAnchor="end"))
        w = maxw * gb / 24.7
        d.add(Rect(x0, y, w, 16, fillColor=c, strokeColor=None))
        d.add(String(x0 + w + 5, y + 4.5, f"{gb:.1f} GB", fontName="Arial-Bold", fontSize=8, fillColor=INK))
    gx = x0 + maxw * 8 / 24.7
    d.add(Line(gx, 12, gx, H - 8, strokeColor=AMBER, strokeWidth=1, strokeDashArray=[3, 2]))
    d.add(String(gx + 3, 14, "8 GB laptop GPU", fontName="Arial-Bold", fontSize=7, fillColor=AMBER))
    return d


def category_chart():
    cats = sorted(ev["base_model"]["per_category_loss"])
    base = [ev["base_model"]["per_category_loss"][c] for c in cats]
    ft = [ev["finetuned_model"]["per_category_loss"][c] for c in cats]
    d = Drawing(470, 330)
    bc = HorizontalBarChart()
    bc.x, bc.y, bc.width, bc.height = 150, 30, 290, 280
    bc.data = [base, ft]
    bc.categoryAxis.categoryNames = cats
    bc.categoryAxis.labels.fontName = "Arial"
    bc.categoryAxis.labels.fontSize = 7.5
    bc.categoryAxis.labels.dx = -4
    bc.categoryAxis.reverseDirection = True
    bc.valueAxis.valueMin, bc.valueAxis.valueMax, bc.valueAxis.valueStep = 0, 3, 0.5
    bc.valueAxis.labels.fontName = "Arial"
    bc.valueAxis.labels.fontSize = 7.5
    bc.bars[0].fillColor, bc.bars[1].fillColor = BASE_C, INDIGO
    bc.bars.strokeColor = None
    bc.barSpacing, bc.groupSpacing = 1, 5
    d.add(bc)
    d.add(Rect(160, 318, 10, 8, fillColor=BASE_C, strokeColor=None))
    d.add(String(174, 319, "Base model", fontName="Arial", fontSize=8))
    d.add(Rect(250, 318, 10, 8, fillColor=INDIGO, strokeColor=None))
    d.add(String(264, 319, "Fine-tuned model", fontName="Arial", fontSize=8))
    d.add(String(295, 8, "Validation loss (lower = reference answer is more likely)", fontName="Arial-Italic",
                 fontSize=7.5, fillColor=MUTED, textAnchor="middle"))
    return d


# ---------------------------------------------------------------- document template
class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm,
                         bottomMargin=2 * cm, title="EduTune AI - Project Report",
                         author="EduTune AI project", subject="Instruction-tuned AI teaching assistant (QLoRA)")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=self.cover_bg),
                               PageTemplate("main", [frame], onPage=self.header_footer)])

    def cover_bg(self, c, doc):
        c.saveState()
        c.setFillColor(INDIGO)
        c.rect(0, A4[1] - 9.5 * cm, A4[0], 9.5 * cm, fill=1, stroke=0)
        c.setFillColor(VIOLET)
        c.rect(0, A4[1] - 9.8 * cm, A4[0], 0.3 * cm, fill=1, stroke=0)
        c.restoreState()

    def header_footer(self, c, doc):
        c.saveState()
        c.setFont("Arial", 8)
        c.setFillColor(MUTED)
        c.drawString(self.leftMargin, A4[1] - 1.3 * cm, "EduTune AI — Instruction-Tuned AI Teaching Assistant")
        c.drawRightString(A4[0] - self.rightMargin, A4[1] - 1.3 * cm, "Project Report & Presentation Guide")
        c.setStrokeColor(LINE)
        c.line(self.leftMargin, A4[1] - 1.45 * cm, A4[0] - self.rightMargin, A4[1] - 1.45 * cm)
        c.drawCentredString(A4[0] / 2, 1.1 * cm, f"Page {doc.page}")
        c.restoreState()

    def afterFlowable(self, flowable):
        level = getattr(flowable, "toc_level", None)
        if level is not None:
            text = flowable.getPlainText()
            key = f"h{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


# ---------------------------------------------------------------- content
from reportlab.platypus.doctemplate import NextPageTemplate  # noqa: E402

story = []

# ---- Cover
cover_title = ParagraphStyle("ctitle", fontName="Arial-Bold", fontSize=36, leading=42, textColor=colors.white)
cover_sub = ParagraphStyle("csub", fontName="Arial", fontSize=15, leading=20, textColor=colors.HexColor("#e0e7ff"))
story += [Spacer(1, 1.2 * cm), Paragraph("EduTune AI", cover_title), Spacer(1, 6),
          Paragraph("Instruction-Tuned AI Teaching Assistant<br/>using Supervised Fine-Tuning, LoRA and QLoRA",
                    cover_sub),
          Spacer(1, 3.6 * cm),
          Paragraph("<b>Project Report &amp; Presentation Guide</b>",
                    ParagraphStyle("x", fontName="Arial-Bold", fontSize=16, leading=20, textColor=INDIGO)),
          Spacer(1, 10),
          P("A complete explanation of the project — the idea, the theory, the dataset, the training, the "
            "evaluation, the application, and how to present and defend it in a viva.")]
facts = [
    ["Base model", summary["model_name"]],
    ["Method", "Supervised fine-tuning (instruction tuning) with LoRA on a 4-bit base model = QLoRA"],
    ["Trainable parameters", f"{fmt(TP)} of {fmt(TOT)} ({summary['trainable_percentage']:.2f}%)"],
    ["Training data", f"{len(dataset)} instruction→response examples ({len(train)} train / {len(val)} validation), 37 topics, 14 categories"],
    ["Hardware", f"{summary['gpu']} (8 GB) — training took {summary['runtime_minutes']:.1f} min, peak {summary['peak_gpu_memory_gb']} GB"],
    ["Key result", f"Validation perplexity {BASE_PPL:.2f} (base) → {FT_PPL:.2f} (fine-tuned), a {ev['perplexity_reduction_percent']:.0f}% reduction"],
]
story += [Spacer(1, 14), table([["Project at a glance", ""]] + facts, [4.2 * cm, 12.4 * cm]),
          Spacer(1, 1.8 * cm),
          P("Presented by: ______________________________&nbsp;&nbsp;&nbsp;&nbsp; Date: ____________________"),
          NextPageTemplate("main"), PageBreak()]

# ---- TOC
toc = TableOfContents()
toc.levelStyles = [S["toc1"], S["toc2"]]
toc.dotsMinLevel = 0
story += [Paragraph("Contents", S["h1"]), toc, PageBreak()]

# ---- 1 Executive summary
story += [H1("1. Executive Summary")]
story += [P("EduTune AI is an AI teaching assistant for <b>Artificial Intelligence, Machine Learning, Deep "
            "Learning and Generative AI</b>. It was built by taking a general-purpose, pre-trained open-source "
            "language model and <b>specialising</b> it through <b>instruction tuning</b> — supervised "
            "fine-tuning on examples of the form <i>instruction → ideal teaching response</i>."),
          P("Fine-tuning a whole language model normally needs data-centre GPUs. This project instead uses "
            "<b>PEFT</b> (parameter-efficient fine-tuning) with <b>LoRA</b> adapters, and loads the frozen base "
            "model in <b>4-bit precision (QLoRA)</b>. As a result, the entire training ran on an ordinary 8 GB "
            f"laptop GPU in about {summary['runtime_minutes']:.0f} minutes, updating only "
            f"{summary['trainable_percentage']:.2f}% of the parameters."),
          P("The result is used through a Streamlit web application that offers a tutor chat with nine "
            "teaching modes, a side-by-side <b>Base Model vs Fine-Tuned Model</b> comparison, a scored quiz, "
            "a viva examiner, an interview coach and an MCQ generator."),
          H2("1.1 Key numbers"),
          table([["Item", "Value"],
                 ["Base model", f"{summary['model_name']} ({fmt(summary['total_parameters'] - TP)} parameters, frozen)"],
                 ["Trainable parameters (LoRA)", f"{fmt(TP)} = {summary['trainable_percentage']:.2f}% of the total"],
                 ["Dataset", f"{len(dataset)} examples: {len(train)} training, {len(val)} validation"],
                 ["Training", f"{summary['epochs']} epochs, {summary['optimizer_steps']} optimiser steps, "
                              f"{summary['runtime_minutes']:.1f} minutes, peak GPU memory {summary['peak_gpu_memory_gb']} GB"],
                 ["Output", f"LoRA adapter of {summary['adapter_size_mb']:.0f} MB (the full model is ~3 GB)"],
                 ["Validation loss (response tokens)", f"{BASE_L} (base) → {FT_L} (fine-tuned)"],
                 ["Validation perplexity", f"{BASE_PPL} (base) → {FT_PPL} (fine-tuned): −{ev['perplexity_reduction_percent']:.0f}%"],
                 ["Tests", "8 / 8 automated system tests pass"]],
                [5.6 * cm, 11 * cm]),
          Spacer(1, 10),
          say("“I took a pretrained language model and specialised it into an AI Teaching Assistant using "
              "instruction tuning. I used PEFT with LoRA, and QLoRA to reduce GPU memory requirements. The model "
              "was trained on educational instruction-response examples and can now perform different teaching "
              "tasks such as simple explanations, analogies, viva preparation, interview preparation, MCQs, and "
              "quizzes.”"),
          PageBreak()]

# ---- 2 Problem & objectives
story += [H1("2. Problem Statement and Objectives"),
          H2("2.1 Problem statement"),
          P("General-purpose large language models (LLMs) are trained to be broadly helpful. They are <b>not "
            "optimised for a particular teaching style or domain</b>. When a student asks one to “explain LSTM "
            "to a beginner” or to “act as a viva examiner”, the answer's structure, depth and tone vary from "
            "question to question. They also rarely follow a fixed output format that an application can rely "
            "on — for example, a multiple-choice question with a machine-readable answer key."),
          P("Specialising a model by <b>fine-tuning</b> solves this, but full fine-tuning is expensive: updating "
            "all 1.5 billion parameters with the Adam optimiser needs roughly 25 GB of GPU memory, far beyond a "
            "student laptop."),
          H2("2.2 Objectives"),
          *numbered([
              "Build an <b>instruction-tuning dataset</b> of high-quality educational examples across AI, ML, DL and GenAI.",
              "Perform <b>genuine supervised fine-tuning</b> (not just prompt engineering) using <b>PEFT + LoRA</b>.",
              "Use <b>QLoRA (4-bit quantisation)</b> so the training fits on a consumer laptop GPU.",
              "Evaluate the fine-tuned model against the base model <b>honestly</b>: quantitative metrics plus side-by-side answers, with no invented 'accuracy'.",
              "Deliver an interactive <b>teaching application</b>: teaching modes, quiz, viva, interview, MCQs.",
              "Keep the code modular and configurable (one <font face='Consolas'>config.py</font>) and handle errors gracefully.",
          ]),
          H2("2.3 Scope"),
          P("The assistant covers 37 topics, including AI fundamentals, intelligent agents, search, knowledge "
            "representation, supervised and unsupervised learning, regression, classification, clustering, "
            "decision trees, random forests, SVM, KNN, gradient descent, over- and underfitting, neural "
            "networks, backpropagation, activation functions, vanishing gradients, CNN, RNN, LSTM, GRU, GAN, "
            "VAE, transformers, attention, LLMs, prompt engineering, fine-tuning, LoRA, QLoRA, PEFT, instruction "
            "tuning, RLHF and DPO."),
          callout("Why not just use ChatGPT with a good prompt?",
                  "Prompt engineering changes only the <i>input</i>; the model's weights stay the same, so its behaviour "
                  "stays brittle and long prompts cost tokens on every request. The project requirement was to "
                  "<b>actually change the model</b> — to teach it a behaviour through training — and to run it "
                  "<b>locally</b> without any paid API. Fine-tuning gives consistent behaviour; LoRA and QLoRA "
                  "make it affordable."),
          PageBreak()]

# ---- 3 Concepts
story += [H1("3. Background Concepts"),
          P("This chapter explains the theory behind every technique used. These are the concepts examiners "
            "most often ask about.")]
story += [H2("3.1 Large language models and next-token prediction"),
          P("An LLM is a transformer neural network trained on huge amounts of text to predict the <b>next "
            "token</b> (a word or piece of a word). Given “The cat sat on the”, it outputs a probability for "
            "every token in its vocabulary, and “mat” gets a high probability. Text is generated one token at a "
            "time: predict, append, repeat. Doing this over trillions of tokens teaches grammar, facts and "
            "reasoning patterns."),
          P("The training loss is <b>cross-entropy</b>: −log P(correct next token). A lower loss means the model "
            "assigns higher probability to the correct text.")]
story += [H2("3.2 Pre-training vs fine-tuning"),
          table([["", "Pre-training", "Fine-tuning"],
                 ["Data", "Trillions of tokens of raw text", "Hundreds to thousands of task examples"],
                 ["Goal", "General language knowledge", "Specialise for a task, domain or style"],
                 ["Cost", "Millions of dollars, GPU clusters", "Minutes to hours; here 16.5 min on a laptop"],
                 ["Who did it here", "Alibaba (Qwen team), before this project", "This project"]],
                [3.2 * cm, 6.7 * cm, 6.7 * cm]),
          Spacer(1, 6),
          P("Fine-tuning is a form of <b>transfer learning</b>: we reuse what the model already knows and only teach "
            "the new behaviour.")]
story += [H2("3.3 Instruction tuning and supervised fine-tuning (SFT)"),
          P("<b>Supervised fine-tuning</b> means training on examples that contain the <i>desired output</i>. "
            "<b>Instruction tuning</b> is SFT where every example is an <i>instruction</i> (optionally with an "
            "<i>input</i>) paired with the ideal <i>response</i>. The model learns <b>how to respond</b> — style, "
            "structure and format — more than new facts."),
          code('{"instruction": "Explain LSTM using a simple real-world analogy.",\n'
               ' "input": "",\n'
               ' "output": "**Analogy:** Imagine you are reading a long story...",\n'
               ' "category": "analogy"}'),
          P("The optional <b>input</b> field carries extra context. In this project it holds, for example, a "
            "student's viva answer that the model must evaluate, or an explanation it must simplify.")]
story += [H2("3.4 PEFT — parameter-efficient fine-tuning"),
          P("In <b>full fine-tuning</b> every weight is updated. That needs memory for the weights, their "
            "gradients and the optimiser state (Adam keeps two extra values per parameter), about 16 bytes per "
            "parameter. It also produces a complete new copy of the model for every task."),
          P("<b>PEFT</b> freezes the pre-trained model and trains only a <b>small number of extra parameters</b>. "
            "Gradients and optimiser states are needed only for those, so memory drops sharply, the saved result "
            "is small, and the original model is untouched. Methods include LoRA, adapters, prefix tuning, prompt "
            "tuning and IA3. This project uses <b>LoRA</b> through the Hugging Face <font face='Consolas'>peft</font> "
            "library.")]
story += [H2("3.5 LoRA — low-rank adaptation"),
          P("For a frozen weight matrix <b>W</b> of size d × k, LoRA does not learn a full update ΔW (d × k "
            "numbers). It learns two thin matrices, <b>B</b> (d × r) and <b>A</b> (r × k), whose product is the "
            "update: ΔW = B·A. The rank <b>r</b> is small (16 here). The layer then computes:"),
          callout("LoRA forward pass", "<font face='Consolas' size='11'>h = W·x + (α / r) · B·A·x</font><br/>"
                  "W is frozen; only A and B are trained. α (lora_alpha = 32) scales the update. B starts at zero, so "
                  "at step 0 the model behaves exactly like the original."),
          lora_diagram(),
          Paragraph("Figure 1 — LoRA inside one linear layer", S["caption"]),
          P("<b>Worked example (from this model).</b> A 1536 × 1536 attention projection has 2,359,296 weights. "
            "With r = 16, LoRA trains A (16 × 1536) and B (1536 × 16) = 16 × (1536 + 1536) = <b>49,152</b> "
            "parameters, which is 2.1% of that layer."),
          P("<b>Why it works:</b> research found that the weight change needed to adapt a pre-trained model has a "
            "low “intrinsic rank”, so a low-rank update captures most of it.")]
story += [H2("3.6 Quantisation and QLoRA"),
          P("<b>Quantisation</b> stores numbers with fewer bits. A 16-bit weight uses 2 bytes; a 4-bit weight uses "
            "0.5 bytes. QLoRA (Dettmers et al., 2023) combines a <b>4-bit quantised, frozen base model</b> with "
            "<b>LoRA adapters kept in 16-bit (bf16)</b>. It introduced three ideas:"),
          *bullets([
              "<b>NF4 (4-bit NormalFloat)</b> — a 4-bit data type whose 16 levels are placed to suit normally distributed neural-network weights.",
              "<b>Double quantisation</b> — the scaling constants of each weight block are themselves quantised, saving about 0.4 bits per parameter.",
              "<b>Paged optimisers</b> — optimiser memory can spill to CPU RAM during spikes instead of crashing with “out of memory”.",
          ]),
          P("During each forward and backward pass, the 4-bit weights are <b>dequantised on the fly</b> to bf16 "
            "for the matrix multiplication. Gradients flow <i>through</i> the frozen weights into the LoRA "
            "matrices. The base weights never change, so rounding error in them does not accumulate, and the "
            "adapters can learn to compensate for it."),
          memory_chart(),
          Paragraph("Figure 2 — GPU memory: why QLoRA makes training fit on a laptop", S["caption"])]
story += [H2("3.7 Completion-only loss"),
          P("Each training text contains a system message, the student's instruction and the response. We want the "
            "model to learn to <b>produce responses</b>, not to reproduce questions. So the loss is computed only on "
            "the response tokens; the prompt tokens are <b>masked</b> (label = −100). In TRL this is "
            "<font face='Consolas'>completion_only_loss=True</font> with a prompt/completion dataset.")]
story += [H2("3.8 Perplexity — and why not “accuracy”"),
          P("<b>Perplexity = e<super>loss</super></b>. It can be read as “how many tokens the model is choosing "
            "between” at each step: perplexity 8.6 means the model is roughly as uncertain as picking among 8.6 "
            "equally likely tokens; 2.5 means it is much more confident about the reference answer."),
          P("Classification accuracy is <b>not meaningful</b> for free-form explanations: many different answers "
            "can be correct, and exact matching would count them all as wrong. So the project uses validation "
            "loss and perplexity for measurement, and side-by-side reading for quality — and it never claims one "
            "automatic “quality score”."),
          PageBreak()]

# ---- 4 Architecture
story += [H1("4. System Architecture"),
          P("The project is a pipeline of nine stages. Each stage is a separate script or module, so it can be "
            "run, tested and explained on its own."),
          pipeline_diagram(),
          Paragraph("Figure 3 — End-to-end pipeline", S["caption"]),
          table([["#", "Stage", "File", "What happens"],
                 ["1", "Knowledge base", "scripts/knowledge/*.py", "Hand-written, structured content for 37 topics and 18 comparisons"],
                 ["2", "Dataset generator", "scripts/generate_dataset.py", "Turns every field into instruction→response pairs in 14 categories"],
                 ["3", "Instruction dataset", "data/instruction_dataset.jsonl", f"{len(dataset)} JSON records"],
                 ["4", "Cleaning + split", "scripts/prepare_dataset.py", "Validates, removes empty/duplicate records, 90/10 stratified split"],
                 ["5", "Base LLM", "utils/model_loader.py", "Loads Qwen2.5-1.5B-Instruct in 4-bit NF4, frozen"],
                 ["6", "QLoRA SFT", "training/train_qlora.py", "Adds LoRA adapters and trains them with TRL's SFTTrainer"],
                 ["7", "LoRA adapter", "models/edu_tune_lora/", "Saved separately — not merged into the base model"],
                 ["8", "Evaluation", "evaluation/evaluate.py", "Loss and perplexity for both models + side-by-side answers"],
                 ["9", "Teaching app", "app.py", "Streamlit interface using the base model + adapter"]],
                [0.7 * cm, 3.2 * cm, 5 * cm, 7.7 * cm]),
          H2("4.1 Inference architecture: one model, two behaviours"),
          P("At run time, the base weights are loaded <b>once</b> and the adapter is attached on top. For a "
            "“Base Model” answer the adapter is temporarily switched off with PEFT's "
            "<font face='Consolas'>disable_adapter()</font>. This halves GPU memory compared with loading two "
            "models, and it guarantees that the <b>only</b> difference between the two answers is the trained "
            "adapter — the same weights, the same prompt and the same system message."),
          inference_diagram(),
          Paragraph("Figure 4 — How the app produces base and fine-tuned answers from one loaded model", S["caption"])]

# ---- 5 Tech stack
story += [H1("5. Technology Stack"),
          table([["Technology", "Version", "Role in this project"],
                 ["Python", "3.11", "Language for the whole pipeline"],
                 ["PyTorch", "2.11 (CUDA 12.8)", "Tensors and GPU execution; the cu128 build is required for RTX 50-series (Blackwell) GPUs"],
                 ["Transformers", "5.17", "Loads the model and tokenizer, chat template, text generation, BitsAndBytesConfig"],
                 ["Datasets", "5.0", "Loads the JSONL training/validation files for the trainer"],
                 ["PEFT", "0.21", "LoraConfig, get_peft_model (adds adapters, freezes base), PeftModel, disable_adapter"],
                 ["TRL", "1.14", "SFTTrainer / SFTConfig: supervised fine-tuning with completion-only loss"],
                 ["bitsandbytes", "0.50", "4-bit NF4 quantisation and the paged 8-bit AdamW optimiser"],
                 ["Accelerate", "1.15", "Device placement and mixed precision inside the Trainer"],
                 ["Streamlit", "1.64", "Web interface: chat, comparison, quiz, viva, interview, MCQ pages"],
                 ["Pandas / NumPy", "3.0 / 2.4", "Training history, statistics, tables"],
                 ["scikit-learn", "1.9", "Stratified train/validation split"],
                 ["Matplotlib", "3.11", "Training-loss plot"]],
                [3.2 * cm, 3 * cm, 10.4 * cm]),
          Spacer(1, 8),
          P("No paid API is used — everything runs locally. The model is downloaded once from Hugging Face "
            "(Apache-2.0 licence)."),
          H1("6. Project Structure"),
          table([["Path", "Purpose"],
                 ["config.py", "ALL settings: model name, paths, hyper-parameters, LoRA, 4-bit, generation, disclaimer"],
                 ["app.py", "Streamlit application (6 pages)"],
                 ["utils/device.py", "CUDA detection, precision choice (bf16/fp16/fp32), out-of-memory help text"],
                 ["utils/model_loader.py", "load_base_model(), load_finetuned_model(), TeachingModel (adapter on/off, streaming)"],
                 ["utils/prompts.py", "Teaching modes, quick actions, build_prompt(), viva/interview/MCQ prompt builders"],
                 ["utils/mcq.py", "Parses MCQ text into questions/options/answers; curated question bank"],
                 ["training/format_dataset.py", "format_instruction() — the single place where records become model input"],
                 ["training/train_qlora.py", "QLoRA training, parameter report, logging, adapter saving"],
                 ["evaluation/evaluate.py", "Validation loss/perplexity + side-by-side generation"],
                 ["evaluation/comparison.py", "Neutral, measurable response metrics (no 'winner')"],
                 ["scripts/knowledge/", "Structured educational content: ai, ml, dl, genai topics + comparisons"],
                 ["scripts/generate_dataset.py", "Builds the instruction dataset"],
                 ["scripts/prepare_dataset.py", "Cleaning, validation, statistics, split"],
                 ["scripts/test_model.py", "PASS/FAIL system tests"],
                 ["data/", "instruction_dataset.jsonl, train.jsonl, validation.jsonl, raw/knowledge_base.json"],
                 ["models/edu_tune_lora/", "The trained LoRA adapter (adapter_config.json, adapter_model.safetensors)"],
                 ["results/", "training_history.csv, training_loss.png, training_summary.json, eval_metrics.json, comparison.json"]],
                [5.2 * cm, 11.4 * cm]),
          PageBreak()]

# ---- 7 Dataset
topics_by_domain = {}
for t in kb["topics"]:
    topics_by_domain.setdefault(t["domain"], []).append(t["name"])
CAT_EXAMPLES = {}
for r in dataset:
    CAT_EXAMPLES.setdefault(r["category"], r["instruction"])
story += [H1("7. The Dataset"),
          H2("7.1 Design approach"),
          P("Quality matters more than quantity in instruction tuning, and the brief warned against repetitive "
            "filler. So the content was written <b>once, by hand, as structured knowledge</b> for each topic, and "
            "a generator turns it into many different teaching tasks. Every fact is written once, which keeps the "
            "answers consistent, and each task type (definition, analogy, viva feedback, MCQ...) gets a distinct, "
            "well-formatted response."),
          table([["Domain", "Topics"]] + [[d, ", ".join(v)] for d, v in topics_by_domain.items()],
                [3.6 * cm, 13 * cm]),
          Spacer(1, 6),
          H2("7.2 Fields written for every topic"),
          table([["Field", "Used to create"],
                 ["definition, question", "Definition answers; exam-ready lines"],
                 ["simple, eli10", "Beginner and “explain like I'm 10” explanations; simplification targets"],
                 ["analogy", "Real-world analogy answers"],
                 ["how_it_works (steps)", "Detailed, step-by-step technical explanations"],
                 ["example", "Worked examples"],
                 ["applications, advantages, limitations", "Applications and pros/cons answers"],
                 ["key_points", "Revision notes, “key idea to remember”"],
                 ["viva (Q&A pairs)", "Viva question lists, single examiner questions, feedback on student answers"],
                 ["interview (Q&A pairs)", "Interview answers, interviewer questions, evaluation of candidate answers"],
                 ["mcqs", "Multiple-choice question sets and single quiz questions"],
                 ["numerical (optional)", "Step-by-step numerical problems (e.g. CNN output size, LoRA parameter count)"],
                 ["troubleshooting (optional)", "Diagnose-and-fix answers (e.g. NaN loss, CUDA out of memory)"]],
                [5.2 * cm, 11.4 * cm]),
          H2("7.3 Generated categories"),
          P("The generator produced <b>14 categories</b>. The instructions are phrased with the <b>same templates "
            "the app uses</b> (<font face='Consolas'>utils/prompts.py</font>), so the model is trained on exactly "
            "the prompts it sees at demo time. Wording is varied with a seeded random generator, so the model "
            "learns the task rather than one phrasing."),
          table([["Category", "Count", "Example instruction"]] +
                [[c, str(n), esc(CAT_EXAMPLES[c][:95])] for c, n in cat_counts.most_common()],
                [3.9 * cm, 1.4 * cm, 11.3 * cm])]
sample = next(r for r in dataset if r["category"] == "viva" and r["input"])
story += [H2("7.4 Example of a record that uses the input field"),
          P("Viva feedback teaches the model to act as an examiner. The student's answer is in <b>input</b>:"),
          code(f"instruction: {sample['instruction']}\n\ninput: {sample['input']}\n\noutput:\n{sample['output']}")]
story += [H2("7.5 Cleaning and splitting"),
          P("<font face='Consolas'>scripts/prepare_dataset.py</font> performs these steps:"),
          *numbered(["Load the JSONL file; malformed lines are reported, not fatal.",
                     "Remove invalid records (missing fields or wrong types).",
                     "Remove empty instructions and empty outputs.",
                     "Remove duplicates, comparing text case- and whitespace-insensitively.",
                     "Validate the required fields (instruction, input, output).",
                     "Split 90% / 10%, <b>stratified by category</b>, so every category appears in validation."]),
          table([["Statistic", "Value"],
                 ["Total examples", str(len(dataset))],
                 ["Training / validation", f"{len(train)} / {len(val)}"],
                 ["Records removed by cleaning", "0 (the generator already produces valid, unique records)"],
                 ["Domains", ", ".join(f"{d}: {n}" for d, n in dom_counts.most_common())],
                 ["Response length (words)", f"min {min(lengths)}, mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}"]],
                [5.2 * cm, 11.4 * cm]),
          H2("7.6 Instruction formatting"),
          P("<font face='Consolas'>training/format_dataset.py</font> holds the single "
            "<font face='Consolas'>format_instruction()</font> function. Each record becomes a conversation — "
            "system message, user message (instruction + input), assistant message (output) — rendered with the "
            "model's <b>own chat template</b>. This is exactly what the model sees:")]
if CHAT_EXAMPLE:
    story += [code(CHAT_EXAMPLE.strip())]
story += [P("A readable “### Instruction / ### Response” (Alpaca-style) version is also available for display "
            "and for models without a chat template. The same system message is used in training and in the app, "
            "so the <b>only</b> difference between base and fine-tuned answers is the adapter."),
          PageBreak()]

# ---- 8 Training
per_layer = [("q_proj", 1536, 1536), ("k_proj", 1536, 256), ("v_proj", 1536, 256), ("o_proj", 1536, 1536),
             ("gate_proj", 1536, 8960), ("up_proj", 1536, 8960), ("down_proj", 8960, 1536)]
rows = [["Module", "Shape (in → out)", "Full weights", "LoRA params r·(in+out)"]]
tot_layer = 0
for name, i, o in per_layer:
    lp = 16 * (i + o)
    tot_layer += lp
    rows.append([name, f"{i} → {o}", fmt(i * o), fmt(lp)])
rows.append(["<b>Per decoder layer</b>", "", "", f"<b>{fmt(tot_layer)}</b>"])
rows.append(["<b>× 28 layers</b>", "", "", f"<b>{fmt(tot_layer * 28)}</b>"])
val_rows = [["Step", "Epoch", "Validation loss", "Perplexity"]] + [
    [h["step"], h["epoch"], f"{float(h['val_loss']):.4f}", f"{2.718281828 ** float(h['val_loss']):.2f}"]
    for h in history if h["val_loss"]]

story += [H1("8. Model and Training"),
          H2("8.1 Choice of base model"),
          P(f"<b>{summary['model_name']}</b> — 1.54 billion parameters, 28 transformer layers, hidden size 1536, "
            "Apache-2.0 licence. It is small enough to fine-tune with QLoRA on an 8 GB laptop GPU, it is "
            "already instruction-capable (it has a chat template), and it is strong for its size. The model is "
            "set in one place — <font face='Consolas'>MODEL_NAME</font> in <font face='Consolas'>config.py</font> "
            "— so it can be swapped for, e.g., Qwen2.5-0.5B (for weaker hardware) or Qwen2.5-3B."),
          H2("8.2 Training configuration"),
          table([["Setting", "Value", "Why"],
                 ["Quantisation", "4-bit NF4 + double quant., bf16 compute", "QLoRA: fits the 8 GB GPU"],
                 ["LoRA rank r / alpha / dropout", "16 / 32 / 0.05", "Standard starting point; alpha = 2r"],
                 ["Target modules", "q, k, v, o, gate, up, down projections", "Adapting attention + MLP layers gives the best quality"],
                 ["Epochs", str(summary["epochs"]), "Enough for the loss to plateau"],
                 ["Batch size × grad. accumulation", f"{summary['batch_size']} × {summary['gradient_accumulation_steps']} = {summary['effective_batch_size']}", "Small micro-batches save memory; accumulation keeps the effective batch at 8"],
                 ["Learning rate / schedule", f"{summary['learning_rate']} / cosine with 5% warm-up", "Typical LoRA LR (higher than full fine-tuning)"],
                 ["Max sequence length", str(summary["max_seq_length"]), "Longest examples fit without truncation"],
                 ["Optimiser", "paged AdamW 8-bit", "QLoRA's memory-safe optimiser"],
                 ["Gradient checkpointing", "On", "Recomputes activations to save memory"],
                 ["Loss", "Completion-only cross-entropy", "Learn answers, not questions"]],
                [4.6 * cm, 5.4 * cm, 6.6 * cm]),
          H2("8.3 Where the trainable parameters come from"),
          P("LoRA adds A and B to each of the 7 projection layers in all 28 decoder layers. The count below "
            "exactly matches what the training script printed:"),
          table(rows, [3.4 * cm, 3.8 * cm, 3.6 * cm, 5.8 * cm]),
          Spacer(1, 6),
          callout("Printed by training/train_qlora.py",
                  f"<font face='Consolas'>Total parameters: {fmt(TOT)}<br/>Trainable parameters: {fmt(TP)}<br/>"
                  f"Trainable percentage: {summary['trainable_percentage']:.4f}%</font>"),
          H2("8.4 What the training script does, step by step"),
          *numbered([
              "Prints the hardware (CUDA available, GPU name, memory) — and warns clearly if there is no GPU.",
              "Loads train/validation JSONL and converts each record to a prompt/completion conversation.",
              "Loads the base model in 4-bit (<font face='Consolas'>BitsAndBytesConfig</font>) and prepares it for k-bit training.",
              "Wraps it with <font face='Consolas'>get_peft_model(model, LoraConfig(...))</font>: base frozen, adapters trainable.",
              "Prints total / trainable parameters and the trainable percentage (the PEFT demonstration).",
              "Trains with TRL's <font face='Consolas'>SFTTrainer</font>, logging epoch, step, training loss, validation loss and learning rate.",
              "Catches CUDA out-of-memory errors and prints which config values to reduce.",
              "Saves <b>only the adapter</b> to <font face='Consolas'>models/edu_tune_lora/</font> (not merged), plus the history CSV, loss plot and a summary JSON.",
          ]),
          H2("8.5 Training results"),
          Image(str(ROOT / "results/training_loss.png"), width=16 * cm, height=9 * cm),
          Paragraph("Figure 5 — Training and validation loss (results/training_loss.png)", S["caption"]),
          callout("How to read the curve (important for the viva)",
                  "Validation loss falls quickly in epoch 1 (1.55 → 1.08), keeps improving in epoch 2, and "
                  "<b>plateaus from about step 175</b> (0.867 → 0.864). Meanwhile, training loss keeps falling to "
                  "about 0.4. This widening gap in epoch 3 is the <b>beginning of memorisation</b>: the model fits the "
                  "training examples more closely without generalising better. Validation loss does not rise, so "
                  "the model is not harmed, but <b>2 epochs would have given almost the same result</b>. Early "
                  "stopping on validation loss would be the principled fix.", bg=AMBER_BG, edge=AMBER),
          table(val_rows, [2.5 * cm, 2.5 * cm, 4 * cm, 4 * cm]),
          Spacer(1, 8),
          table([["Result", "Value"],
                 ["Runtime", f"{summary['runtime_minutes']} minutes"],
                 ["Peak GPU memory", f"{summary['peak_gpu_memory_gb']} GB (of 8 GB)"],
                 ["Final validation loss / perplexity (trainer)", f"{summary['final_validation_loss']} / {summary['final_validation_perplexity']}"],
                 ["Average training loss over the run", str(summary["final_train_loss"])],
                 ["Adapter size", f"{summary['adapter_size_mb']} MB"]],
                [7 * cm, 9.6 * cm]),
          Spacer(1, 8),
          PageBreak()]

# ---- 9 Evaluation
def pick(q):
    return next(c for c in comp if c["question"] == q or c["instruction"] == q)


def cmp_table(item, limit=900):
    b = item["base_model_response"]
    f = item["finetuned_model_response"]
    b = b if len(b) <= limit else b[:limit].rsplit(" ", 1)[0] + " …"
    f = f if len(f) <= limit else f[:limit].rsplit(" ", 1)[0] + " …"
    t = Table([[Paragraph("<b>BASE MODEL</b>", S["cellb"]), Paragraph("<b>FINE-TUNED MODEL</b>", S["cellb"])],
               [Paragraph(md(b), S["cell"]), Paragraph(md(f), S["cell"])]],
              colWidths=[8.3 * cm, 8.3 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), BASE_C), ("BACKGROUND", (1, 0), (1, 0), INDIGO),
                           ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
    obs = [Paragraph(esc(o), S["small"], bulletText="•") for o in item["observations"]]
    return [H3(f"Prompt: “{esc(item['instruction'])}”"), t, Spacer(1, 4)] + obs + [Spacer(1, 8)]


story += [H1("9. Evaluation"),
          H2("9.1 Methodology"),
          P("<font face='Consolas'>evaluation/evaluate.py</font> compares the <b>same loaded model</b> with the "
            "adapter off (base) and on (fine-tuned):"),
          *bullets([
              f"<b>Quantitative:</b> validation loss and perplexity on the {len(val)} held-out examples, counting "
              "<b>only the response tokens</b>, with a per-category breakdown.",
              "<b>Qualitative:</b> both models answer 14 fixed questions with greedy (deterministic) decoding. These "
              "include the required ones (Explain RNN, Explain LSTM, vanishing gradient, Compare RNN and LSTM, "
              "CNN for beginners, overfitting, LoRA, QLoRA), several teaching modes, and <b>3 topics that are not in "
              "the training data</b> (batch normalization, dropout, word embeddings).",
              "For each pair, <b>measurable differences</b> are recorded — length, structure, analogy/example use, "
              "MCQ format compliance. <b>No automatic winner is declared.</b>",
          ]),
          H2("9.2 Quantitative results"),
          table([["Metric", "Base model", "Fine-tuned model", "Change"],
                 ["Validation loss (response tokens)", str(BASE_L), str(FT_L), f"−{(1 - FT_L / BASE_L) * 100:.0f}%"],
                 ["Perplexity", str(BASE_PPL), str(FT_PPL), f"−{ev['perplexity_reduction_percent']:.0f}%"],
                 ["Response tokens scored", fmt(ev["base_model"]["response_tokens"]), fmt(ev["finetuned_model"]["response_tokens"]), ""]],
                [5.6 * cm, 3.4 * cm, 3.8 * cm, 3.8 * cm]),
          Spacer(1, 4),
          P(f"<i>Note: the evaluation script scores every response token equally, so its fine-tuned loss "
            f"({FT_L}) differs slightly from the trainer's own figure ({summary['final_validation_loss']}), which "
            "averages per batch.</i>", "small"),
          category_chart(),
          Paragraph("Figure 6 — Validation loss per category, base vs fine-tuned", S["caption"]),
          callout("Interpreting the per-category results honestly", bullets([
              "The loss drops in <b>every</b> category, most for structured formats (definitions, MCQs, numerical, application).",
              f"<b>Definition</b> falls to {ev['finetuned_model']['per_category_loss']['definition']}. This is "
              "suspiciously low, and the reason is instructive: the same definition text also appears inside "
              "other training examples for that topic (e.g. analogy and detailed answers). The model has "
              "effectively seen it — a form of <b>overlap/leakage</b> between train and validation.",
              f"<b>Troubleshooting</b> barely improves ({ev['base_model']['per_category_loss']['troubleshooting']} → "
              f"{ev['finetuned_model']['per_category_loss']['troubleshooting']}), but there is only "
              f"{val_cat['troubleshooting']} validation example, so this is not statistically meaningful.",
              "Loss measures how likely the model finds the <i>reference</i> answer — the learned style and "
              "content — not whether an answer is factually correct.",
          ]), bg=AMBER_BG, edge=AMBER),
          PageBreak(),
          H2("9.3 Qualitative comparison (real outputs from results/comparison.json)")]
for q in ["Explain like I'm 10 years old: What is an RNN?", "Create 2 multiple-choice questions on RNN.",
          "What is dropout?"]:
    story += cmp_table(pick(q))
story += [H2("9.4 Findings"),
          *bullets([
              "<b>Format following improved clearly.</b> Asked for 2 MCQs, the base model wrote questions with <i>no "
              "answer key</i> (0/2 parseable); the fine-tuned model produced 2/2 in the exact "
              "Question / A–D / Answer / Explanation format that the Quiz page needs.",
              "<b>Consistent teaching structure.</b> Fine-tuned answers are shorter and use labels learned in training "
              "(“Key idea to remember”, “The grown-up version”, “Exam-ready definition”).",
              "<b>Style transfers to unseen topics.</b> For dropout — which was not in the dataset — the model still "
              "answered in the learned viva style.",
              "<b>Not always better.</b> For “Explain CNN to a beginner” the base model's longer answer included an "
              "example and an analogy that the short fine-tuned answer did not. Fine-tuned MCQs can contain errors: "
              "in the example above, options A and D overlap, and in testing one quiz question wrongly described "
              "an “RL loop” in DPO. Instruction tuning on ~750 examples mainly changes behaviour, not the 1.5B "
              "model's knowledge.",
          ]),
          say("“Perplexity on held-out teaching answers dropped by 71%, and the fine-tuned model reliably follows the "
              "formats it was trained on — for example, 2 out of 2 parseable MCQs versus 0 for the base model. But I "
              "don't claim it is better at everything: it is shorter and can still make factual mistakes, which is "
              "why the app shows a disclaimer.”"),
          PageBreak()]

# ---- 10 Application
story += [H1("10. The Teaching Application"),
          P("Run with <font face='Consolas'>streamlit run app.py</font> and open http://localhost:8501. The model "
            "and adapter load once (about 15 seconds) and are cached with <font face='Consolas'>st.cache_resource</font>. "
            "Answers are streamed token by token."),
          H2("10.1 Sidebar"),
          *bullets(["<b>Navigation</b> between the six pages.",
                    "<b>Teaching mode</b> — changes the instruction sent to the model (see 10.3).",
                    "<b>Topic</b> — used by quick actions, quiz, viva, interview and MCQ pages (or “Any”).",
                    "<b>Generation settings</b> — max new tokens and temperature.",
                    "The <b>safety disclaimer</b>: “EduTune AI is an educational assistant. Its answers may contain "
                    "mistakes. Verify important academic information with textbooks, lecturers, or authoritative sources.”"]),
          H2("10.2 Pages"),
          table([["Page", "What it does"],
                 ["AI Tutor", "Chat with the fine-tuned tutor in the chosen teaching mode, plus seven quick-action buttons. "
                              "Shows the exact instruction sent to the model."],
                 ["Base vs Fine-Tuned", "The same question to both models, side by side, with deterministic decoding and a "
                                        "list of observable differences. Also browses the saved evaluation results."],
                 ["Quiz Mode", "One MCQ at a time, A–D options, “Correct!” or the right answer with an explanation, running "
                               "score (e.g. Score: 4 / 5). Questions come from the model (parsed) or from the curated bank."],
                 ["Viva Mode", "The AI examiner asks one question; the student answers; the examiner gives a verdict, what was "
                               "right, what to add, and a model answer; then the next question."],
                 ["Interview Mode", "Interview question → candidate answer → What you got right / What you missed / "
                                    "Suggested improved answer."],
                 ["MCQ Generator", "Topic, number of questions (1–10), Easy/Medium/Hard → MCQs with hidden answers and "
                                   "explanations."]],
                [3.6 * cm, 13 * cm]),
          H2("10.3 Teaching modes"),
          table([["Mode", "Instruction template sent to the model"]] +
                [[m, esc(v["template"])] for m, v in TEACHING_MODES.items()],
                [4.2 * cm, 12.4 * cm]),
          H2("10.4 Quick actions"),
          table([["Button", "Instruction"]] +
                [[esc(k.split(" ", 1)[1]), esc(v)] for k, v in ACTIONS.items()] +
                [[SIMPLIFY_ACTION.split(" ", 1)[1], "Simplify this explanation so that a beginner can understand it. "
                                                    "(the previous answer is passed as input)"]],
                [4.2 * cm, 12.4 * cm]),
          H2("10.5 How Quiz, Viva and Interview use the fine-tuning"),
          P("These pages depend on formats the model <b>learned during training</b>. The dataset contains "
            "“Create 1 multiple-choice question on X” examples in a fixed format that "
            "<font face='Consolas'>utils/mcq.py</font> can parse, “Ask me one viva question about X” examples, and "
            "examiner-feedback examples whose student answer is in the input field. If the model's MCQ output "
            "cannot be parsed, the quiz falls back to the curated question bank and tells the user so."),
          H2("10.6 Error handling"),
          table([["Situation", "What the user sees"],
                 ["Model cannot be loaded (no internet, wrong name)", "A clear message and a “Retry loading model” button — no traceback"],
                 ["LoRA adapter missing", "A warning explaining how to train it; the base model still works on the comparison page"],
                 ["CUDA not available", "Printed warning in the scripts; the app runs on CPU (slowly)"],
                 ["CUDA out of memory", "A message listing exactly which config.py values to reduce"],
                 ["Missing or invalid dataset", "Scripts stop with instructions to run the generator/preparation steps"],
                 ["Empty question or answer", "“Please type a question first.”"],
                 ["Generation error", "A friendly error message; the app keeps running"]],
                [6.3 * cm, 10.3 * cm]),
          PageBreak()]

# ---- 11 hardware, 12 testing, 13 how to run
story += [H1("11. Hardware and Performance"),
          table([["Aspect", "This project"],
                 ["GPU", f"{summary['gpu']}, 8 GB, Blackwell architecture (needs PyTorch built for CUDA 12.8)"],
                 ["Precision", "Weights 4-bit NF4; computation and adapters in bfloat16"],
                 ["Training memory", f"Peak {summary['peak_gpu_memory_gb']} GB"],
                 ["Training time", f"{summary['runtime_minutes']} minutes for 3 epochs"],
                 ["Inference", "One model in memory; answers are streamed"],
                 ["CPU fallback", "Detected automatically with a clear warning; 4-bit training needs a GPU"]],
                [4 * cm, 12.6 * cm]),
          Spacer(1, 6),
          P("If memory is insufficient, all relevant settings are in <font face='Consolas'>config.py</font>: "
            "BATCH_SIZE = 1 (raise GRADIENT_ACCUMULATION_STEPS), MAX_SEQ_LENGTH = 512, keep "
            "GRADIENT_CHECKPOINTING and USE_4BIT on, LORA_R = 8, or a smaller MODEL_NAME such as Qwen2.5-0.5B-Instruct."),
          H1("12. Testing"),
          P("<font face='Consolas'>python scripts/test_model.py</font> runs end-to-end checks with clear PASS/FAIL output:"),
          code("[PASS] GPU detection              CUDA available - NVIDIA GeForce RTX 5050 Laptop GPU (7.96 GB)\n"
               "[PASS] Dataset loading            instruction_dataset=840, train=756, validation=84\n"
               "[PASS] Dataset formatting         Alpaca + chat prompt/completion formats OK\n"
               "[PASS] Prompt building            9 teaching modes, empty question rejected\n"
               "[PASS] MCQ parser                 MCQ format parsed\n"
               "[PASS] Base model loading         Qwen/Qwen2.5-1.5B-Instruct loaded (4-bit) on cuda:0\n"
               "[PASS] Fine-tuned model loading   LoRA adapter attached (18,464,768 adapter parameters)\n"
               "[PASS] Inference                  base: 'Sure! A Recurrent Neural Network (RNN) is a type o'\n"
               "                                  fine-tuned: 'An RNN is like reading a long story one word at a '\n"
               "8/8 tests passed"),
          P("The Streamlit app was also exercised headlessly with Streamlit's AppTest: every page rendered "
            "without exceptions, and chat, quick actions, quiz scoring and viva feedback worked end to end."),
          H1("13. How to Install and Run"),
          code("# 1. Virtual environment (Windows)\n"
               "py -3.11 -m venv venv\n"
               "venv\\Scripts\\activate\n\n"
               "# 2. PyTorch with CUDA, then the rest\n"
               "pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128\n"
               "pip install -r requirements.txt\n\n"
               "# 3. Pipeline\n"
               "python utils/device.py                # check the GPU\n"
               "python scripts/generate_dataset.py    # build the dataset\n"
               "python scripts/prepare_dataset.py     # clean + split\n"
               "python training/train_qlora.py        # QLoRA training (~17 min)\n"
               "python evaluation/evaluate.py         # base vs fine-tuned evaluation\n"
               "python scripts/test_model.py          # PASS/FAIL tests\n\n"
               "# 4. Application\n"
               "streamlit run app.py                  # http://localhost:8501"),
          P("Inference from Python, without the UI:"),
          code("from utils.model_loader import TeachingModel\n"
               "from utils.prompts import build_prompt\n\n"
               "tm = TeachingModel()                                  # base + LoRA adapter, loaded once\n"
               "q = build_prompt(\"What is an LSTM?\", \"Explain Like I'm 10\")\n"
               "print(tm.generate(q, use_adapter=True))               # fine-tuned\n"
               "print(tm.generate(q, use_adapter=False))              # base model"),
          PageBreak()]

# ---- 14 Presentation guide
story += [H1("14. Presentation Guide"),
          H2("14.1 Suggested slide outline (about 12–15 minutes)"),
          table([["#", "Slide", "Content", "Time"],
                 ["1", "Title", "EduTune AI — instruction-tuned teaching assistant", "0:30"],
                 ["2", "Problem", "General LLMs lack a consistent teaching style; full fine-tuning is too expensive", "1:00"],
                 ["3", "Objective", "Specialise a small open model with instruction tuning on a laptop GPU", "0:30"],
                 ["4", "Architecture", "Pipeline diagram (Figure 3)", "1:00"],
                 ["5", "Dataset", "37 topics → 840 examples, 14 categories, example record", "1:30"],
                 ["6", "Concepts", "Fine-tuning, instruction tuning, SFT, PEFT", "1:00"],
                 ["7", "LoRA + QLoRA", "Figure 1, NF4, memory chart (Figure 2), 1.18% trainable", "1:30"],
                 ["8", "Training", "Configuration + loss curve (Figure 5)", "1:00"],
                 ["9", "Evaluation", "Perplexity −71%, per-category chart, honest findings", "1:30"],
                 ["10", "LIVE DEMO", "See 14.2", "4:00"],
                 ["11", "Limitations & future work", "Chapter 16", "0:45"],
                 ["12", "Conclusion", "Summary sentence + thank you", "0:30"]],
                [0.8 * cm, 3.2 * cm, 10.8 * cm, 1.8 * cm]),
          H2("14.2 Live demo script"),
          P("Start the app a minute before presenting so the model is already loaded."),
          table([["Step", "Do this", "Say this"],
                 ["1. Dataset", "Open data/instruction_dataset.jsonl (or run prepare_dataset.py for the statistics)",
                  "“This is an instruction dataset: each record is an instruction and the ideal teaching response.”"],
                 ["2. Base model", "Base vs Fine-Tuned page, question “Explain RNN to a beginner.” — look at the left column",
                  "“This is the pre-trained model with no training from me.”"],
                 ["3. QLoRA", "Show config.py LoRA settings and results/training_summary.json",
                  "“The base model is quantised to 4-bit and frozen; only LoRA adapters — 1.18% of the parameters — are trained.”"],
                 ["4. Fine-tuned", "Look at the right column for the same question",
                  "“Same weights, same prompt — the only difference is my 46 MB adapter.”"],
                 ["5. Differences", "Scroll to “Observable differences”",
                  "“I report measurable differences instead of claiming an automatic score.”"],
                 ["6. ELI10", "AI Tutor → mode “Explain Like I'm 10” → “What is an LSTM?”",
                  "“The mode changes the instruction; the model was trained on this exact phrasing.”"],
                 ["7. Viva", "Viva Mode → Ask me a question → give a partial answer",
                  "“The model learned to act as an examiner from feedback examples in the dataset.”"],
                 ["8. MCQ / Quiz", "Quiz Mode → answer 2–3 questions; MCQ Generator → 5 medium on RNN",
                  "“The fine-tuned model produces a parseable answer key; the base model doesn't.”"],
                 ["9. Statistics", "Show results/training_loss.png and eval_metrics.json",
                  "“Validation perplexity dropped from 8.6 to 2.5; training took 16.5 minutes on this laptop.”"]],
                [2.3 * cm, 6.5 * cm, 7.8 * cm]),
          callout("Demo safety tips", bullets([
              "Use the <b>Base vs Fine-Tuned</b> page with deterministic decoding for the comparison — results are repeatable.",
              "For the quiz, the <b>Curated question bank</b> option is 100% reliable if a model-generated question looks wrong.",
              "Never re-run training during the demo — it overwrites the adapter.",
              "If the GPU is busy or the model fails to load, the saved results in the comparison page still work.",
          ]), bg=AMBER_BG, edge=AMBER),
          H2("14.3 Numbers to remember"),
          table([["Question", "Answer"],
                 ["Base model?", "Qwen2.5-1.5B-Instruct, 1.54 B parameters, 28 layers"],
                 ["How many parameters trained?", f"{fmt(TP)} — {summary['trainable_percentage']:.2f}%"],
                 ["LoRA settings?", "r = 16, alpha = 32, dropout 0.05, 7 projection modules"],
                 ["Quantisation?", "4-bit NF4 with double quantisation, bf16 compute"],
                 ["Dataset?", f"{len(dataset)} examples, 37 topics, 14 categories, {len(train)}/{len(val)} split"],
                 ["Training?", f"3 epochs, 285 steps, LR 2e-4, effective batch 8, {summary['runtime_minutes']:.1f} min, {summary['peak_gpu_memory_gb']} GB peak"],
                 ["Result?", f"Perplexity {BASE_PPL} → {FT_PPL} (−{ev['perplexity_reduction_percent']:.0f}%)"],
                 ["Adapter size?", f"{summary['adapter_size_mb']:.0f} MB vs ~3 GB for the full model"]],
                [5 * cm, 11.6 * cm]),
          PageBreak()]

# ---- 15 Viva Q&A
QA = [
    ("What is the difference between fine-tuning and prompt engineering?",
     "Prompt engineering changes only the input; the model's weights stay the same. Fine-tuning changes the weights "
     "by training on examples, which gives more consistent behaviour without long prompts. This project fine-tunes."),
    ("What is instruction tuning?",
     "Supervised fine-tuning on (instruction, optional input, response) examples, so the model learns to follow "
     "requests in the desired style and format."),
    ("Why did you use LoRA instead of full fine-tuning?",
     "Full fine-tuning of 1.5 B parameters needs ~25 GB (weights, gradients and Adam states). LoRA trains only 1.18% "
     "of the parameters, so gradients and optimiser states are tiny, the base model stays intact, and the result is "
     "a 46 MB adapter."),
    ("Explain the LoRA equation.",
     "h = Wx + (α/r)·BAx. W is frozen; A (r×k) and B (d×r) are trainable with rank r = 16; α = 32 scales the update. "
     "B starts at zero so training begins from the original model."),
    ("What is the difference between LoRA and QLoRA?",
     "Both train LoRA adapters on a frozen model. In QLoRA the frozen base model is stored in 4-bit NF4 instead of "
     "16-bit, which cuts its memory about 4×. The adapters stay in bf16."),
    ("If the weights are 4-bit, how can training be accurate?",
     "The weights are only stored in 4-bit; for every matrix multiplication they are dequantised to bf16. The base "
     "weights never change, so errors do not accumulate, and the adapters can compensate for quantisation error."),
    ("What is NF4? What is double quantisation?",
     "NF4 is a 4-bit data type whose levels are optimal for normally distributed weights. Double quantisation also "
     "quantises the per-block scaling constants, saving about 0.4 bits per parameter."),
    ("Which layers did you apply LoRA to, and why?",
     "q, k, v, o (attention) and gate, up, down (MLP) in all 28 layers. Adapting both attention and MLP layers "
     "generally gives better quality than attention alone."),
    ("How did you compute the trainable percentage?",
     "PEFT's get_nb_trainable_parameters(): 18,464,768 trainable of 1,562,179,072 total = 1.18%. By hand: "
     "659,456 LoRA parameters per layer × 28 layers."),
    ("Why compute the loss only on the response?",
     "We want the model to learn to generate good answers, not to reproduce questions or the system prompt. "
     "Prompt tokens are masked with label −100."),
    ("How did you build the dataset, and is it good quality?",
     "Hand-written structured content for 37 topics, converted by a generator into 14 task categories using the "
     "app's own templates, then cleaned, de-duplicated and split 90/10 stratified. The facts are consistent, but "
     "it is less diverse than a crowd-sourced dataset."),
    ("Why perplexity and not accuracy?",
     "Many different explanations can be correct, so exact-match accuracy is meaningless for free text. Perplexity "
     "measures how likely the model finds the reference answers; side-by-side reading covers quality."),
    ("Is 71% lower perplexity proof that the model is better?",
     "It proves that the model learned the target style and content of the dataset. It does not prove factual "
     "correctness — which is why I also compare answers qualitatively and report cases where the base model was better."),
    ("Did the model overfit?",
     "Validation loss plateaued from step 175 while training loss kept falling — the start of memorisation — but "
     "validation loss never rose. Two epochs or early stopping would give the same result."),
    ("Why is the definition-category loss so low?",
     "Definition text also appears inside other training examples for the same topic, so there is overlap between "
     "training and validation. A topic-level split would be a stricter test."),
    ("How does the app show base and fine-tuned answers without loading two models?",
     "It loads the base model once, attaches the adapter with PEFT, and uses disable_adapter() for base answers. "
     "Same weights and same prompt, so the only difference is the adapter."),
    ("Why did you not merge the adapter?",
     "Keeping it separate allows switching between base and fine-tuned for the comparison, keeps the file small "
     "(46 MB), and allows several adapters on one base model. Merging is optional and removes the tiny extra latency."),
    ("Why Qwen2.5-1.5B-Instruct?",
     "Open licence, strong for its size, has a chat template, and fits QLoRA training on an 8 GB laptop GPU. It is "
     "configurable in config.py."),
    ("What are RLHF and DPO, and did you use them?",
     "Both align models with human preferences after SFT. RLHF trains a reward model and optimises with PPO; DPO "
     "optimises directly on (chosen, rejected) pairs. This project uses only SFT; DPO is listed as future work."),
    ("What happens if CUDA runs out of memory?",
     "The scripts catch the error and print which settings to reduce: batch size, sequence length, LoRA rank, "
     "4-bit and gradient checkpointing, or a smaller model."),
]
story += [H1("15. Expected Viva Questions and Answers")]
for i, (q, a) in enumerate(QA, 1):
    story += [KeepTogether([Paragraph(f"<b>Q{i}. {q}</b>", ParagraphStyle("q", parent=S["body"], textColor=INDIGO,
                                                                          spaceAfter=2)),
                            Paragraph(a, S["body"])])]
story += [PageBreak()]

# ---- 16 limitations, 17 conclusion
story += [H1("16. Limitations and Future Work"),
          H2("16.1 Limitations"),
          *bullets([
              "<b>Split by example, not by topic.</b> Validation examples share topics (and some text) with training "
              "examples, so perplexity overstates generalisation to new topics.",
              "<b>Generated dataset.</b> Categories for the same topic share facts; the data is consistent but less "
              "diverse than human-written data at scale.",
              "<b>Knowledge is unchanged.</b> ~750 examples teach style and format; the 1.5 B model can still make "
              "factual mistakes (e.g. flawed MCQ options).",
              "<b>Shorter answers.</b> The learned style is concise, which sometimes drops examples a beginner would find useful.",
              "<b>No human evaluation.</b> Teaching quality was not rated by students or teachers.",
          ]),
          H2("16.2 Future work"),
          *bullets([
              "Topic-level train/validation split and a separate held-out test set.",
              "Early stopping on validation loss (2 epochs appear sufficient).",
              "Preference tuning with <b>DPO</b> on (better answer, worse answer) pairs.",
              "Retrieval-augmented generation (RAG) over textbooks to reduce factual errors.",
              "A larger model (Qwen2.5-3B) and a larger, more diverse dataset.",
              "A human evaluation study with students.",
          ]),
          H1("17. Conclusion"),
          P("EduTune AI shows the complete, genuine workflow of specialising a large language model: building an "
            "instruction dataset, fine-tuning with PEFT/LoRA on a 4-bit quantised base model (QLoRA), evaluating "
            "honestly, and deploying the result in a usable teaching application — all on a consumer laptop GPU in "
            "under 20 minutes of training."),
          P(f"By training only {summary['trainable_percentage']:.2f}% of the parameters, the model learned a "
            f"consistent teaching style and reliable output formats, and its perplexity on held-out teaching "
            f"answers dropped from {BASE_PPL} to {FT_PPL}. The evaluation also makes clear what instruction tuning "
            "does <i>not</i> do: it shapes behaviour, but it does not guarantee factual accuracy — which is why "
            "the application carries a disclaimer and the report states its limitations openly."),
          PageBreak()]

# ---- Appendices
story += [H1("Appendix A — Key Code"),
          H2("A.1 4-bit quantisation (utils/model_loader.py)"),
          code("BitsAndBytesConfig(\n"
               "    load_in_4bit=True,\n"
               "    bnb_4bit_quant_type=\"nf4\",           # NormalFloat4\n"
               "    bnb_4bit_use_double_quant=True,      # quantise the scaling constants too\n"
               "    bnb_4bit_compute_dtype=torch.bfloat16,\n"
               ")"),
          H2("A.2 LoRA (training/train_qlora.py)"),
          code("model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)\n"
               "lora_config = LoraConfig(\n"
               "    r=16, lora_alpha=32, lora_dropout=0.05,\n"
               "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",\n"
               "                    \"gate_proj\", \"up_proj\", \"down_proj\"],\n"
               "    bias=\"none\", task_type=\"CAUSAL_LM\",\n"
               ")\n"
               "model = get_peft_model(model, lora_config)      # freeze base, add trainable A and B\n"
               "trainable, total = model.get_nb_trainable_parameters()"),
          H2("A.3 Supervised fine-tuning (training/train_qlora.py)"),
          code("SFTConfig(num_train_epochs=3, per_device_train_batch_size=2, gradient_accumulation_steps=4,\n"
               "          learning_rate=2e-4, lr_scheduler_type=\"cosine\", max_length=1024,\n"
               "          completion_only_loss=True,           # loss on the response only\n"
               "          bf16=True, gradient_checkpointing=True, optim=\"paged_adamw_8bit\",\n"
               "          eval_strategy=\"steps\", eval_steps=25)\n"
               "trainer = SFTTrainer(model=model, args=sft_config, train_dataset=ds[\"train\"],\n"
               "                     eval_dataset=ds[\"validation\"], processing_class=tokenizer)\n"
               "trainer.train()\n"
               "trainer.model.save_pretrained(\"models/edu_tune_lora\")   # adapter only"),
          H2("A.4 Base vs fine-tuned from one model (utils/model_loader.py)"),
          code("model = PeftModel.from_pretrained(base_model, \"models/edu_tune_lora\")\n\n"
               "with model.disable_adapter():      # temporarily = pure base model\n"
               "    base_answer = model.generate(**inputs)\n"
               "finetuned_answer = model.generate(**inputs)   # adapter enabled"),
          H1("Appendix B — Glossary"),
          table([["Term", "Meaning"],
                 ["Adapter", "Small set of extra trainable weights added to a frozen model (here: LoRA matrices)"],
                 ["bf16", "bfloat16 — a 16-bit floating-point format with the range of fp32; used for computation"],
                 ["Chat template", "The model's own format for system/user/assistant messages"],
                 ["Cross-entropy", "Loss = −log probability of the correct token"],
                 ["Epoch", "One full pass over the training data"],
                 ["Gradient accumulation", "Summing gradients over several micro-batches before one update"],
                 ["Gradient checkpointing", "Recomputing activations in the backward pass to save memory"],
                 ["LoRA", "Low-Rank Adaptation: learn ΔW = B·A with small rank r"],
                 ["NF4", "4-bit NormalFloat data type used by QLoRA"],
                 ["PEFT", "Parameter-Efficient Fine-Tuning: train few parameters, freeze the rest"],
                 ["Perplexity", "e<super>loss</super>; lower = the model finds the reference text more likely"],
                 ["QLoRA", "LoRA on a 4-bit quantised, frozen base model"],
                 ["Quantisation", "Storing numbers with fewer bits (here 16 → 4)"],
                 ["SFT", "Supervised Fine-Tuning: training on examples with desired outputs"],
                 ["Token", "A unit of text (word or sub-word) processed by the model"],
                 ["TRL", "Hugging Face library for SFT, DPO and RLHF training"]],
                [4 * cm, 12.6 * cm])]

out = ROOT / "docs" / "EduTune_AI_Project_Report.pdf"
out.parent.mkdir(exist_ok=True)
doc = Doc(str(out))
doc.multiBuild(story)
print("Wrote", out)
