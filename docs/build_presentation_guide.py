"""Builds docs/EduTune_AI_Faculty_Presentation_Guide.pdf - how to present the project to faculty."""
import json

from pdf_style import *  # noqa: F401,F403
from pdf_style import ROOT

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.doctemplate import BaseDocTemplate, NextPageTemplate, PageTemplate
from reportlab.platypus.frames import Frame
from reportlab.platypus.tableofcontents import TableOfContents

summary = json.loads((ROOT / "results/training_summary.json").read_text())
ev = json.loads((ROOT / "results/eval_metrics.json").read_text())
TP, TOT = summary["trainable_parameters"], summary["total_parameters"]
PCT = summary["trainable_percentage"]
BASE_PPL, FT_PPL = ev["base_model"]["perplexity"], ev["finetuned_model"]["perplexity"]
RED = ev["perplexity_reduction_percent"]


class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm,
                         bottomMargin=2 * cm, title="EduTune AI - Faculty Presentation Guide",
                         subject="How to present the EduTune AI project")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=self.cover_bg),
                               PageTemplate("main", [frame], onPage=self.header_footer)])

    def cover_bg(self, c, doc):
        c.saveState()
        c.setFillColor(VIOLET)
        c.rect(0, A4[1] - 9.5 * cm, A4[0], 9.5 * cm, fill=1, stroke=0)
        c.setFillColor(INDIGO)
        c.rect(0, A4[1] - 9.8 * cm, A4[0], 0.3 * cm, fill=1, stroke=0)
        c.restoreState()

    def header_footer(self, c, doc):
        c.saveState()
        c.setFont("Arial", 8)
        c.setFillColor(MUTED)
        c.drawString(self.leftMargin, A4[1] - 1.3 * cm, "EduTune AI — Faculty Presentation Guide")
        c.drawRightString(A4[0] - self.rightMargin, A4[1] - 1.3 * cm, "For the presenter")
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


def slide(num, title, time, on_slide, script, visual=None):
    """A card: what goes on the slide, and what to say."""
    head = Paragraph(f"<b>Slide {num} · {title}</b> <font color='#e0e7ff'>&nbsp;&nbsp;({time})</font>",
                     ParagraphStyle("sh", parent=S["cellb"], fontSize=10.5, leading=14))
    left = [Paragraph("<b>Put on the slide</b>", ParagraphStyle("l", parent=S["cell"], textColor=INDIGO))]
    left += [Paragraph(i, S["cell"], bulletText="•") for i in on_slide]
    if visual:
        left += [Spacer(1, 4), Paragraph(f"<i>Visual: {visual}</i>", S["small"])]
    right = [Paragraph("<b>What to say</b>", ParagraphStyle("r", parent=S["cell"], textColor=GREEN)),
             Paragraph(f"“{script}”", ParagraphStyle("rs", parent=S["cell"], fontName="Arial-Italic"))]
    t = Table([[head, ""], [left, right]], colWidths=[7.6 * cm, 9 * cm])
    t.setStyle(TableStyle([("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 0), (1, 0), INDIGO),
                           ("BACKGROUND", (1, 1), (1, 1), GREEN_BG), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                           ("LINEBEFORE", (1, 1), (1, 1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return KeepTogether([t, Spacer(1, 10)])


def demo_step(num, title, do, expect, say_text, backup):
    rows = [["Do", do], ["You should see", expect], ["Say", f"<i>“{say_text}”</i>"], ["If it goes wrong", backup]]
    data = [[Paragraph(f"<b>Demo step {num} · {title}</b>", S["cellb"]), ""]]
    data += [[Paragraph(f"<b>{k}</b>", S["cell"]), Paragraph(v, S["cell"])] for k, v in rows]
    t = Table(data, colWidths=[3.3 * cm, 13.3 * cm])
    t.setStyle(TableStyle([("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 0), (1, 0), VIOLET),
                           ("BACKGROUND", (0, 3), (-1, 3), GREEN_BG), ("BACKGROUND", (0, 4), (-1, 4), AMBER_BG),
                           ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return KeepTogether([t, Spacer(1, 9)])


def qa(q, a):
    return KeepTogether([Paragraph(f"<b>Q. {q}</b>", ParagraphStyle("q", parent=S["body"], textColor=INDIGO,
                                                                    spaceAfter=2)),
                         Paragraph(a, S["body"]), Spacer(1, 3)])


story = []

# ---------------------------------------------------------------- cover
story += [Spacer(1, 1.2 * cm),
          Paragraph("How to Present EduTune AI", ParagraphStyle("t", fontName="Arial-Bold", fontSize=30, leading=36,
                                                                 textColor=colors.white)),
          Spacer(1, 8),
          Paragraph("A step-by-step guide for presenting the project to faculty:<br/>slides, what to say, the live "
                    "demo, and how to answer questions",
                    ParagraphStyle("s", fontName="Arial", fontSize=14, leading=19, textColor=colors.HexColor("#ede9fe"))),
          Spacer(1, 3.8 * cm),
          callout("How to use this guide", bullets([
              "Read Sections 1–2 once to understand the <b>story</b> you are telling.",
              "Build your slides from <b>Section 3</b> (one card = one slide) and rehearse the “What to say” lines aloud.",
              "Practise the <b>live demo</b> in Section 4 at least twice on your own laptop.",
              "Section 5 tells you how to describe the dataset <b>honestly</b> — read it carefully.",
              "Before the day, go through the <b>checklist</b> in Section 8.",
          ])),
          Spacer(1, 10),
          table([["The project in one line", ""],
                 ["What", "An AI teaching assistant for AI / ML / DL / GenAI, made by fine-tuning an open-source LLM"],
                 ["How", f"Instruction tuning with QLoRA: 4-bit frozen base model + LoRA adapters ({PCT:.2f}% of parameters trained)"],
                 ["Proof", f"Perplexity {BASE_PPL} → {FT_PPL} (−{RED:.0f}%), side-by-side answers, a working app"]],
                [2.6 * cm, 14 * cm]),
          NextPageTemplate("main"), PageBreak()]

toc = TableOfContents()
toc.levelStyles = [S["toc1"], S["toc2"]]
story += [Paragraph("Contents", S["h1"]), toc, PageBreak()]

# ---------------------------------------------------------------- 1 story
story += [H1("1. The Story You Are Telling"),
          P("Faculty remember a clear story better than a list of technologies. Every slide should support "
            "this single storyline:"),
          table([["Part", "Message"],
                 ["Problem", "General chatbots don't teach in a consistent, structured way, and fine-tuning a whole "
                             "LLM normally needs expensive GPUs."],
                 ["Idea", "Take a small open-source LLM and <b>teach it to be a tutor</b> by training it on examples of good "
                          "teaching answers (instruction tuning)."],
                 ["Trick", "Use <b>QLoRA</b>: shrink the frozen model to 4-bit and train only small LoRA adapters, so it "
                           f"fits on a laptop GPU ({summary['runtime_minutes']:.0f} minutes, "
                           f"{summary['peak_gpu_memory_gb']} GB)."],
                 ["Proof", f"Perplexity dropped by {RED:.0f}%, the model follows teaching formats (e.g. MCQs with answer "
                           "keys), and the base vs fine-tuned difference is visible live."],
                 ["Product", "A working app: tutor chat with 9 teaching modes, quiz, viva examiner, interview coach, "
                             "MCQ generator."],
                 ["Honesty", "What it does not do: it doesn't make the model know more facts, so it can still make "
                             "mistakes."]],
                [2.6 * cm, 14 * cm]),
          Spacer(1, 8),
          callout("Your opening sentence (memorise this)",
                  "“I took a pretrained language model and specialised it into an AI Teaching Assistant using "
                  "instruction tuning. I used PEFT with LoRA, and QLoRA to reduce GPU memory requirements. The model "
                  "was trained on educational instruction-response examples and can now perform different teaching "
                  "tasks such as simple explanations, analogies, viva preparation, interview preparation, MCQs, and "
                  "quizzes.”", bg=GREEN_BG, edge=GREEN),
          H2("1.1 What faculty will be looking for"),
          *bullets([
              "<b>Is it real fine-tuning?</b> Yes — show the training script, the trainable-parameter printout and the adapter file.",
              "<b>Do you understand it?</b> Be able to explain LoRA (h = Wx + (α/r)·BAx) and why 4-bit saves memory.",
              "<b>Did it work?</b> Show numbers (perplexity) <i>and</i> a side-by-side comparison.",
              "<b>Are you honest about limitations?</b> Mention them before they ask — this earns marks.",
              "<b>What was your contribution?</b> Be clear about what you built and how AI tools helped (Section 5).",
          ]),
          H2("1.2 Time plan (15 minutes)"),
          table([["Part", "Slides", "Time"],
                 ["Introduction and problem", "1–3", "2:00"],
                 ["How it works (architecture, dataset, concepts)", "4–7", "4:30"],
                 ["Training and results", "8–9", "2:30"],
                 ["Live demo", "10", "4:00"],
                 ["Limitations and conclusion", "11–12", "1:15"],
                 ["Buffer", "", "0:45"]],
                [10 * cm, 3 * cm, 3.6 * cm]),
          P("If you only have <b>10 minutes</b>: merge slides 2–3, skip slide 6 (explain the concepts while showing slide 7) "
            "and shorten the demo to steps 1, 2 and 4.", "small"),
          PageBreak()]

# ---------------------------------------------------------------- 2 before you build slides
story += [H1("2. Slide Design Tips"),
          *bullets([
              "One idea per slide, at most 5 bullets, and large fonts (at least 24 pt for bullets).",
              "Use the diagrams and charts from <font face='Consolas'>docs/EduTune_AI_Project_Report.pdf</font> "
              "(Figures 1–6) and <font face='Consolas'>results/training_loss.png</font>.",
              "Put the <b>numbers</b> on the slides; put the <b>explanations</b> in what you say.",
              "Show a real dataset record and real outputs — real material is more convincing than descriptions.",
              "Keep code off the slides except the 5-line LoRA config (slide 7). Faculty can ask to see the code.",
              "End with a slide showing the one-sentence summary; leave it on screen during questions.",
          ]),
          callout("Files to have open in tabs before you start", bullets([
              "The app in the browser (<font face='Consolas'>streamlit run app.py</font>), already loaded.",
              "<font face='Consolas'>data/instruction_dataset.jsonl</font> in VS Code (to show real records).",
              "<font face='Consolas'>config.py</font> (LoRA settings) and <font face='Consolas'>training/train_qlora.py</font>.",
              "<font face='Consolas'>results/training_loss.png</font> and <font face='Consolas'>results/training_summary.json</font>.",
              "<font face='Consolas'>docs/EduTune_AI_Project_Report.pdf</font> as a backup of every figure and number.",
          ])),
          PageBreak()]

# ---------------------------------------------------------------- 3 slides
story += [H1("3. Slide-by-Slide Script"),
          P("Each card is one slide: what to put on it (left) and what to say (right). Say it in your own words — "
            "don't read it out.")]
story += [
    slide(1, "Title", "0:30",
          ["EduTune AI — Instruction-Tuned AI Teaching Assistant", "Your name, guide's name, department, date",
           "Subtitle: Supervised fine-tuning with LoRA and QLoRA"],
          "Good morning. My project is EduTune AI — an AI teaching assistant that I created by fine-tuning an "
          "open-source language model to teach AI and machine learning concepts."),
    slide(2, "Problem statement", "1:00",
          ["General LLMs are not optimised for a teaching style", "Answers vary in structure, depth and format",
           "Apps can't rely on their output format (e.g. MCQ answer keys)",
           "Full fine-tuning of a 1.5B model needs ~25 GB of GPU memory"],
          "Chatbots like ChatGPT are general-purpose. When a student asks one to explain LSTM to a beginner, or to "
          "act as a viva examiner, the style changes from question to question. Specialising a model means "
          "fine-tuning it, but full fine-tuning needs data-centre GPUs, which students don't have."),
    slide(3, "Objectives", "0:30",
          ["Build an instruction dataset for AI/ML/DL/GenAI", "Fine-tune with PEFT + LoRA, using QLoRA (4-bit)",
           "Run everything on a laptop GPU", "Compare base vs fine-tuned honestly", "Build a teaching application"],
          "My objectives were to create a teaching dataset, actually fine-tune a model — not just write prompts — "
          "on my own laptop, measure whether it improved, and wrap it in a usable application."),
    slide(4, "Architecture", "1:00",
          ["Knowledge base → generator → 840 examples", "Cleaning + 90/10 split",
           "4-bit base model + LoRA → QLoRA training", "Adapter → evaluation → Streamlit app"],
          "The project is a pipeline. I start with structured educational content, turn it into instruction-response "
          "examples, clean and split them, and train LoRA adapters on a 4-bit frozen model. The trained adapter is "
          "evaluated and then used by the teaching app.",
          "Figure 3 (pipeline diagram) from the project report"),
    slide(5, "The instruction dataset", "1:30",
          ["840 examples, 37 topics, 14 categories", "Format: instruction → response (+ optional input)",
           "Categories: definitions, analogies, ELI10, viva, interview, MCQ, numerical…",
           "756 training / 84 validation (stratified)"],
          "Instruction tuning needs examples of the behaviour we want. Each record is an instruction and an ideal "
          "response. The content was written as structured notes for 37 topics, and a generator script turned each "
          "note into different teaching tasks — definitions, analogies, viva feedback, MCQs. Here is a real record.",
          "one real JSON record from data/instruction_dataset.jsonl + the category table"),
    slide(6, "Key concepts", "1:00",
          ["Fine-tuning: adapt a pretrained model", "Instruction tuning: instruction → response pairs",
           "SFT: learn from desired outputs", "PEFT: train only a few parameters"],
          "Fine-tuning adapts a pretrained model to a task. Instruction tuning is fine-tuning on instruction–response "
          "pairs. It is supervised because every example contains the desired answer. And PEFT means we don't update "
          "all the weights — only a small number of extra parameters."),
    slide(7, "LoRA and QLoRA", "1:30",
          ["LoRA: h = Wx + (α/r)·BAx — W frozen, A and B trained",
           f"r = 16, α = 32 → {TP / 1e6:.1f} M trainable of {TOT / 1e9:.2f} B ({PCT:.2f}%)",
           "QLoRA: base model stored in 4-bit NF4 (~1 GB instead of ~3 GB)",
           "Adapters trained in bf16; paged 8-bit optimiser"],
          "LoRA keeps the original weight matrix frozen and learns two small matrices, A and B, whose product is the "
          f"update. With rank 16 that is only {PCT:.2f} percent of the parameters. QLoRA additionally stores the frozen "
          "model in 4-bit, which is about four times smaller, so everything fits on my 8 GB laptop GPU.",
          "Figure 1 (LoRA diagram) and Figure 2 (memory chart)"),
    slide(8, "Training", "1:00",
          ["Qwen2.5-1.5B-Instruct, 3 epochs, LR 2e-4, effective batch 8",
           f"{summary['runtime_minutes']:.1f} minutes, peak {summary['peak_gpu_memory_gb']} GB GPU memory",
           "Loss computed on responses only", f"Output: {summary['adapter_size_mb']:.0f} MB adapter (model is ~3 GB)"],
          "I trained with Hugging Face TRL's SFTTrainer. Training took about seventeen minutes and never used more "
          "than three gigabytes of GPU memory. The loss curve shows validation loss dropping and then levelling off "
          "around step 175 — so two epochs would have been enough.",
          "results/training_loss.png"),
    slide(9, "Results", "1:30",
          [f"Validation perplexity: {BASE_PPL} → {FT_PPL} (−{RED:.0f}%)",
           "MCQ format: 2/2 parseable vs 0/2 for the base model",
           "Consistent teaching structure, also on unseen topics",
           "Not always better — shorter, can still make mistakes"],
          "On held-out examples, perplexity fell by seventy-one percent — the model learned the target teaching "
          "style. It also reliably follows formats: asked for MCQs, it gives answer keys; the base model doesn't. "
          "I don't claim it is better at everything: its answers are shorter, and it can still make factual mistakes.",
          "Figure 6 (per-category chart) + one side-by-side example"),
    slide(10, "Live demo", "4:00", ["(switch to the app — see Section 4)"],
          "Now let me show you the system working live."),
    slide(11, "Limitations and future work", "0:45",
          ["Split by example, not topic — validation overlaps training topics",
           "Instruction tuning changes style, not knowledge", "No human evaluation yet",
           "Future: topic-level split, DPO, RAG over textbooks, bigger model"],
          "The main limitations: validation examples share topics with training, so the metrics show style learning "
          "rather than new knowledge, and the model can still be wrong. Next steps are a stricter split, preference "
          "tuning with DPO, and retrieval from textbooks to improve accuracy."),
    slide(12, "Conclusion", "0:30",
          ["The one-sentence summary", "Thank you — questions?"],
          "To conclude: I took a pretrained language model and specialised it into an AI teaching assistant using "
          "instruction tuning with LoRA and QLoRA, trained on a laptop in under twenty minutes. Thank you — I'm happy "
          "to take questions."),
]
story += [PageBreak()]

# ---------------------------------------------------------------- 4 demo
story += [H1("4. The Live Demo (4 minutes)"),
          P("The demo is where you score the most marks, and also where things can go wrong. Practise it until you "
            "can do it without thinking. Start the app <b>at least 2 minutes before</b> you present, because loading "
            "the model takes about 15 seconds on first use."),
          code("venv\\Scripts\\activate\nstreamlit run app.py        # opens http://localhost:8501"),
          demo_step(1, "Show the dataset (20 s)",
                    "Switch to VS Code and show <font face='Consolas'>data/instruction_dataset.jsonl</font>. Point at one line.",
                    "One JSON record per line with instruction, input, output and category.",
                    "This is the instruction dataset the model was trained on — 840 examples like this one.",
                    "Show the example record on your dataset slide instead."),
          demo_step(2, "Base vs fine-tuned (90 s) — the most important step",
                    "App → <b>Base vs Fine-Tuned</b>. Type <i>Explain RNN to a beginner.</i>, keep “Deterministic "
                    "decoding” ticked, click <b>Compare</b>.",
                    "Two columns: BASE MODEL (longer, conversational) and FINE-TUNED MODEL (structured, with a "
                    "“Key idea to remember” line). Below them: a list of observable differences.",
                    "Both answers come from the same model and the same prompt. The only difference is my 46 MB "
                    "adapter, which is switched off for the left column. You can see the learned teaching structure on the right.",
                    "Scroll down to “Saved evaluation results” and pick a saved question — these were generated "
                    "earlier and always work."),
          demo_step(3, "Explain Like I'm 10 (40 s)",
                    "App → <b>AI Tutor</b>. Sidebar teaching mode: <b>Explain Like I'm 10</b>. Ask <i>What is an LSTM?</i>",
                    "A short analogy-based answer ending with “The grown-up version: …”.",
                    "The teaching mode changes the instruction sent to the model — and the model was trained on "
                    "exactly this instruction format.",
                    "Use the quick action button “🧠 Give Analogy” instead."),
          demo_step(4, "Viva mode (60 s)",
                    "App → <b>Viva Mode</b>. Choose a topic (e.g. LSTM) in the sidebar. Click <b>Ask me a question</b>, "
                    "type a deliberately partial answer, and click <b>Submit answer</b>.",
                    "The examiner's question, then feedback with a verdict, what you got right and a model answer.",
                    "The model learned to act as an examiner from feedback examples in the dataset, where the student's "
                    "answer is given in the input field.",
                    "If the question is odd, click the button again for a new one."),
          demo_step(5, "Quiz / MCQ (40 s)",
                    "App → <b>Quiz Mode</b>, click <b>New Question</b>, pick an answer, click <b>Submit</b>. Optionally: "
                    "<b>MCQ Generator</b> → RNN, 5 questions, Medium.",
                    "A question with options A–D, then “Correct!” or the right answer with an explanation, and the score.",
                    "The fine-tuned model writes MCQs in a fixed format with an answer key, so the app can grade them. "
                    "The base model doesn't reliably do that.",
                    "If a generated question looks wrong, switch the question source to <b>Curated question bank</b> and "
                    "say: “model-generated questions can contain errors — that's a known limitation.”"),
          demo_step(6, "Training evidence (20 s)",
                    "Show <font face='Consolas'>config.py</font> (LoRA settings) and "
                    "<font face='Consolas'>results/training_loss.png</font>.",
                    f"LORA_R = 16, LORA_ALPHA = 32, USE_4BIT = True; the loss curve.",
                    f"These are the actual settings: rank 16 on all attention and MLP layers, {PCT:.2f} percent trainable.",
                    "Use the figures on your slides."),
          callout("Golden rules for the demo", bullets([
              "Use <b>Base vs Fine-Tuned with deterministic decoding</b> for the comparison — the result is repeatable.",
              "<b>Never re-run training</b> during the presentation; it takes 17 minutes and overwrites the adapter.",
              "Close other GPU-heavy programs (games, video editors) before starting.",
              "If the app fails completely, don't debug in front of the faculty. Say “I'll show the saved results”, "
              "and use the project report figures and <font face='Consolas'>results/comparison.json</font>.",
              "Take your own screenshots of every demo step the day before, as a last-resort backup.",
          ]), bg=AMBER_BG, edge=AMBER),
          PageBreak()]

# ---------------------------------------------------------------- 5 honesty
story += [H1("5. Talking About the Dataset and AI Assistance Honestly"),
          P("Faculty often ask <b>where the training data came from</b> and <b>what you did yourself</b>. Answer "
            "accurately — it protects you and it is expected in academic work."),
          H2("5.1 The facts"),
          *bullets([
              "The educational content (definitions, analogies, viva and interview Q&amp;A, MCQs, numerical problems) is "
              "written as structured notes in <font face='Consolas'>scripts/knowledge/*.py</font>.",
              "Those notes were <b>written with the help of an AI coding assistant (Claude)</b>, based on standard "
              "textbook material and the original papers (e.g. LSTM 1997, Transformer 2017, LoRA 2021, QLoRA 2023).",
              "A generator script converts the notes into 840 training examples. It does not look anything up; it "
              "only rearranges the written notes into different task formats.",
              "The model was then genuinely fine-tuned on that data on your laptop, and the results are real measurements.",
          ]),
          H2("5.2 How to say it"),
          callout("Suggested wording",
                  "“The training content was written as a structured knowledge base for 37 topics, with the help of an "
                  "AI assistant, based on standard textbook material, and I reviewed it. A generator script then "
                  "converted it into 840 instruction–response examples in 14 teaching styles. The fine-tuning, "
                  "evaluation and application were run on my own laptop, and all the numbers I show come from those runs.”",
                  bg=GREEN_BG, edge=GREEN),
          callout("Important — do this before the presentation",
                  "Only say “I reviewed it” if you actually have. Before the day, read through the topics you are most "
                  "likely to be asked about (RNN, LSTM, CNN, overfitting, LoRA, QLoRA, instruction tuning) in "
                  "<font face='Consolas'>scripts/knowledge/</font> and compare them with your textbook or lecture notes. "
                  "If your institution has rules about declaring AI assistance, follow them — mention it in your report "
                  "and slides as well.", bg=AMBER_BG, edge=AMBER),
          H2("5.3 Things NOT to claim"),
          table([["Don't say", "Why", "Say instead"],
                 ["“The data was collected from the internet.”", "It wasn't — it was written as structured notes.",
                  "“It was written as a structured knowledge base and converted by a generator.”"],
                 ["“The model is always correct.”", "It isn't; instruction tuning changes style, not knowledge.",
                  "“It follows the teaching style reliably, but it can still make mistakes.”"],
                 ["“It is better than ChatGPT.”", "Never measured, and not true for general knowledge.",
                  "“It is specialised for a teaching style that runs locally on a laptop.”"],
                 ["“Accuracy is 97%.”", "Accuracy is not a valid metric for free-text answers.",
                  f"“Perplexity dropped from {BASE_PPL} to {FT_PPL}.”"],
                 ["“I trained a model from scratch.”", "The base model was pre-trained by the Qwen team.",
                  f"“I fine-tuned a pretrained model, training {PCT:.2f}% of its parameters.”"]],
                [5 * cm, 5 * cm, 6.6 * cm]),
          PageBreak()]

# ---------------------------------------------------------------- 6 faculty questions
story += [H1("6. Questions Faculty Are Likely to Ask"),
          P("Short, confident answers are best. If you don't know something, say so and explain how you would find out.")]
FQ = [
    ("Is this just ChatGPT with a prompt?",
     "No. No external API is used. The model runs locally, and I changed its behaviour by training: "
     f"{TP:,} LoRA parameters were learned from my dataset. You can see the adapter file and the training logs."),
    ("What exactly did you train?",
     f"Only the LoRA adapter matrices — {PCT:.2f}% of the parameters — on the attention and MLP layers of all 28 "
     "layers. The 1.5-billion-parameter base model stayed frozen in 4-bit."),
    ("Where did the data come from?",
     "It was written as structured notes for 37 topics with AI assistance, based on textbook material, and a "
     "generator converted it into 840 examples in 14 categories (see Section 5)."),
    ("Why only 840 examples? Is that enough?",
     "Instruction tuning is about teaching behaviour and format, and research such as the LIMA paper shows that "
     "a few hundred to a few thousand high-quality examples are enough for that. The loss curve plateaued, which "
     "suggests more epochs wouldn't help; more diverse data would."),
    ("How do you know the model improved?",
     f"Validation perplexity on held-out answers fell from {BASE_PPL} to {FT_PPL}, and in side-by-side tests the "
     "fine-tuned model follows the trained formats — for example 2 out of 2 MCQs with answer keys versus 0 for the base model."),
    ("Why didn't you use accuracy?",
     "Many different explanations are correct, so exact-match accuracy is meaningless for free text. Perplexity "
     "measures how well the model predicts reference answers, and I compared answers qualitatively as well."),
    ("Did the model overfit?",
     "Validation loss plateaued from about step 175 while training loss kept falling — the start of memorisation — "
     "but validation loss never went up. Early stopping at 2 epochs would give the same result."),
    ("What is the difference between LoRA and QLoRA?",
     "Both train small adapters on a frozen model. QLoRA stores the frozen model in 4-bit NF4 instead of 16-bit, "
     "cutting its memory about four times."),
    ("Why does 4-bit not destroy accuracy?",
     "The weights are only stored in 4-bit; they are converted back to 16-bit for every calculation. They are "
     "frozen, so errors don't accumulate, and the adapters can compensate."),
    ("Why Qwen2.5-1.5B?",
     "It is open-source, instruction-capable, strong for its size, and fits QLoRA training on an 8 GB laptop GPU. "
     "The model name is one setting in config.py, so it can be swapped."),
    ("What was your contribution?",
     "Designing the pipeline, preparing and reviewing the dataset, configuring and running the QLoRA training, "
     "evaluating base versus fine-tuned, and building the teaching application. (Adjust this to what you personally did.)"),
    ("What is novel here?",
     "The techniques are established; the contribution is applying them end-to-end to a specific educational "
     "use case, on consumer hardware, with an honest evaluation and a usable application."),
    ("What happens when the model is wrong?",
     "The app shows a disclaimer to verify with textbooks, the quiz has a curated question bank, and future work "
     "includes retrieval from trusted textbooks (RAG) to ground answers."),
    ("Could this work with a bigger model?",
     "Yes — change MODEL_NAME in config.py (e.g. Qwen2.5-3B) and retrain; QLoRA makes 3B feasible on 8 GB."),
    ("How long would full fine-tuning take instead?",
     "Full fine-tuning needs about 16 bytes per parameter — roughly 25 GB for this model — so it would not even fit "
     "on this GPU. That is exactly why PEFT and QLoRA are used."),
]
story += [qa(q, a) for q, a in FQ]
story += [PageBreak()]

# ---------------------------------------------------------------- 7 dos and donts
story += [H1("7. Presentation Do's and Don'ts"),
          table([["Do", "Don't"],
                 ["Start with the problem and your one-sentence summary", "Start by listing libraries and versions"],
                 ["Show real outputs and real numbers", "Describe results you can't show"],
                 ["Explain LoRA with the diagram and one equation", "Put long code on slides"],
                 ["Mention limitations yourself", "Wait for faculty to find them"],
                 ["Say “the model can make mistakes”", "Claim the model is always correct"],
                 ["Keep the demo to the rehearsed steps", "Improvise new questions live"],
                 ["Pause after key numbers (1.18%, −71%)", "Rush through the results slide"],
                 ["Say “I don't know, but I would check…” when unsure", "Guess confidently"],
                 ["Be clear about what AI tools helped with", "Present AI-assisted work as entirely your own"]],
                [8.3 * cm, 8.3 * cm]),
          H1("8. Checklist"),
          H2("8.1 The day before"),
          *bullets([
              "Rehearse the full presentation with a timer, at least twice.",
              "Run <font face='Consolas'>python scripts/test_model.py</font> — all 8 tests should PASS.",
              "Run the demo steps once and take screenshots of each as a backup.",
              "Review the knowledge-base content for the main topics (Section 5.2).",
              "Charge the laptop fully; the GPU is slower on battery power.",
          ]),
          H2("8.2 Thirty minutes before"),
          *bullets([
              "Plug in the charger and close other programs, especially anything using the GPU.",
              "Start the app: <font face='Consolas'>streamlit run app.py</font>, and wait until a first question has been answered.",
              "Open the tabs listed in Section 2 (dataset, config.py, loss curve, report PDF).",
              "Increase the browser zoom (Ctrl +) so the faculty can read the answers.",
              "If the room has no internet, the app still works — the model is already downloaded. If it tries to "
              "connect, run <font face='Consolas'>$env:HF_HUB_OFFLINE=1</font> (PowerShell) or "
              "<font face='Consolas'>set HF_HUB_OFFLINE=1</font> (Command Prompt) before starting it.",
          ]),
          H2("8.3 Numbers to remember"),
          table([["What", "Number"],
                 ["Base model", "Qwen2.5-1.5B-Instruct (1.54 B parameters, 28 layers)"],
                 ["Trainable parameters", f"{TP:,} ({PCT:.2f}%)"],
                 ["LoRA", "r = 16, α = 32, dropout 0.05, 7 target modules"],
                 ["Quantisation", "4-bit NF4 + double quantisation, bf16 compute"],
                 ["Dataset", "840 examples · 37 topics · 14 categories · 756 / 84 split"],
                 ["Training", f"3 epochs · 285 steps · LR 2e-4 · {summary['runtime_minutes']:.1f} min · "
                              f"{summary['peak_gpu_memory_gb']} GB peak"],
                 ["Result", f"Perplexity {BASE_PPL} → {FT_PPL} (−{RED:.0f}%)"],
                 ["Adapter", f"{summary['adapter_size_mb']:.0f} MB (full model ~3 GB)"]],
                [4.5 * cm, 12.1 * cm]),
          Spacer(1, 10),
          callout("Last line before questions",
                  "“To summarise: a pretrained model, specialised into a teaching assistant with instruction tuning, "
                  f"training only {PCT:.2f} percent of its parameters, on a laptop, in under twenty minutes. Thank you.”",
                  bg=GREEN_BG, edge=GREEN)]

out = ROOT / "docs" / "EduTune_AI_Faculty_Presentation_Guide.pdf"
Doc(str(out)).multiBuild(story)
print("Wrote", out)
