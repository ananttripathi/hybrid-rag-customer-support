"""Generates Project_Proposal_and_Methodology.pdf (Stage 1 deliverable)."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem, PageBreak
)
from reportlab.lib.enums import TA_CENTER

OUT = "Project_Proposal_and_Methodology.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleMain", fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=6, textColor=colors.HexColor("#1a1a2e")))
styles.add(ParagraphStyle(name="Subtitle", fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=20))
styles.add(ParagraphStyle(name="H1", fontSize=15, leading=18, spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#0b3d91")))
styles.add(ParagraphStyle(name="H2", fontSize=12.5, leading=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#16324f")))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14.5, spaceAfter=6))
styles.add(ParagraphStyle(name="BulletTxt", fontSize=10, leading=14, spaceAfter=3))
styles.add(ParagraphStyle(name="Mono", fontName="Courier", fontSize=9, leading=12, backColor=colors.HexColor("#f4f4f4"), spaceAfter=6))

doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)
story = []

def h1(t): story.append(Paragraph(t, styles["H1"]))
def h2(t): story.append(Paragraph(t, styles["H2"]))
def p(t): story.append(Paragraph(t, styles["Body"]))
def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, styles["BulletTxt"])) for i in items], bulletType="bullet", leftIndent=14))
def mono(t): story.append(Paragraph(t.replace("\n", "<br/>"), styles["Mono"]))

story.append(Paragraph("Project Proposal &amp; Methodology", styles["TitleMain"]))
story.append(Paragraph("Hybrid RAG &amp; Fine-Tuning for Customer Support", styles["Subtitle"]))

# ---------------- Task 1.1.1 ----------------
h1("1. Project Scope")
h2("1.1.1 Target Use Case")
p("<b>Domain:</b> Automated Tier-1 customer support for the <b>post-purchase / order-operations</b> "
  "category of an e-commerce business. This scope was chosen because it is the single largest, "
  "most repetitive slice of inbound support volume, and it maps directly onto the corporate SOP "
  "corpus provided (refund, shipping, returns, account/password recovery, billing, subscriptions, "
  "escalation, order tracking, payment methods, data privacy, and working hours).")
p("<b>In scope intents</b> (drawn from the Bitext categories present in the sampled data): order "
  "status/tracking, cancellations, refunds, returns, shipping delays, password/account recovery, "
  "billing disputes, payment-method changes, subscription cancellation, and complaint escalation.")
p("<b>Out of scope:</b> pre-sales / product-discovery conversations, marketing content, and any "
  "action that mutates account or payment state (the system only classifies intent and drafts a "
  "policy-grounded response — it never executes a refund, cancellation, or account change itself).")

h2("Expected JSON Interface")
p("<b>Input</b> to the fine-tuned intent router (Stage 4):")
mono('{"query": "the raw, possibly noisy customer message"}')
p("<b>Output</b> of the intent router — the structured target the model is fine-tuned to emit:")
mono('{"intent": "track_order", "category": "SHIPPING"}')
p("This structured output is what drives the Hybrid RAG retrieval step (Task 4.3): the intent is "
  "mapped to a specific SOP search string instead of the raw, noisy query.")

h2("1.1.2 Success Criteria")
p("Quantitative targets, defined <i>before</i> development began, evaluated on the held-out test "
  "split described in Stage 2:")
data = [
    ["Metric", "Baseline (expected)", "Target for Hybrid RAG (V2)"],
    ["Format Adherence Rate", "n/a (free text)", "≥ 90% valid JSON"],
    ["Intent Accuracy (exact match)", "n/a", "≥ 75% on test split, ≥ 60% on adversarial subset"],
    ["ROUGE-1 / ROUGE-L", "low (ungrounded)", "Improve over Baseline and V1 (Naive RAG)"],
    ["BLEU", "low (ungrounded)", "Improve over Baseline and V1 (Naive RAG)"],
    ["Consistency Rate", "100% (greedy)", "100% (greedy, do_sample=False)"],
    ["Hallucination Frequency", "high (unconstrained)", "≥ 40% relative reduction vs Baseline"],
]
t = Table(data, colWidths=[5.2*cm, 4.4*cm, 5.4*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b3d91")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTSIZE", (0,0), (-1,-1), 8.5),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f7fb")]),
    ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(t)
story.append(Spacer(1, 10))
p("These targets are re-measured identically for Baseline, Solution V1, and Solution V2 in the "
  "Comparative Analysis Report, so improvement is attributable and quantified stage-by-stage.")

story.append(PageBreak())

# ---------------- Task 1.4 ----------------
h1("2. Proposed Methodology")

h2("1.4.1 Baseline Model Selection")
p("<b>Selected model: <font face='Courier'>Qwen/Qwen2.5-1.5B-Instruct</font></b>.")
bullets([
    "<b>Hardware constraints:</b> the assignment targets a free-tier T4 GPU (Colab); this project was "
    "additionally executed end-to-end locally on Apple Silicon (M-series, MPS backend, no CUDA). At 1.5B "
    "parameters the model loads comfortably in fp16 on either environment without requiring the "
    "8B-class model's >16GB VRAM footprint.",
    "<b>Dataset characteristics:</b> the Bitext-derived instruction set (~1,500 rows after cleaning) is "
    "small; a 1.5B model has enough capacity to learn the JSON-extraction task via LoRA without the "
    "severe overfitting risk a larger model would carry on so little data.",
    "<b>Fine-tuning feasibility:</b> Qwen2.5-Instruct ships with a native ChatML template "
    "(<font face='Courier'>apply_chat_template</font>) and strong out-of-the-box structured-output "
    "behaviour, which materially improves LoRA sample-efficiency for the JSON-router task.",
    "<b>Rejected alternatives:</b> TinyLlama-1.1B-Chat is lighter but empirically weaker at emitting "
    "strict JSON; Llama-3-8B-Instruct gives the best raw quality but its memory footprint (>16GB even "
    "4-bit) is infeasible on the constrained hardware used here.",
    "<b>Hardware adaptation note:</b> because execution is on Apple Silicon rather than a CUDA GPU, "
    "<font face='Courier'>bitsandbytes</font> 4-bit quantisation (CUDA-only) is unavailable. The model "
    "is instead loaded in fp16 directly on the MPS device — functionally equivalent for this project's "
    "purpose of demonstrating the RAG + PEFT pipeline, and explicitly documented as a deviation from "
    "the Colab/T4 reference environment.",
])

h2("1.4.2 Retrieval Strategy")
bullets([
    "<b>Embedding model:</b> <font face='Courier'>sentence-transformers/all-MiniLM-L6-v2</font> "
    "(384-dim, CPU-friendly, ~80MB) — used identically across NB4, NB5, and NB7 to keep retrieval "
    "results comparable across evaluation stages.",
    "<b>Chunking:</b> each corporate SOP Markdown file is treated as a single retrieval unit (documents "
    "are short, 150-250 words each, single-topic) — no further sub-chunking is required at this corpus size.",
    "<b>Vector storage:</b> ChromaDB, persisted to <font face='Courier'>./chroma_db</font>, one collection "
    "for the full SOP corpus.",
    "<b>Similarity search:</b> naive RAG (V1) embeds the raw customer query and retrieves the top-1 "
    "nearest document (<font face='Courier'>k=1</font>). Hybrid RAG (V2) instead searches using the "
    "structured intent string extracted by the fine-tuned router, which is far less noisy than the raw query.",
    "<b>Retrieval evaluation:</b> validated directly (does the retrieved document match the query's true "
    "policy area?) and indirectly, via the downstream ROUGE/BLEU and hallucination-frequency scores of "
    "the generated answer.",
])

h2("1.4.3 Fine-Tuning Strategy")
bullets([
    "<b>Technique:</b> LoRA (not QLoRA) via <font face='Courier'>peft</font> — QLoRA's 4-bit quantisation "
    "path depends on <font face='Courier'>bitsandbytes</font>, which requires CUDA and is unavailable on "
    "the Apple Silicon execution environment used here.",
    "<b>Target modules:</b> <font face='Courier'>q_proj</font>, <font face='Courier'>v_proj</font> "
    "(attention projections) — updates ~0.5-1% of total parameters.",
    "<b>Adapter config:</b> rank <font face='Courier'>r=16</font>, "
    "<font face='Courier'>lora_alpha=32</font> (2×r), <font face='Courier'>lora_dropout=0.05</font>, "
    "<font face='Courier'>task_type=CAUSAL_LM</font>.",
    "<b>ChatML structure:</b> system + user (raw query) + assistant, where the assistant turn is forced "
    "to a strict JSON string <font face='Courier'>{\"intent\": ..., \"category\": ...}</font> generated "
    "via <font face='Courier'>json.dumps</font>, with <font face='Courier'>-100</font> label-masking "
    "applied to padding tokens so loss is computed only on real content.",
    "<b>Training loop:</b> custom PyTorch loop with AdamW (<font face='Courier'>lr=2e-4</font>), "
    "batch size 4, validation every <font face='Courier'>VAL_EVERY</font> steps, early stopping on "
    "validation loss (<font face='Courier'>PATIENCE=3</font>).",
    "<b>Checkpointing / logging:</b> best-val-loss checkpoint saved to "
    "<font face='Courier'>./intent_lora_best/</font>, final checkpoint to "
    "<font face='Courier'>./intent_lora/</font>, plus a <font face='Courier'>training_log.csv</font> "
    "and loss-curve plot as reproducibility evidence.",
])

h2("1.4.4 Evaluation Framework")
bullets([
    "<b>Format Adherence Rate:</b> percentage of router outputs that parse successfully with "
    "<font face='Courier'>json.loads()</font>.",
    "<b>Intent Accuracy:</b> exact-match rate of the extracted <font face='Courier'>intent</font> field "
    "against ground truth, plus a fuzzy-match variant (e.g. <font face='Courier'>order_tracking</font> "
    "vs <font face='Courier'>track_order</font>) to capture near-misses.",
    "<b>ROUGE-1 / ROUGE-L / BLEU:</b> computed against SOP-grounded reference answers (not the generic "
    "Bitext <font face='Courier'>response</font> column), so policy-specific language is rewarded.",
    "<b>Consistency Rate:</b> repeated greedy-decoded generations "
    "(<font face='Courier'>do_sample=False, temperature=None</font>) for the same input must be identical.",
    "<b>Hallucination Frequency:</b> percentage of generations containing claims (dates, numbers, policy "
    "terms) not supported by the retrieved SOP context, checked for Baseline, V1, and V2 identically.",
    "All five metrics are re-computed under an identical configuration (same <font face='Courier'>MODEL_ID</font>, "
    "same embedding model, same test split) across all three system versions so the deltas are attributable.",
])

doc.build(story)
print("Wrote", OUT)
