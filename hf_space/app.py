"""Hybrid RAG & Fine-Tuning for Customer Support — live demo.

Runs all three architectures from the capstone side by side on one query:
  1. Baseline   — zero-shot Qwen2.5-1.5B-Instruct, no context.
  2. Naive RAG  — raw query embedded, top-1 SOP retrieved, answer grounded in it.
  3. Hybrid RAG — LoRA-fine-tuned intent router extracts structured JSON intent,
                  intent drives the SOP search string (not the noisy raw query),
                  same base model generates the grounded answer.
"""
import json
import re
import glob
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SOP_DIR = "Dataset/sop_documents"
LORA_DIR = "intent_lora_best"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

GEN_KWARGS = dict(max_new_tokens=90, do_sample=False, temperature=None, top_p=None)
ROUTER_GEN_KWARGS = dict(max_new_tokens=30, do_sample=False, temperature=None, top_p=None)

ROUTER_SYSTEM_PROMPT = (
    "You are a customer support intent router. Given a customer message, respond with a single "
    "JSON object containing exactly two fields: \"intent\" and \"category\". Output JSON only, "
    "with no explanation."
)
FEWSHOT_EXAMPLES = [
    ("where is my order #12345", '{"intent": "track_order", "category": "SHIPPING"}'),
    ("i want a refund for my last purchase", '{"intent": "get_refund", "category": "REFUND"}'),
    ("i forgot my password and cant log in", '{"intent": "recover_password", "category": "ACCOUNT"}'),
    ("can I cancel my subscription please", '{"intent": "newsletter_subscription", "category": "SUBSCRIPTION"}'),
]
INTENT_TO_SEARCH_STRING = {
    "track_order": "order tracking status", "cancel_order": "order cancellation policy",
    "change_order": "order cancellation policy", "get_refund": "refund policy",
    "check_refund_policy": "refund policy", "track_refund": "refund policy",
    "delivery_period": "shipping delays", "delivery_options": "shipping delays",
    "change_shipping_address": "shipping delays", "set_up_shipping_address": "shipping delays",
    "recover_password": "password reset account recovery", "registration_problems": "account recovery",
    "create_account": "account recovery", "delete_account": "data privacy account deletion",
    "edit_account": "account recovery", "switch_account": "account recovery",
    "check_invoice": "billing disputes", "get_invoice": "billing disputes",
    "payment_issue": "payment methods", "check_payment_methods": "payment methods",
    "newsletter_subscription": "subscription cancellation", "complaint": "escalation matrix",
    "contact_customer_service": "escalation matrix", "contact_human_agent": "escalation matrix",
    "review": "escalation matrix", "place_order": "order tracking status",
}

print("Loading tokenizer + models (first launch takes ~1-2 min)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
base_model.eval()

router_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
router_model = PeftModel.from_pretrained(router_base, LORA_DIR).merge_and_unload()
router_model.eval()

print("Building retrieval index from SOP documents...")
sop_files = sorted(glob.glob(f"{SOP_DIR}/*.md"))
docs = []
for f in sop_files:
    loaded = TextLoader(f, encoding="utf-8").load()
    for d in loaded:
        d.metadata["source_file"] = f.split("/")[-1]
    docs.extend(loaded)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = Chroma.from_documents(docs, embeddings)
print(f"Ready. {len(docs)} SOP documents indexed.")


def generate_baseline(query):
    messages = [{"role": "system", "content": "You are a helpful customer support assistant."},
                {"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = base_model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **GEN_KWARGS)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_with_context(query, context):
    system_prompt = (
        f"You are a customer support assistant. Answer strictly using this SOP:\n\n{context}\n\n"
        "If the SOP does not cover the question, say you will escalate to a human agent."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = base_model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **GEN_KWARGS)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_naive_rag(query, k=1):
    retrieved = vector_db.similarity_search(query, k=k)
    answer = generate_with_context(query, "\n\n".join(d.page_content for d in retrieved))
    return answer, retrieved[0].metadata["source_file"]


def extract_json(raw_text):
    try:
        return json.loads(raw_text.strip())
    except Exception:
        match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def run_router(query):
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
    for q, a in FEWSHOT_EXAMPLES:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = router_model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **ROUTER_GEN_KWARGS)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return raw, extract_json(raw)


def generate_hybrid_rag(query, k=1):
    raw_router_output, parsed = run_router(query)
    if parsed and "intent" in parsed:
        intent = parsed["intent"]
        search_string = INTENT_TO_SEARCH_STRING.get(intent, intent.replace("_", " "))
    else:
        intent, search_string = None, query
    retrieved = vector_db.similarity_search(search_string, k=k)
    answer = generate_with_context(query, "\n\n".join(d.page_content for d in retrieved))
    return answer, retrieved[0].metadata["source_file"], raw_router_output, intent, search_string


def compare_all(query):
    if not query or not query.strip():
        return "Enter a query above.", "Enter a query above.", "Enter a query above."

    baseline_out = generate_baseline(query)

    rag_out, rag_doc = generate_naive_rag(query, k=1)
    rag_display = f"**Retrieved:** `{rag_doc}`\n\n{rag_out}"

    hybrid_out, hybrid_doc, router_raw, intent, search_string = generate_hybrid_rag(query, k=1)
    hybrid_display = (
        f"**Router output:** `{router_raw}`\n\n"
        f"**Search string used:** `{search_string}`  |  **Retrieved:** `{hybrid_doc}`\n\n"
        f"{hybrid_out}"
    )

    return baseline_out, rag_display, hybrid_display


EXAMPLES = [
    "my package is still not here and its been forever, this is ridiculous, where even is it??",
    "i want my money back for the thing i bought last week",
    "cant log into my account, forgot the password",
    "how do i stop getting charged every month for this subscription",
]

with gr.Blocks(title="Hybrid RAG Customer Support") as demo:
    gr.Markdown(
        "# Hybrid RAG & Fine-Tuning for Customer Support\n"
        "Compare all 3 architectures from the capstone on the same query, side by side. "
        f"Model: `{MODEL_ID}` · Router: LoRA-fine-tuned intent extractor · Retrieval: ChromaDB + MiniLM.\n\n"
        "CPU inference on the free tier — each query takes ~20-40s."
    )
    query_box = gr.Textbox(label="Customer query", placeholder="Type a support message...", lines=2)
    gr.Examples(examples=EXAMPLES, inputs=query_box)
    run_btn = gr.Button("Compare all 3 systems", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Baseline (no context)")
            baseline_box = gr.Markdown()
        with gr.Column():
            gr.Markdown("### 2. Naive RAG (raw-query retrieval)")
            rag_box = gr.Markdown()
        with gr.Column():
            gr.Markdown("### 3. Hybrid RAG (fine-tuned router)")
            hybrid_box = gr.Markdown()

    run_btn.click(compare_all, inputs=query_box, outputs=[baseline_box, rag_box, hybrid_box])

if __name__ == "__main__":
    demo.launch()
