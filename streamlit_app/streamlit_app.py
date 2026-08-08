"""Hybrid RAG & Fine-Tuning for Customer Support — live demo (Streamlit Community Cloud).

Runs all three architectures from the capstone side by side on one query:
  1. Baseline   — zero-shot base model, no context.
  2. Naive RAG  — raw query embedded, top-1 SOP retrieved, answer grounded in it.
  3. Hybrid RAG — LoRA-fine-tuned intent router extracts structured JSON intent,
                  intent drives the SOP search string (not the noisy raw query),
                  same base model generates the grounded answer.

TWO DEPARTURES FROM THE GRADED CAPSTONE, both purely to fit Streamlit Community Cloud's free
1GB RAM cap (the capstone notebooks are unaffected and use the real setup):

1. Smaller model: the capstone fine-tunes/evaluates Qwen2.5-1.5B-Instruct end-to-end (see
   Fine_Tuning_Pipeline.ipynb). Its ~152K-token vocab alone needs ~1GB just for the embedding
   table. This demo uses HuggingFaceTB/SmolLM2-135M-Instruct (49K vocab) with its own LoRA
   router (scripts/train_demo_router.py — same cleaning/split/ChatML/LoRA recipe, different
   base model), and a single shared model (adapter toggled via disable_adapter()) rather than
   two full copies.

2. No ChromaDB/LangChain/sentence-transformers: those together cost ~350-400MB just to import,
   dominated by sentence-transformers' own dependency tree. For a 13-document corpus this is
   pure overhead, so retrieval here is plain `transformers` mean-pooling + numpy cosine
   similarity — same embedding model, same results, far less memory. The capstone notebooks
   still use ChromaDB as required by the assignment brief (Task 3.2.2).

3. No `peft`: importing it alone costs ~225MB (transformers' full model-registration graph
   plus its own extras). The LoRA math itself is simple — `output = base(x) + scaling *
   B(A(x))` — so the adapter is applied by hand (see `apply_lora_manually` below) directly on
   the plain `transformers` model, reading the same `adapter_config.json` /
   `adapter_model.safetensors` that `peft.PeftModel.save_pretrained` produced. Numerically
   identical to loading via `peft`; the capstone notebooks use real `peft` throughout.
"""
import glob
import json
import re
from contextlib import contextmanager

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SOP_DIR = "streamlit_app/Dataset/sop_documents"
LORA_DIR = "streamlit_app/intent_lora_demo"
DEVICE = "cpu"  # Streamlit Community Cloud gives CPU-only containers

GEN_KWARGS = dict(max_new_tokens=80, do_sample=False, temperature=None, top_p=None)
ROUTER_GEN_KWARGS = dict(max_new_tokens=24, do_sample=False, temperature=None, top_p=None)

ROUTER_SYSTEM_PROMPT = (
    "You are a customer support intent router. Given a customer message, respond with a single "
    "JSON object containing exactly two fields: \"intent\" and \"category\". Output JSON only, "
    "with no explanation."
)
# One worked example: small models follow a demonstrated pattern far more reliably than a
# bare instruction (zero-shot instruction-following is weak at 135M scale).
ANSWER_FEWSHOT = (
    "Example — SOP: \"Refunds are issued within 5-7 business days to the original payment "
    "method.\" Question: \"where's my refund\" Answer: \"Your refund will be issued within "
    "5-7 business days to your original payment method.\"\n\n"
)
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


class _LoRAWrapper:
    """Adds a LoRA delta on top of one frozen nn.Linear's own forward, with an on/off switch."""

    def __init__(self, base_linear, lora_A, lora_B, scaling):
        self.original_forward = base_linear.forward
        self.A = lora_A  # (r, in_features)
        self.B = lora_B  # (out_features, r)
        self.scaling = scaling
        self.enabled = True

    def __call__(self, x):
        result = self.original_forward(x)
        if self.enabled:
            result = result + F.linear(F.linear(x, self.A), self.B) * self.scaling
        return result


def apply_lora_manually(base_model, lora_dir):
    """Monkey-patches q_proj/v_proj forwards with the trained LoRA delta. Returns the list of
    wrappers so callers can flip `.enabled` on all of them at once (mirrors peft's
    disable_adapter() context, without importing peft)."""
    with open(f"{lora_dir}/adapter_config.json") as f:
        cfg = json.load(f)
    scaling = cfg["lora_alpha"] / cfg["r"]
    state_dict = load_file(f"{lora_dir}/adapter_model.safetensors")

    model_dtype = next(base_model.parameters()).dtype
    pattern = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(\w+)\.lora_(A|B)\.weight")
    grouped = {}
    for key, tensor in state_dict.items():
        m = pattern.match(key)
        if not m:
            continue
        layer_idx, module_name, ab = m.groups()
        # adapter was trained/saved in fp32; cast to match the (possibly lower-precision) base model
        grouped.setdefault((int(layer_idx), module_name), {})[ab] = tensor.to(model_dtype)

    wrappers = []
    for (layer_idx, module_name), ab in grouped.items():
        target_linear = getattr(base_model.model.layers[layer_idx].self_attn, module_name)
        wrapper = _LoRAWrapper(target_linear, ab["A"], ab["B"], scaling)
        target_linear.forward = wrapper
        wrappers.append(wrapper)
    return wrappers


@contextmanager
def adapter_disabled(wrappers):
    for w in wrappers:
        w.enabled = False
    try:
        yield
    finally:
        for w in wrappers:
            w.enabled = True


@st.cache_resource(show_spinner="Loading model (first run takes ~20-30s)...")
def load_model():
    """One shared model: the LoRA adapter is toggled on/off via adapter_disabled() instead of
    loading two full copies — keeps memory to a single ~270MB model on this free-tier host."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bf16 (half the memory of fp32) + low_cpu_mem_usage=True (skips transformers' default
    # random-init-then-overwrite loading path, which otherwise transiently double-allocates
    # the model) — together these cut peak RSS for this step from ~820MB to ~60MB.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    lora_wrappers = apply_lora_manually(model, LORA_DIR)
    return tokenizer, model, lora_wrappers


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    return tok, model


def embed(texts, embed_tok, embed_model):
    inputs = embed_tok(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = embed_model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1).float().numpy()


@st.cache_resource(show_spinner="Building retrieval index...")
def load_corpus():
    embed_tok, embed_model = load_embedder()
    sop_files = sorted(glob.glob(f"{SOP_DIR}/*.md"))
    names, texts = [], []
    for f in sop_files:
        with open(f, "r", encoding="utf-8") as fh:
            texts.append(fh.read())
        names.append(f.split("/")[-1])
    vectors = embed(texts, embed_tok, embed_model)
    return names, texts, vectors


def retrieve(query, names, texts, vectors, embed_tok, embed_model, k=1):
    q_vec = embed([query], embed_tok, embed_model)[0]
    scores = vectors @ q_vec
    top_idx = np.argsort(-scores)[:k]
    return [(names[i], texts[i]) for i in top_idx]


tokenizer, model, lora_wrappers = load_model()
embed_tok, embed_model = load_embedder()
sop_names, sop_texts, sop_vectors = load_corpus()


def _generate(messages, gen_kwargs, use_adapter):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        if use_adapter:
            out = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **gen_kwargs)
        else:
            with adapter_disabled(lora_wrappers):
                out = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **gen_kwargs)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_baseline(query):
    messages = [{"role": "system", "content": "You are a helpful customer support assistant."},
                {"role": "user", "content": query}]
    return _generate(messages, GEN_KWARGS, use_adapter=False)


def generate_with_context(query, context):
    system_prompt = (
        f"{ANSWER_FEWSHOT}You are a customer support assistant. Answer strictly using this SOP:\n\n"
        f"{context}\n\nGive a short, direct answer to the customer's question."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    return _generate(messages, GEN_KWARGS, use_adapter=False)


def generate_naive_rag(query, k=1):
    retrieved = retrieve(query, sop_names, sop_texts, sop_vectors, embed_tok, embed_model, k=k)
    context = "\n\n".join(text for _, text in retrieved)
    answer = generate_with_context(query, context)
    return answer, retrieved[0][0]


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
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}, {"role": "user", "content": query}]
    raw = _generate(messages, ROUTER_GEN_KWARGS, use_adapter=True)
    return raw, extract_json(raw)


def generate_hybrid_rag(query, k=1):
    raw_router_output, parsed = run_router(query)
    if parsed and "intent" in parsed:
        intent = parsed["intent"]
        search_string = INTENT_TO_SEARCH_STRING.get(intent, intent.replace("_", " "))
    else:
        intent, search_string = None, query
    retrieved = retrieve(search_string, sop_names, sop_texts, sop_vectors, embed_tok, embed_model, k=k)
    context = "\n\n".join(text for _, text in retrieved)
    answer = generate_with_context(query, context)
    return answer, retrieved[0][0], raw_router_output, intent, search_string


st.set_page_config(page_title="Hybrid RAG Customer Support", page_icon="🎧", layout="wide")
st.title("Hybrid RAG & Fine-Tuning for Customer Support")
st.caption(
    f"Demo model: `{MODEL_ID}` (small, free-tier friendly) · Router: LoRA-fine-tuned intent "
    f"extractor · Retrieval: manual cosine similarity ({len(sop_names)} SOP docs) + MiniLM."
)
st.info(
    "The graded capstone fine-tunes and evaluates **Qwen2.5-1.5B-Instruct** with a **ChromaDB** "
    "vector index end-to-end (see the notebooks). This public demo swaps in a much smaller model "
    "and a lightweight retrieval implementation instead — free hosting caps this app at 1GB RAM, "
    "well under what the real setup needs.",
    icon="ℹ️",
)

EXAMPLES = [
    "my package is still not here and its been forever, this is ridiculous, where even is it??",
    "i want my money back for the thing i bought last week",
    "cant log into my account, forgot the password",
    "how do i stop getting charged every month for this subscription",
]

query = st.text_input("Customer query", placeholder="Type a support message...")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex[:28] + "...", help=ex):
        query = ex

run = st.button("Compare all 3 systems", type="primary")

if run and query.strip():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Baseline")
        st.caption("No context")
        with st.spinner("Generating..."):
            st.write(generate_baseline(query))

    with col2:
        st.subheader("2. Naive RAG")
        st.caption("Raw-query retrieval")
        with st.spinner("Generating..."):
            rag_out, rag_doc = generate_naive_rag(query, k=1)
        st.markdown(f"**Retrieved:** `{rag_doc}`")
        st.write(rag_out)

    with col3:
        st.subheader("3. Hybrid RAG")
        st.caption("Fine-tuned router")
        with st.spinner("Generating..."):
            hybrid_out, hybrid_doc, router_raw, intent, search_string = generate_hybrid_rag(query, k=1)
        st.markdown(f"**Router:** `{router_raw}`")
        st.markdown(f"**Search string:** `{search_string}` | **Retrieved:** `{hybrid_doc}`")
        st.write(hybrid_out)
elif run:
    st.warning("Enter a query above.")
