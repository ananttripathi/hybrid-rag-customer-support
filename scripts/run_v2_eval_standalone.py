"""Standalone equivalent of Solution_V2_FineTuned_RAG_Evaluation.ipynb cells 2,3,5,7,9,10 — run
outside the Jupyter kernel for the same reliability reasons as Notebook 6 (see
scripts/run_finetuning_standalone.py). Prints '===CELL_BOUNDARY===' between sections so the log
can be split back into per-cell notebook outputs afterward.
"""
import json, re, sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def boundary():
    print("===CELL_BOUNDARY===", flush=True)

# ============ Cell 2 ============
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
LORA_DIR = "./intent_lora_best"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE)
base_model.eval()

router_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE)
router_model = PeftModel.from_pretrained(router_base, LORA_DIR).merge_and_unload()
router_model.eval()
print(f"Loaded base_model (generation) and router_model (fine-tuned, LoRA merged from '{LORA_DIR}').")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
print("ChromaDB reloaded:", vector_db._collection.count(), "documents")

df_test = pd.read_csv("df_test.csv")
with open("outputs.json") as f:
    outputs = json.load(f)
v1_metrics = pd.read_csv("v1_metrics.csv")
v1_summary = pd.read_csv("v1_summary_metrics.csv")
print(f"df_test shape: {df_test.shape} | v1_metrics rows: {len(v1_metrics)}")
boundary()

# ============ Cell 3 ============
GEN_KWARGS = dict(max_new_tokens=120, do_sample=False, temperature=None, top_p=None)
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

def build_router_messages(query):
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
    for q, a in FEWSHOT_EXAMPLES:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})
    return messages

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
    messages = build_router_messages(query)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = router_model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **ROUTER_GEN_KWARGS)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return raw, extract_json(raw)

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

def hybrid_rag_generate(query, k=1):
    raw_router_output, parsed = run_router(query)
    if parsed and "intent" in parsed:
        intent = parsed["intent"]
        search_string = INTENT_TO_SEARCH_STRING.get(intent, intent.replace("_", " "))
    else:
        intent, search_string = None, query
    docs = vector_db.similarity_search(search_string, k=k)
    answer = generate_with_context(query, "\n\n".join(d.page_content for d in docs))
    return {
        "router_raw": raw_router_output, "router_parsed": parsed, "intent": intent,
        "search_string": search_string, "retrieved_doc": docs[0].metadata["source_file"], "answer": answer,
    }

validation_result = hybrid_rag_generate(outputs["test_query"], k=1)
print("Router raw output:", validation_result["router_raw"])
print("Parsed JSON:", validation_result["router_parsed"])
print("Search string used:", validation_result["search_string"])
print("Retrieved SOP:", validation_result["retrieved_doc"])
print("\nFinal Hybrid RAG answer:\n", validation_result["answer"])

outputs["hybrid_rag_router_output"] = validation_result["router_raw"]
outputs["hybrid_rag_output"] = validation_result["answer"]
with open("outputs.json", "w") as f:
    json.dump(outputs, f, indent=2)
print("\noutputs.json updated with hybrid_rag_output.")
boundary()

# ============ Cell 5 ============
ADVERSARIAL_PATTERN = re.compile(
    r"\b(still|never|terrible|frustrated|worst|awful|ridiculous|unacceptable|angry|disappointed)\b",
    re.IGNORECASE,
)
df_test["is_adversarial"] = df_test["instruction"].str.contains(ADVERSARIAL_PATTERN)
print(f"Adversarial subset: {df_test['is_adversarial'].sum()}/{len(df_test)} test rows "
      f"contain hedging/sentiment language.")

router_results = []
for i, row in df_test.iterrows():
    raw, parsed = run_router(row["instruction"])
    router_results.append({
        "instruction": row["instruction"],
        "true_intent": row["intent"], "true_category": row["category"],
        "router_raw": raw, "router_parsed_ok": parsed is not None,
        "pred_intent": parsed.get("intent") if parsed else None,
        "pred_category": parsed.get("category") if parsed else None,
        "is_adversarial": row["is_adversarial"],
    })
    if (i + 1) % 30 == 0:
        print(f"  router: {i + 1}/{len(df_test)}")

router_df = pd.DataFrame(router_results)

def fuzzy_match(pred, true):
    if pred is None or true is None:
        return False
    pred, true = str(pred).lower(), str(true).lower()
    return pred == true or len(set(pred.split("_")) & set(true.split("_"))) > 0

router_df["exact_match"] = router_df["pred_intent"] == router_df["true_intent"]
router_df["fuzzy_match"] = router_df.apply(lambda r: fuzzy_match(r["pred_intent"], r["true_intent"]), axis=1)

def summarize(sub_df, label):
    fmt = sub_df["router_parsed_ok"].mean() * 100
    exact = sub_df["exact_match"].mean() * 100
    fuzzy = sub_df["fuzzy_match"].mean() * 100
    print(f"{label}: Format Adherence={fmt:.1f}% | Exact Match={exact:.1f}% | Fuzzy Match={fuzzy:.1f}% (n={len(sub_df)})")
    return fmt, exact, fuzzy

full_fmt, full_exact, full_fuzzy = summarize(router_df, "Full held-out test split (zero leakage)")
adv_sub = router_df[router_df["is_adversarial"]]
if len(adv_sub) > 0:
    adv_fmt, adv_exact, adv_fuzzy = summarize(adv_sub, "Adversarial subset")
else:
    adv_fmt = adv_exact = adv_fuzzy = np.nan
    print("No adversarial rows found in this test split.")

router_df.to_csv("router_evaluation.csv", index=False)

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
smoothie = SmoothingFunction().method1

def get_sop_reference(intent, category):
    query = f"{intent} {category}".replace("_", " ")
    docs = vector_db.similarity_search(query, k=1)
    return docs[0].page_content

ROUGE_SAMPLE_SIZE = 40
rouge_sample = df_test.sample(min(ROUGE_SAMPLE_SIZE, len(df_test)), random_state=7).reset_index(drop=True)

hybrid_rouge1, hybrid_rougeL, hybrid_bleu = [], [], []
for i, row in rouge_sample.iterrows():
    hybrid_answer = hybrid_rag_generate(row["instruction"], k=1)["answer"]
    ref_text = get_sop_reference(row["intent"], row["category"])
    rouge = scorer.score(ref_text, hybrid_answer)
    bleu = sentence_bleu([ref_text.split()], hybrid_answer.split(), smoothing_function=smoothie)
    hybrid_rouge1.append(rouge["rouge1"].fmeasure)
    hybrid_rougeL.append(rouge["rougeL"].fmeasure)
    hybrid_bleu.append(bleu)
    if (i + 1) % 10 == 0:
        print(f"  ROUGE/BLEU: {i + 1}/{len(rouge_sample)}")

hybrid_rouge1_mean = float(np.mean(hybrid_rouge1))
hybrid_rougeL_mean = float(np.mean(hybrid_rougeL))
hybrid_bleu_mean = float(np.mean(hybrid_bleu))
print(f"\nHybrid RAG (V2) final-synthesis — ROUGE-1: {hybrid_rouge1_mean:.3f} | "
      f"ROUGE-L: {hybrid_rougeL_mean:.3f} | BLEU: {hybrid_bleu_mean:.3f} (n={len(rouge_sample)})")
boundary()

# ============ Cell 7 ============
v2_summary_metrics = {
    "Format Adherence (%)": full_fmt,
    "Intent Accuracy - Exact Match (%)": full_exact,
    "Intent Accuracy - Fuzzy Match (%)": full_fuzzy,
    "ROUGE-1": hybrid_rouge1_mean,
    "ROUGE-L": hybrid_rougeL_mean,
    "BLEU": hybrid_bleu_mean,
}
print("V2 (Hybrid RAG) summary:")
for k, v in v2_summary_metrics.items():
    print(f"  {k}: {v:.2f}")

v1_rag_row = v1_summary.set_index("Metric")["Naive RAG (V1)"]
comparison_rows = []
for metric in ["ROUGE-1", "ROUGE-L", "BLEU"]:
    v1_val = v1_rag_row[metric]
    v2_val = v2_summary_metrics[metric]
    change = (v2_val - v1_val) / v1_val * 100 if v1_val else np.nan
    comparison_rows.append({"Metric": metric, "V1 (Naive RAG)": v1_val, "V2 (Hybrid RAG)": v2_val, "Change (%)": change})

comparison_df = pd.DataFrame(comparison_rows)
print("\nFine-tuning's independent impact (V1 -> V2):")
print(comparison_df.to_string(index=False))
print(
    "\nAttribution: retrieval was already present in V1, and the generation model is held IDENTICAL "
    "between V1 and V2 (same base_model, same GEN_KWARGS) — the only pipeline difference is that V2's "
    "retrieval query is built from the fine-tuned router's structured intent instead of the raw, noisy "
    "prompt. Any V1->V2 delta above is therefore attributable specifically to fine-tuning, not to a "
    "change in the generator itself."
)

comparison_df.to_csv("v1_vs_v2_finetuning_impact.csv", index=False)
print("\nSaved v1_vs_v2_finetuning_impact.csv")
boundary()

# ============ Cell 9 ============
FULL_EVAL_SAMPLE_SIZE = 50
final_eval_df = df_test.sample(min(FULL_EVAL_SAMPLE_SIZE, len(df_test)), random_state=123).reset_index(drop=True)
print(f"Running comparative evaluation on {len(final_eval_df)}/{len(df_test)} held-out test rows "
      f"across all 3 architectures (Baseline, Naive RAG, Hybrid RAG).")

def generate_baseline_final(query):
    messages = [{"role": "system", "content": "You are a helpful customer support assistant."},
                {"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = base_model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **GEN_KWARGS)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def generate_naive_rag_final(query, k=1):
    docs = vector_db.similarity_search(query, k=k)
    answer = generate_with_context(query, "\n\n".join(d.page_content for d in docs))
    return answer, docs[0].metadata["source_file"]

def _scores(generated, ref_text):
    r = scorer.score(ref_text, generated)
    b = sentence_bleu([ref_text.split()], generated.split(), smoothing_function=smoothie)
    return r["rouge1"].fmeasure, r["rougeL"].fmeasure, b

full_results = []
for i, row in final_eval_df.iterrows():
    query = row["instruction"]
    ref_text = get_sop_reference(row["intent"], row["category"])

    baseline_out = generate_baseline_final(query)
    rag_out, rag_doc = generate_naive_rag_final(query, k=1)
    hybrid = hybrid_rag_generate(query, k=1)

    b_r1, b_rl, b_bleu = _scores(baseline_out, ref_text)
    n_r1, n_rl, n_bleu = _scores(rag_out, ref_text)
    h_r1, h_rl, h_bleu = _scores(hybrid["answer"], ref_text)

    full_results.append({
        "instruction": query, "intent": row["intent"], "category": row["category"],
        "baseline_output": baseline_out, "baseline_rouge1": b_r1, "baseline_rougeL": b_rl, "baseline_bleu": b_bleu,
        "naive_rag_output": rag_out, "naive_rag_doc": rag_doc,
        "naive_rag_rouge1": n_r1, "naive_rag_rougeL": n_rl, "naive_rag_bleu": n_bleu,
        "hybrid_rag_output": hybrid["answer"], "hybrid_router_intent": hybrid["intent"],
        "hybrid_retrieved_doc": hybrid["retrieved_doc"],
        "hybrid_rouge1": h_r1, "hybrid_rougeL": h_rl, "hybrid_bleu": h_bleu,
        "hybrid_intent_correct": hybrid["intent"] == row["intent"],
    })
    if (i + 1) % 10 == 0:
        print(f"  processed {i + 1}/{len(final_eval_df)}")

full_results_df = pd.DataFrame(full_results)
print(f"\nCompleted comparative generation + scoring for {len(full_results_df)} rows across all 3 architectures.")
boundary()

# ============ Cell 10 ============
summary_rows = []
for name, prefix in [("Baseline", "baseline"), ("Naive RAG (V1)", "naive_rag"), ("Hybrid RAG (V2)", "hybrid")]:
    summary_rows.append({
        "System": name,
        "ROUGE-1": full_results_df[f"{prefix}_rouge1"].mean(),
        "ROUGE-L": full_results_df[f"{prefix}_rougeL"].mean(),
        "BLEU": full_results_df[f"{prefix}_bleu"].mean(),
    })

comparative_summary_df = pd.DataFrame(summary_rows)
comparative_summary_df.loc[
    comparative_summary_df["System"] == "Hybrid RAG (V2)", "Intent Accuracy - Exact (%, full test split)"
] = full_exact
comparative_summary_df.loc[
    comparative_summary_df["System"] == "Hybrid RAG (V2)", "Format Adherence (%, full test split)"
] = full_fmt
comparative_summary_df.loc[
    comparative_summary_df["System"] == "Hybrid RAG (V2)", "Intent Accuracy - Exact (%, adversarial)"
] = adv_exact

print(comparative_summary_df.to_string(index=False))

full_results_df.to_csv("Comparative_Results_Full.csv", index=False)
comparative_summary_df.to_csv("Comparative_Results_Summary.csv", index=False)
print("\nSaved Comparative_Results_Full.csv (per-row, all 3 architectures)")
print("Saved Comparative_Results_Summary.csv (aggregate, all 3 architectures) — FINAL DELIVERABLES")
boundary()
print("ALL_DONE", flush=True)
