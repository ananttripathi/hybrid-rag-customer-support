"""Fine-tunes a small, demo-only LoRA intent router for the public Streamlit app.

Separate from the graded capstone: the capstone's Fine_Tuning_Pipeline.ipynb trains against
Qwen2.5-1.5B-Instruct (kept untouched). Free hosting (Streamlit Community Cloud, 1GB RAM cap)
can't fit that model, so this script re-runs the identical cleaning/split/ChatML/LoRA recipe
against HuggingFaceTB/SmolLM2-135M-Instruct (much smaller vocab: 49,152 vs Qwen's 151,936 —
vocab size dominates memory footprint far more than parameter count at this scale).
"""
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUT_DIR = "streamlit_app/intent_lora_demo"
MAX_LENGTH = 128

ROUTER_SYSTEM_PROMPT = (
    "You are a customer support intent router. Given a customer message, respond with a single "
    "JSON object containing exactly two fields: \"intent\" and \"category\". Output JSON only, "
    "with no explanation."
)

# --- Reproduce Data_Preparation.ipynb's cleaning exactly (same source file, same random_state) ---
df = pd.read_csv("sampled_data.csv")
df["_norm"] = df["instruction"].str.lower().str.strip()
df = df.drop_duplicates(subset="_norm").drop(columns="_norm").reset_index(drop=True)
df = df.dropna(subset=["instruction", "intent", "category"]).reset_index(drop=True)
df = df[
    (df["instruction"].str.strip().str.len() > 0)
    & (df["intent"].str.strip().str.len() > 0)
    & (df["category"].str.strip().str.len() > 0)
].reset_index(drop=True)
df["instruction"] = df["instruction"].str.strip()
df["intent"] = df["intent"].str.strip().str.lower()
df["category"] = df["category"].str.strip().str.upper()
print(f"Cleaned dataset: {len(df)} rows")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def to_chatml(row):
    target = json.dumps({"intent": row["intent"], "category": row["category"]})
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": target},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False), target


chatml_texts, targets = [], []
for _, row in df.iterrows():
    text, target = to_chatml(row)
    chatml_texts.append(text)
    targets.append(target)
df["text"] = chatml_texts
df["target_json"] = targets

# --- Tokenise + label-mask (prompt tokens and padding both -> -100) ---
ASSISTANT_MARKER_CANDIDATES = ["<|im_start|>assistant\n", "assistant\n", "<|assistant|>\n"]
marker_ids = None
for cand in ASSISTANT_MARKER_CANDIDATES:
    ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
    if ids:
        marker_ids = ids
        break


def find_assistant_start(input_ids):
    n = len(marker_ids)
    for i in range(len(input_ids) - n + 1):
        if input_ids[i:i + n] == marker_ids:
            return i + n
    return None


def tokenize_batch(examples):
    input_ids_list, attn_list, labels_list = [], [], []
    for text in examples["text"]:
        enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        input_ids = enc["input_ids"]
        labels = list(input_ids)
        a_start = find_assistant_start(input_ids)
        for i in range(len(labels)):
            if input_ids[i] == tokenizer.pad_token_id:
                labels[i] = -100
            elif a_start is not None and i < a_start:
                labels[i] = -100
        input_ids_list.append(input_ids)
        attn_list.append(enc["attention_mask"])
        labels_list.append(labels)
    return {"input_ids": input_ids_list, "attention_mask": attn_list, "labels": labels_list}


hf_dataset = Dataset.from_pandas(df[["text", "intent", "category"]])
hf_dataset = hf_dataset.map(tokenize_batch, batched=True, batch_size=32)

# --- Same 80/10/10 stratified split, same seed, as Data_Preparation.ipynb ---
indices = np.arange(len(df))
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=df["intent"])
valid_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=df["intent"].iloc[temp_idx]
)
hf_train = hf_dataset.select(train_idx)
hf_valid = hf_dataset.select(valid_idx)
hf_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
hf_valid.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
print(f"Train: {len(hf_train)} | Valid: {len(hf_valid)}")

# --- LoRA fine-tune (same recipe as the capstone's Fine_Tuning_Pipeline.ipynb) ---
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEVICE)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(base_model, lora_config).to(DEVICE)
peft_model.print_trainable_parameters()

BATCH_SIZE, LEARNING_RATE, MAX_STEPS, VAL_EVERY, PATIENCE, SEED = 4, 2e-4, 200, 20, 3, 42
torch.manual_seed(SEED)

train_loader = DataLoader(hf_train, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(hf_valid, batch_size=BATCH_SIZE, shuffle=False)
optimizer = AdamW(filter(lambda p: p.requires_grad, peft_model.parameters()), lr=LEARNING_RATE)

best_val_loss, best_state_dict, patience_counter, step, early_stopped = float("inf"), None, 0, 0, False
train_iter = iter(train_loader)
peft_model.train()
t0 = time.time()

while step < MAX_STEPS and not early_stopped:
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    loss = peft_model(**batch).loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step += 1

    if step % VAL_EVERY == 0 or step == MAX_STEPS:
        peft_model.eval()
        val_loss_total, val_batches = 0.0, 0
        with torch.no_grad():
            for vbatch in valid_loader:
                vbatch = {k: v.to(DEVICE) for k, v in vbatch.items()}
                val_loss_total += peft_model(**vbatch).loss.item()
                val_batches += 1
        val_loss = val_loss_total / val_batches
        peft_model.train()
        print(f"step {step:4d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} "
              f"| elapsed {time.time() - t0:.0f}s", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(peft_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at step {step}")
                early_stopped = True

peft_model.load_state_dict(best_state_dict)
peft_model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"\nSaved demo router (best val_loss {best_val_loss:.4f}) to {OUT_DIR}")
