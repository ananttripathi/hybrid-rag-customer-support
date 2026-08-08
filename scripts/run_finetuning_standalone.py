"""Standalone equivalent of Fine_Tuning_Pipeline.ipynb cells 2,4,6,7,9 — run outside the Jupyter
kernel (which showed anomalous 10-20x slowdown vs a plain script for reasons not fully diagnosed;
plain-script execution is proven fast and reliable via scripts/bench_full_training.py).
Prints a '===CELL_BOUNDARY===' marker between sections so the captured log can be split back into
per-cell notebook outputs afterward.
"""
import sys, json, copy, time
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_from_disk

def boundary():
    print("===CELL_BOUNDARY===", flush=True)

# ============ Cell 2: Configure PEFT Framework ============
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# bf16 (not fp32): halves memory for weights+activations vs fp32, and unlike fp16 keeps fp32's full
# exponent range so it stays numerically stable for training on MPS. This edit was made after an
# fp32 run on this machine caused system-wide swap thrashing (see reproducibility notes).
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()  # required so grads flow into LoRA layers under checkpointing

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(base_model, lora_config)
peft_model.to(DEVICE)
peft_model.print_trainable_parameters()
boundary()

# ============ Cell 4: Configure Training Parameters ============
hf_train = load_from_disk("./tokenized_train")
hf_valid = load_from_disk("./tokenized_valid")
hf_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
hf_valid.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

BATCH_SIZE = 2  # reduced from 4 to further cut activation memory after the swap-thrashing incident
LEARNING_RATE = 2e-4
MAX_STEPS = 200
VAL_EVERY = 20
PATIENCE = 3
SEED = 42

torch.manual_seed(SEED)

train_loader = DataLoader(hf_train, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(hf_valid, batch_size=BATCH_SIZE, shuffle=False)

optimizer = AdamW(filter(lambda p: p.requires_grad, peft_model.parameters()), lr=LEARNING_RATE)

print(f"Train batches/epoch: {len(train_loader)} | Valid batches: {len(valid_loader)}")
print(f"MAX_STEPS={MAX_STEPS}, VAL_EVERY={VAL_EVERY}, PATIENCE={PATIENCE}, "
      f"batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
print("Optimiser: AdamW over LoRA-adapter parameters only (base model frozen).")
boundary()

# ============ Cell 6: Execute Fine-Tuning ============
train_losses, val_losses = [], []
best_val_loss = float("inf")
best_state_dict = None
patience_counter = 0
step = 0
early_stopped = False

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
    out = peft_model(**batch)
    loss = out.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    step += 1
    train_losses.append((step, loss.item()))

    if step % VAL_EVERY == 0 or step == MAX_STEPS:
        peft_model.eval()
        val_loss_total, val_batches = 0.0, 0
        with torch.no_grad():
            for vbatch in valid_loader:
                vbatch = {k: v.to(DEVICE) for k, v in vbatch.items()}
                vout = peft_model(**vbatch)
                val_loss_total += vout.loss.item()
                val_batches += 1
        val_loss = val_loss_total / val_batches
        val_losses.append((step, val_loss))
        peft_model.train()

        elapsed = time.time() - t0
        print(f"step {step:4d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | elapsed {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(peft_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered at step {step}: val_loss did not improve for {PATIENCE} checks.")
                early_stopped = True

final_state_dict = copy.deepcopy(peft_model.state_dict())
print(f"\nTraining finished at step {step}/{MAX_STEPS} (early_stopped={early_stopped}). "
      f"Best val_loss: {best_val_loss:.4f}")
boundary()

# ============ Cell 7: Plot Loss Curves ============
train_df = pd.DataFrame(train_losses, columns=["step", "train_loss"])
val_df = pd.DataFrame(val_losses, columns=["step", "val_loss"])
best_step = val_df.loc[val_df["val_loss"].idxmin(), "step"]

plt.figure(figsize=(9, 5))
plt.plot(train_df["step"], train_df["train_loss"], label="Train Loss", alpha=0.6)
plt.plot(val_df["step"], val_df["val_loss"], label="Validation Loss", marker="o")
plt.axvline(best_step, color="green", linestyle="--", label=f"Best checkpoint (step {int(best_step)})")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss — LoRA Intent Router")
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=100)
plt.close()

print(f"Lowest validation loss: {val_df['val_loss'].min():.4f} at step {int(best_step)}")
if val_df["val_loss"].iloc[-1] > val_df["val_loss"].min():
    print("Validation loss rose after its minimum — early stopping correctly preserves the BEST "
          "checkpoint rather than the final (potentially overfit) one.")
else:
    print("Validation loss was still improving when training stopped.")
boundary()

# ============ Cell 9: Save Training Outputs ============
peft_model.load_state_dict(final_state_dict)
peft_model.save_pretrained("./intent_lora")
print("Saved FINAL checkpoint (last step) to ./intent_lora")

peft_model.load_state_dict(best_state_dict)
peft_model.save_pretrained("./intent_lora_best")
tokenizer.save_pretrained("./intent_lora_best")
print("Saved BEST checkpoint (lowest val_loss) to ./intent_lora_best  <- used by Notebook 7")

train_log_df = pd.merge(train_df, val_df, on="step", how="outer").sort_values("step")
train_log_df.to_csv("training_log.csv", index=False)
print("\nSaved training_log.csv")

reproducibility_info = {
    "MODEL_ID": MODEL_ID,
    "lora_config": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "max_steps": MAX_STEPS,
    "actual_steps": step,
    "val_every": VAL_EVERY,
    "patience": PATIENCE,
    "seed": SEED,
    "best_val_loss": best_val_loss,
    "best_step": int(best_step),
    "early_stopped": early_stopped,
    "device": DEVICE,
    "precision": "bfloat16 + gradient checkpointing (MPS training — bitsandbytes/QLoRA unavailable, "
                 "see Project Proposal 1.4.3; switched from fp32 after fp32 caused system swap thrashing)",
}
with open("training_reproducibility.json", "w") as f:
    json.dump(reproducibility_info, f, indent=2)

print("\nSaved training_reproducibility.json:")
print(json.dumps(reproducibility_info, indent=2))
boundary()
print("ALL_DONE", flush=True)
