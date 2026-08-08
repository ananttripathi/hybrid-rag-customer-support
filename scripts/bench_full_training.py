import sys, time, copy, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_from_disk

def log(msg):
    print(msg, flush=True)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEVICE)
log(f"model loaded in {time.time()-t0:.1f}s")

lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj"], task_type=TaskType.CAUSAL_LM)
peft_model = get_peft_model(base_model, lora_config).to(DEVICE)
peft_model.print_trainable_parameters()

hf_train = load_from_disk("./tokenized_train")
hf_valid = load_from_disk("./tokenized_valid")
hf_train.set_format("torch", columns=["input_ids","attention_mask","labels"])
hf_valid.set_format("torch", columns=["input_ids","attention_mask","labels"])

BATCH_SIZE = 4
train_loader = DataLoader(hf_train, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(hf_valid, batch_size=BATCH_SIZE, shuffle=False)
log(f"train batches: {len(train_loader)} valid batches: {len(valid_loader)}")

optimizer = AdamW(filter(lambda p: p.requires_grad, peft_model.parameters()), lr=2e-4)

MAX_STEPS = 20
VAL_EVERY = 10
PATIENCE = 3

train_iter = iter(train_loader)
peft_model.train()
step = 0
t_start = time.time()
while step < MAX_STEPS:
    tstep = time.time()
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
    log(f"step {step} loss {loss.item():.4f} step_time {time.time()-tstep:.2f}s total_elapsed {time.time()-t_start:.1f}s")

    if step % VAL_EVERY == 0:
        tval = time.time()
        peft_model.eval()
        val_loss_total, val_batches = 0.0, 0
        with torch.no_grad():
            for j, vbatch in enumerate(valid_loader):
                vbatch = {k: v.to(DEVICE) for k, v in vbatch.items()}
                vout = peft_model(**vbatch)
                val_loss_total += vout.loss.item()
                val_batches += 1
                if j % 10 == 0:
                    log(f"  val batch {j}/{len(valid_loader)} elapsed {time.time()-tval:.1f}s")
        val_loss = val_loss_total / val_batches
        log(f"VALIDATION at step {step}: val_loss={val_loss:.4f} val_time={time.time()-tval:.1f}s")
        peft_model.train()

log(f"DONE. total time for {MAX_STEPS} steps + validations: {time.time()-t_start:.1f}s")
