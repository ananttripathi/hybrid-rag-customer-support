---
title: Hybrid RAG Customer Support
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
---

# Hybrid RAG & Fine-Tuning for Customer Support

Live demo of the capstone: compares Baseline, Naive RAG, and Hybrid RAG (LoRA fine-tuned intent
router) side by side on the same customer query.

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval: ChromaDB over `Dataset/sop_documents/`
- Router: LoRA adapter in `intent_lora_best/`, trained in the capstone's Fine_Tuning_Pipeline notebook

Full project + notebooks: see the GitHub repo linked in the Space settings.

Runs on free CPU hardware — each comparison takes ~20-40s (3 model generations per query).
