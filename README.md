# Hybrid RAG & Fine-Tuning for Customer Support

End-to-end capstone: baseline LLM → retrieval-assisted generation (Naive RAG) → LoRA-fine-tuned
intent router driving retrieval (Hybrid RAG). Built on `Qwen/Qwen2.5-1.5B-Instruct`.

**Live demo:** _link pending — deploying to Hugging Face Spaces_

## Notebooks (run free in Colab — click a badge)

| # | Notebook | |
|---|---|---|
| 1 | Data Understanding & EDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ananttripathi/hybrid-rag-customer-support/blob/main/Data_Understanding_and_EDA.ipynb) |
| 2 | Data Preparation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ananttripathi/hybrid-rag-customer-support/blob/main/Data_Preparation.ipynb) |
| 3 | Baseline Model Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ananttripathi/hybrid-rag-customer-support/blob/main/Baseline_Model_Evaluation.ipynb) |
| 4 | RAG Implementation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ananttripathi/hybrid-rag-customer-support/blob/main/RAG_Implementation.ipynb) |
| 5 | Solution V1 — RAG Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ananttripathi/hybrid-rag-customer-support/blob/main/Solution_V1_RAG_Evaluation.ipynb) |
| 6 | Fine-Tuning Pipeline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ananttripathi/hybrid-rag-customer-support/blob/main/Fine_Tuning_Pipeline.ipynb) |
| 7 | Solution V2 — Fine-Tuned RAG Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ananttripathi/hybrid-rag-customer-support/blob/main/Solution_V2_FineTuned_RAG_Evaluation.ipynb) |

Run notebooks in order — each saves artifacts (`sampled_data.csv`, `tokenized_*/`, `chroma_db/`,
`intent_lora_best/`, etc.) the next one depends on. On Colab: enable GPU runtime (Runtime → Change
runtime type → T4) for notebooks 3, 4, 5, 6, 7.

## Reports

- `Project_Proposal_and_Methodology.pdf` / `.docx`
- `Comparative_Analysis_Report.pdf` / `.docx`

## Live demo (`hf_space/`)

Gradio app comparing Baseline / Naive RAG / Hybrid RAG side by side on any query. Self-contained
(bundles SOP docs + the fine-tuned LoRA adapter). See `hf_space/README.md`.

## Results summary

| System | ROUGE-1 | ROUGE-L | BLEU | Intent Exact Match |
|---|---|---|---|---|
| Baseline | 0.167 | 0.090 | 0.001 | — |
| Naive RAG (V1) | 0.104 | 0.064 | 0.001 | — |
| Hybrid RAG (V2) | 0.106 | 0.067 | 0.001 | 65.1% |

Full breakdown, methodology, and limitations in `Comparative_Analysis_Report.pdf`.

## Stack

Python 3.11 · transformers · peft (LoRA) · sentence-transformers · ChromaDB · LangChain · Gradio

Executed on Apple Silicon (MPS) rather than the reference Colab/T4 — hardware adaptations
(precision choices, batch sizes) documented in the Project Proposal.
