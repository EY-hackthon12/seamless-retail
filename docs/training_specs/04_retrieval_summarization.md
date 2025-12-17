# Knowledge Retrieval & Summarization Training Spec

## 1. Knowledge Retrieval Agent (Embeddings)
- **Model**: Instructor-XL (1.5B)
- **Objective**: Semantic search and contrastive separation of domain documents.
- **Training Method**: Contrastive Fine-Tuning.

### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 2.0e-5 | Very low LR required to fine-tune encoders. |
| **Loss Function** | MultipleNegativesRankingLoss | Standard for semantic search; treats other samples in batch as negatives. |
| **Batch Size** | 64+ | Larger batch sizes provide more "hard negatives," improving retrieval quality. |
| **Epochs** | 1 - 2 | Embedding spaces distort easily; minimal training is best. |
| **Pooling** | Mean Pooling | Standard for Instructor models. |
| **Max Seq Length** | 512 | Standard limit for BERT/T5-based embedding models. |

### Data & Preprocessing
- **Instruction Format**: Crucial for Instructor-XL.
- **Input**: `["Represent the document for retrieval: ", "Document text..."]`
- **Datasets**: Domain-specific corpus (Wiki, Internal Docs) chunked into 256-512 token segments.

---

## 2. Summarization Agent
- **Model**: Mistral 7B (Fine-tuned specifically for summarization)
- **Objective**: Concise, bulleted, or executive summaries without hallucination.
- **Training Method**: QLoRA.

### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 1.0e-4 | Slightly lower than general instruct tuning to prevent "creative" hallucinations. |
| **Epochs** | 3 | Ensuring the model learns the specific "Summary" output format perfectly. |
| **Context Length** | 8192 - 16384 | Summarization requires reading long documents; maximize context usage. |
| **Loss Masking** | Response Only | Mask the input text loss so the model only learns to generate the summary. |

### Data & Preprocessing
- **Datasets**: CNN/DailyMail, XSum, GovReport.
- **Prompt Strategy**: Append "TL;DR" or "Provide a structured summary:" to the end of inputs.
