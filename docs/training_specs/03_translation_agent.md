# Translation Agent Training Spec

## Model Details
- **Model**: NLLB-200 3.3B (No Language Left Behind)
- **Objective**: High-fidelity translation, domain adaptation (e.g., medical/legal).
- **Training Method**: Full Fine-Tuning or LoRA (Sequence-to-Sequence).

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate (LR)** | 2.0e-5 | Translation models are highly sensitive; high LR destroys multilingual ability. |
| **Label Smoothing** | 0.1 | Prevents the model from becoming overconfident, standard in NMT. |
| **Dropout** | 0.1 | Standard regularization for transformers. |
| **Epochs** | 1 - 5 | 1 for large corpora, up to 5 for small domain data. Early stopping is crucial. |
| **Batch Size** | 128 (Tokens) | Defined in max_tokens (e.g., 4096 tokens per batch) rather than absolute sentence count. |
| **Optimizer** | AdamW | Betas 0.9, 0.98. Standard optimizer for translation transformers. |
| **LoRA Config** | r=16, alpha=32 | Optional. Target `q`, `v` layers in both Encoder and Decoder if using LoRA. |

## Data & Preprocessing
- **Datasets**: FLORES-200, OPUS (EuroParl, JW300) for specific language pairs.
- **Input Format**: Must use language codes (e.g., `fra_Latn` for French).

### Example
**Input**: `eng_Latn: Hello world </s> fra_Latn`
**Target**: `Bonjour le monde </s>`
