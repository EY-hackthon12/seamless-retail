# Code Generation Agent Training Spec

## Model Details
- **Model**: StarCoder2 15B
- **Objective**: Python/JS/C++ expertise, rigorous syntax adherence, and explanation.
- **Training Method**: QLoRA (due to 15B size fitting on 24GB-48GB VRAM).

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate (LR)** | 1.0e-4 | Slightly lower than text models to preserve precise syntax knowledge. |
| **Scheduler** | Cosine | Standard for code models. |
| **Epochs** | 3 | Code models require more iterations to internalize new constraints. |
| **Batch Size** | 16 (Effective) | Lower batch size to accommodate the larger 15B parameter model memory footprint. |
| **Context Length** | 8192 | Essential for long code files; StarCoder2 is optimized for long context. |
| **LoRA Rank (r)** | 32 | Code syntax is structured; slightly lower rank than reasoning models is often sufficient. |
| **LoRA Alpha** | 64 | Alpha = 2 × Rank. |
| **Target Modules** | All Linear | `c_attn`, `c_proj`, `c_fc` (Falcon/BigCode style naming). |
| **Quantization** | 4-bit (NF4) | Required to fit 15B params on consumer hardware. |

## Data & Preprocessing
- **Datasets**: WizardCoder (Evolved Instructions), HumanEval-like problems, StackExchange Q&A.
- **Special Token Handling**: Ensure `<fim_prefix>`, `<fim_suffix>`, and `<fim_middle>` tokens are preserved if doing Fill-In-The-Middle (FIM) training.
