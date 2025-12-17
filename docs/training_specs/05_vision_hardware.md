# Vision Agent & Hardware Requirements

## 1. Vision Agent (Optional)
- **Model**: Stable Diffusion XL (SDXL) Base 1.0
- **Objective**: Domain-specific image generation (e.g., architectural designs, fashion sketches).
- **Training Method**: LoRA (Text Encoder + UNet).

### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **LR (UNet)** | 1.0e-4 | Standard SDXL LoRA rate. |
| **LR (Text Enc)** | 5.0e-5 | The text encoder requires gentler updates than the visual UNet. |
| **Batch Size** | 1 | Higher batch sizes often degrade quality in diffusion LoRAs. |
| **Resolution** | 1024x1024 | Native resolution for SDXL. |
| **Precision** | fp16 | Mandatory for memory efficiency. |
| **LoRA Rank** | 128 | Higher rank needed for image style capture compared to text. |
| **Steps** | 1500 - 3000 | Approx 100-200 epochs depending on dataset size (usually 15-50 images). |

---

## 2. Hardware Requirements
To execute the training settings above, the following local hardware configuration is recommended:

- **GPU**: NVIDIA RTX 3090 / 4090 (24GB VRAM) or A6000 (48GB VRAM).
    - *Note*: StarCoder2 15B QLoRA training requires ~22GB VRAM.
    - *Note*: Mistral 7B QLoRA requires ~16GB VRAM.
- **RAM**: 64GB System RAM (for loading datasets).
- **Storage**: 1TB NVMe SSD (Fast loading is critical for training throughput).
