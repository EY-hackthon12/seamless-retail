# 🔥 Power Tools

**Super scripts for Seamless Retail** - launch the app even without downloaded models.

---

## Scripts

### `super_launcher.py`

The ultimate launcher that just works, no matter what.

```bash
# Full launch (with graceful degradation)
python power_tools/super_launcher.py

# Dry-run mode (test without starting services)
python power_tools/super_launcher.py --dry-run

# Quick tests only (verify imports work)
python power_tools/super_launcher.py --test
```

**Features:**
- 🖥️ **Hardware Auto-Detection** - Detects CPU/GPU and classifies tier
- 📦 **Model Status Check** - Shows which models are available
- ⚡ **Graceful Degradation** - Runs with stub responses if models missing
- 🔄 **Service Management** - Start/stop all services with Ctrl+C

---

## Model-Free Operation

When models are not downloaded, the super launcher provides **stub responses**:

| Component | Stub Behavior |
|-----------|---------------|
| LLM Responses | Returns placeholder text explaining models aren't loaded |
| Sales Prediction | Returns random values with confidence = 0 |
| RAG Search | Returns empty results |
| Intent Classification | Defaults to "general_chat" |

---

## Hardware Tiers

The launcher detects your hardware and classifies it:

| Tier | VRAM | Description |
|------|------|-------------|
| CPU_ONLY | N/A | No GPU available |
| LOW_VRAM | <8GB | Basic GPU acceleration |
| CONSUMER | 8-12GB | RTX 3060/4060 class |
| PROSUMER | 12-24GB | RTX 4080/4090 class |
| DATACENTER | 24GB+ | A100, H100 class |

---

## Quick Test

Verify your environment is ready:

```bash
python power_tools/super_launcher.py --test
```

Expected output:
```
🧪 QUICK TESTS (No Models Required)
====================================
  ✅ PyTorch 2.x.x
  ✅ FAISS available
  ✅ Neural architectures importable
  ✅ Model stubs functional
  ✅ Hardware detection: CONSUMER
====================================
  Results: 5/5 tests passed
```

---

## Related Scripts

- `scripts/testing/test_all_trainable.py` - Test all trainable components
- `scripts/brain/serve_brain.py` - Brain prediction server
- `cognitive_brain/api.py` - Cognitive Brain API
