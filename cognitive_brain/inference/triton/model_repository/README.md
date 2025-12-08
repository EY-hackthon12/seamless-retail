# NVIDIA Triton Model Repository
# ================================
# This directory contains vision and ensemble models for Triton Inference Server.
#
# Structure:
#   model_repository/
#   ├── yolov9_shelf/         # ONNX model for shelf analysis
#   │   ├── 1/model.onnx
#   │   └── config.pbtxt
#   ├── clip_visual/          # CLIP for visual search
#   │   ├── 1/model.onnx
#   │   └── config.pbtxt
#   └── ensemble_retail/      # Ensemble combining models
#       └── config.pbtxt
#
# Usage:
#   docker run --gpus=1 -p8000:8000 -p8001:8001 -p8002:8002 \
#     -v $(pwd)/model_repository:/models \
#     nvcr.io/nvidia/tritonserver:24.08-py3 \
#     tritonserver --model-repository=/models
