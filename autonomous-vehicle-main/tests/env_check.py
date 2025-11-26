# env_check.py
import sys
import torch

print("✅ Python path:", sys.executable)
print("✅ Python version:", sys.version)
print("✅ PyTorch version:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())

try:
    import carla
    print("✅ CARLA import successful")
except Exception as e:
    print("❌ CARLA import failed:", e)
