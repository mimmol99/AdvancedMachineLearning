import torch
import hydra
import mlflow
import subprocess
import sys
print("-" * 50)
print("🛠️  SYSTEM DIAGNOSTICS")
print("-" * 50)

# 1. CHECK CUDA
if torch.cuda.is_available():
    print(f"Python version: {sys.version}")
    print(f"✅ CUDA AVAILABLE: Yes")
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🧠 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"📦 PyTorch Version: {torch.__version__}")
    
    # Test allocazione memoria
    x = torch.rand(5, 5).cuda()
    print(f"🔢 Tensor Test: Allocated on {x.device}")
else:
    #run nvidia-smi to check GPU status
    result = subprocess.run("nvidia-smi", shell=True)
    print(f"{result.stdout}")
    print("❌ CUDA NOT AVAILABLE (Using CPU)")

# 2. CHECK LIBRARY VERSIONS
print("-" * 50)
print(f"📚 Hydra Version: {hydra.__version__}")
print(f"📉 MLflow Version: {mlflow.__version__}")
print("-" * 50)
