import torch
import time

print("=== GPU DIAGNOSTIC ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved(0)/1024**2:.1f} MB")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("❌ CUDA not available - running on CPU")

# Test a simple GPU operation
try:
    start_time = time.time()
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    end_time = time.time()
    print(f"✓ GPU matrix multiplication test: {(end_time - start_time)*1000:.1f} ms")
except Exception as e:
    print(f"❌ GPU test failed: {e}")