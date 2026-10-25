import torch

print("PyTorch version:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

    # Create random matrices directly in GPU memory
    a = torch.rand(2000, 2000, device="cuda")
    b = torch.rand(2000, 2000, device="cuda")

    # Perform matrix multiplication on GPU
    result = torch.matmul(a, b)

    # Wait for GPU computation to finish
    torch.cuda.synchronize()

    print("GPU calculation completed successfully!")
    print("Result shape:", result.shape)

else:
    print("ERROR: CUDA acceleration unavailable!")