python tests/test_flash_attention.py 2>&1mkdir -p minitorch/cuda_kernels

# Original combine kernels (map / zip / reduce / batched matmul)
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC

# Flash Attention 1 kernels (forward + backward)
nvcc -o minitorch/cuda_kernels/flash_attention.so --shared src/flash_attention.cu \
    -Xcompiler -fPIC -O2
