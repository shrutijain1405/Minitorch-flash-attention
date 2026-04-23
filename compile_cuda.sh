mkdir -p minitorch/cuda_kernels
nvcc -arch=native -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
nvcc -arch=native -o minitorch/cuda_kernels/flash_attention_forward.so --shared src/flash_attention_forward.cu -Xcompiler -fPIC
nvcc -arch=native -o minitorch/cuda_kernels/flash_attention_backward.so --shared src/flash_attention_backward.cu -Xcompiler -fPIC
