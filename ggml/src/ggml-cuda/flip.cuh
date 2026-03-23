#include "common.cuh"

#define CUDA_FLIP_BLOCK_SIZE 256

void ggml_cuda_op_flip(ggml_backend_cuda_context &ctx, ggml_tensor *dst);
