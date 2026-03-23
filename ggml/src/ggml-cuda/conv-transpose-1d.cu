#include "conv-transpose-1d.cuh"

// Original naive kernel (reference implementation)
static __global__ void conv_transpose_1d_kernel_naive(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, const float * src1,  float * dst) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;

    float accumulator = 0;

    for (int c = 0; c < src0_ne2; c++) {
        int idx = global_index % dst_ne0;

        int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        int input_offset = src1_ne0 * c;

        for (int i = 0; i < src1_ne0; i++) {
            if (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                continue;
            }
            int weight_idx = idx - i*s0;

            float kernel_weight = src0[kernel_offset + weight_idx];
            float input_value = src1[input_offset + i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
    GGML_UNUSED_VARS(p0, d0, src0_ne3, src1_ne3, dst_ne3, src1_ne1, dst_ne1, src1_ne2, dst_ne2);
}

// Optimized kernel for HiFi-GAN upsampling patterns
// Layout: kernel[k + out_ch * K + ic * (K * OC)], input[in_pos + ic * L], output[out_pos * OC + out_ch]
static __global__ void conv_transpose_1d_kernel_optimized(
        const int stride,
        const int kernel_size,      // src0_ne0
        const int out_channels,     // src0_ne1
        const int in_channels,      // src0_ne2
        const int input_length,     // src1_ne0
        const int output_length,    // dst_ne0
        const float * __restrict__ kernel,
        const float * __restrict__ input,
        float * __restrict__ output) {

    // Each block handles one output position, each thread handles one output channel
    const int out_pos = blockIdx.x;
    const int out_ch = threadIdx.x;

    if (out_pos >= output_length || out_ch >= out_channels) {
        return;
    }

    float acc = 0.0f;

    // For each input channel
    for (int ic = 0; ic < in_channels; ic++) {
        // For each input position that contributes to this output position
        // out_pos = in_pos * stride + k, where 0 <= k < kernel_size
        // So: in_pos = (out_pos - k) / stride, valid when (out_pos - k) % stride == 0

        for (int k = 0; k < kernel_size; k++) {
            const int tmp = out_pos - k;
            if (tmp < 0 || tmp % stride != 0) {
                continue;
            }
            const int in_pos = tmp / stride;
            if (in_pos >= input_length) {
                continue;
            }

            // kernel index: k + out_ch * kernel_size + ic * (kernel_size * out_channels)
            const int kernel_idx = k + out_ch * kernel_size + ic * (kernel_size * out_channels);
            // input index: in_pos + ic * input_length
            const int input_idx = in_pos + ic * input_length;

            acc += kernel[kernel_idx] * input[input_idx];
        }
    }

    // output index: out_pos + out_ch * output_length (GGML layout: positions vary fastest)
    output[out_pos + out_ch * output_length] = acc;
}

// Further optimized: iterate over input positions instead of kernel positions
// Better for cases where kernel_size >> stride (common in HiFi-GAN: kernel=16, stride=8)
static __global__ void conv_transpose_1d_kernel_fast(
        const int stride,
        const int kernel_size,
        const int out_channels,
        const int in_channels,
        const int input_length,
        const int output_length,
        const float * __restrict__ kernel,
        const float * __restrict__ input,
        float * __restrict__ output) {

    const int out_pos = blockIdx.x;
    const int out_ch = threadIdx.x;

    if (out_pos >= output_length || out_ch >= out_channels) {
        return;
    }

    float acc = 0.0f;

    // Calculate which input positions contribute to this output
    // out_pos = in_pos * stride + k, so in_pos = (out_pos - k) / stride
    // Valid range: in_pos from max(0, ceil((out_pos - kernel_size + 1) / stride)) to min(input_length-1, out_pos / stride)
    const int in_start = max(0, (out_pos - kernel_size + stride) / stride);
    const int in_end = min(input_length, out_pos / stride + 1);

    for (int in_pos = in_start; in_pos < in_end; in_pos++) {
        const int k = out_pos - in_pos * stride;
        // k should be in [0, kernel_size), but double-check
        if (k < 0 || k >= kernel_size) continue;

        // Accumulate over all input channels
        for (int ic = 0; ic < in_channels; ic++) {
            const int kernel_idx = k + out_ch * kernel_size + ic * (kernel_size * out_channels);
            const int input_idx = in_pos + ic * input_length;
            acc += kernel[kernel_idx] * input[input_idx];
        }
    }

    // output index: out_pos + out_ch * output_length (GGML layout: positions vary fastest)
    output[out_pos + out_ch * output_length] = acc;
}

static void conv_transpose_1d_f32_f32_cuda(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, const float * src1,  float * dst,
        cudaStream_t stream) {

    const int kernel_size = src0_ne0;
    const int out_channels = src0_ne1;
    const int in_channels = src0_ne2;
    const int input_length = src1_ne0;
    const int output_length = dst_ne0;

    // Use optimized kernel when out_channels fits in a single block
    if (out_channels <= 1024) {
        // Use fast kernel: one block per output position, one thread per channel
        conv_transpose_1d_kernel_fast<<<output_length, out_channels, 0, stream>>>(
            s0, kernel_size, out_channels, in_channels,
            input_length, output_length,
            src0, src1, dst);
    } else {
        // Fallback to naive kernel for large channel counts
        const int num_blocks = (output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
        conv_transpose_1d_kernel_naive<<<num_blocks, CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(
            s0, p0, d0, output_size,
            src0_ne0, src0_ne1, src0_ne2, src0_ne3,
            src1_ne0, src1_ne1, src1_ne2, src1_ne3,
            dst_ne0, dst_ne1, dst_ne2, dst_ne3,
            src0, src1, dst);
    }

    GGML_UNUSED_VARS(p0, d0, src0_ne3, src1_ne2, src1_ne3, dst_ne1, dst_ne2, dst_ne3, src1_ne1);
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;

    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = opts[0];
    const int p0 = 0;//opts[3];
    const int d0 = 1;//opts[4];

    const int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_cuda(s0, p0, d0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        src0_d, src1_d, dst_d, stream);
}
