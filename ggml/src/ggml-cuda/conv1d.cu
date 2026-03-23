#include "conv1d.cuh"

// Optimized direct conv1d kernel
// GGML tensor layout (ne[0] is contiguous/innermost dimension):
// kernel: [K, IC, OC] means ne[0]=K, ne[1]=IC, ne[2]=OC
//         GGML index: k + ic * K + oc * (K * IC)
// input:  [L, IC, N]  means ne[0]=L, ne[1]=IC, ne[2]=N
//         GGML index: l + ic * L + n * (L * IC)
// output: [OL, OC, N] means ne[0]=OL, ne[1]=OC, ne[2]=N
//         GGML index: ol + oc * OL + n * (OL * OC)

// Each thread computes one output element
static __global__ void conv_1d_kernel(
        const int stride,
        const int padding,
        const int dilation,
        const int kernel_size,
        const int in_channels,
        const int out_channels,
        const int input_length,
        const int output_length,
        const int batch_size,
        const float * __restrict__ kernel,
        const float * __restrict__ input,
        float * __restrict__ output) {

    const int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int total_output = output_length * out_channels * batch_size;

    if (global_idx >= total_output) {
        return;
    }

    // Decode output index for GGML layout [OL, OC, N]
    // index = ol + oc * OL + n * (OL * OC)
    const int out_pos = global_idx % output_length;
    const int out_ch = (global_idx / output_length) % out_channels;
    const int batch = global_idx / (output_length * out_channels);

    float acc = 0.0f;

    // Convolution: sum over kernel positions and input channels
    for (int k = 0; k < kernel_size; k++) {
        const int in_pos = out_pos * stride + k * dilation - padding;

        // Check bounds
        if (in_pos < 0 || in_pos >= input_length) {
            continue;
        }

        // Sum over input channels
        // kernel GGML layout [K, IC, OC]: index = k + ic * K + oc * (K * IC)
        // input GGML layout [L, IC, N]: index = in_pos + ic * L + batch * (L * IC)
        for (int ic = 0; ic < in_channels; ic++) {
            const int kernel_idx = k + ic * kernel_size + out_ch * (kernel_size * in_channels);
            const int input_idx = in_pos + ic * input_length + batch * (input_length * in_channels);
            acc += kernel[kernel_idx] * input[input_idx];
        }
    }

    // output GGML layout [OL, OC, N]: index = out_pos + out_ch * OL + batch * (OL * OC)
    const int output_idx = out_pos + out_ch * output_length + batch * (output_length * out_channels);
    output[output_idx] = acc;
}

// Optimized kernel for HiFi-GAN ResBlock pattern: kernel_size=3, stride=1, dilation varies
static __global__ void conv_1d_kernel_k3(
        const int padding,
        const int dilation,
        const int kernel_size,  // should be 3
        const int in_channels,
        const int out_channels,
        const int input_length,
        const int output_length,
        const int batch_size,
        const float * __restrict__ kernel,
        const float * __restrict__ input,
        float * __restrict__ output) {

    const int out_pos = blockIdx.x;
    const int out_ch = threadIdx.x;

    if (out_pos >= output_length || out_ch >= out_channels) {
        return;
    }

    // Process all batches for this output position and channel
    for (int batch = 0; batch < batch_size; batch++) {
        float acc = 0.0f;

        // Unrolled loop for kernel_size=3
        #pragma unroll
        for (int k = 0; k < 3; k++) {
            const int in_pos = out_pos + k * dilation - padding;

            if (in_pos >= 0 && in_pos < input_length) {
                // GGML layout indexing
                // kernel [K, IC, OC]: index = k + ic * K + oc * (K * IC)
                // input [L, IC, N]: index = in_pos + ic * L + batch * (L * IC)
                int ic = 0;
                for (; ic + 3 < in_channels; ic += 4) {
                    acc += kernel[k + ic * kernel_size + out_ch * (kernel_size * in_channels)] *
                           input[in_pos + ic * input_length + batch * (input_length * in_channels)];
                    acc += kernel[k + (ic+1) * kernel_size + out_ch * (kernel_size * in_channels)] *
                           input[in_pos + (ic+1) * input_length + batch * (input_length * in_channels)];
                    acc += kernel[k + (ic+2) * kernel_size + out_ch * (kernel_size * in_channels)] *
                           input[in_pos + (ic+2) * input_length + batch * (input_length * in_channels)];
                    acc += kernel[k + (ic+3) * kernel_size + out_ch * (kernel_size * in_channels)] *
                           input[in_pos + (ic+3) * input_length + batch * (input_length * in_channels)];
                }
                for (; ic < in_channels; ic++) {
                    acc += kernel[k + ic * kernel_size + out_ch * (kernel_size * in_channels)] *
                           input[in_pos + ic * input_length + batch * (input_length * in_channels)];
                }
            }
        }

        // output [OL, OC, N]: index = out_pos + out_ch * OL + batch * (OL * OC)
        output[out_pos + out_ch * output_length + batch * (output_length * out_channels)] = acc;
    }
}

static void conv_1d_f32_f32_cuda(
        const int stride, const int padding, const int dilation,
        const int kernel_size, const int in_channels, const int out_channels,
        const int input_length, const int output_length, const int batch_size,
        const float * kernel, const float * input, float * output,
        cudaStream_t stream) {

    const int total_output = output_length * out_channels * batch_size;

    // Choose kernel based on parameters
    if (kernel_size == 3 && stride == 1 && out_channels <= 512) {
        // Use specialized kernel_size=3 kernel for HiFi-GAN ResBlock
        conv_1d_kernel_k3<<<output_length, out_channels, 0, stream>>>(
            padding, dilation, kernel_size, in_channels, out_channels,
            input_length, output_length, batch_size,
            kernel, input, output);
    } else if (out_channels <= 512) {
        // Use general kernel with one thread per output channel
        const int num_blocks = (total_output + CUDA_CONV1D_BLOCK_SIZE - 1) / CUDA_CONV1D_BLOCK_SIZE;
        conv_1d_kernel<<<num_blocks, CUDA_CONV1D_BLOCK_SIZE, 0, stream>>>(
            stride, padding, dilation,
            kernel_size, in_channels, out_channels,
            input_length, output_length, batch_size,
            kernel, input, output);
    } else {
        // General case
        const int num_blocks = (total_output + CUDA_CONV1D_BLOCK_SIZE - 1) / CUDA_CONV1D_BLOCK_SIZE;
        conv_1d_kernel<<<num_blocks, CUDA_CONV1D_BLOCK_SIZE, 0, stream>>>(
            stride, padding, dilation,
            kernel_size, in_channels, out_channels,
            input_length, output_length, batch_size,
            kernel, input, output);
    }
}

void ggml_cuda_op_conv_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // kernel
    const float * src0_d = (const float *)src0->data;

    const ggml_tensor * src1 = dst->src[1];  // input
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int stride = opts[0];
    const int padding = opts[1];
    const int dilation = opts[2];

    // src0 (kernel): [K, IC, OC]
    const int kernel_size = src0->ne[0];
    const int in_channels = src0->ne[1];
    const int out_channels = src0->ne[2];

    // src1 (input): [L, IC, N]
    const int input_length = src1->ne[0];
    const int batch_size = src1->ne[2];

    // dst (output): [OL, OC, N]
    const int output_length = dst->ne[0];

    conv_1d_f32_f32_cuda(
        stride, padding, dilation,
        kernel_size, in_channels, out_channels,
        input_length, output_length, batch_size,
        src0_d, src1_d, dst_d, stream);
}
