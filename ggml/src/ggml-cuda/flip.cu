// #include "flip.cuh"

// static __global__ void flip_f32(const float *__restrict__ src,
//                                 float *__restrict__ dst,
//                                 const int64_t total_elems, int axis,
//                                 const int64_t *__restrict__ ne,
//                                 const int64_t *__restrict__ strides) {
//   int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx >= total_elems)
//     return;

//   // 拆线性 index → 多维
//   int64_t rem = idx;
//   int64_t src_idx = 0;

//   for (int d = 0; d < GGML_MAX_DIMS; ++d) {
//     int64_t coord = rem % ne[d];
//     rem /= ne[d];

//     if (d == axis) {
//       coord = ne[d] - 1 - coord;
//     }

//     src_idx += coord * strides[d];
//   }

//   dst[idx] = src[src_idx];
// }

// static __global__ void flip_f32(const float *__restrict__ src,
//                                 float *__restrict__ dst,
//                                 const int64_t total_elems, int axis,
//                                 int64_t ne0, int64_t ne1, int64_t ne2,
//                                 int64_t ne3) {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (tid >= total_elems)
//     return;

//   int idx = tid;
//   int src_offset = 0;
//   int stride = 1;

//   if (ne0 > 0) {
//     int coord = idx % ne0;
//     idx /= ne0;
//     if (0 == axis) {
//       coord = ne0 - 1 - coord;
//     }
//     src_offset += coord * stride;
//     stride *= ne0;
//   }
//   if (ne1 > 0) {
//     int coord = idx % ne1;
//     idx /= ne1;
//     if (1 == axis) {
//       coord = ne1 - 1 - coord;
//     }
//     src_offset += coord * stride;
//     stride *= ne1;
//   }
//   if (ne2 > 0) {
//     int coord = idx % ne2;
//     idx /= ne2;
//     if (2 == axis) {
//       coord = ne2 - 1 - coord;
//     }
//     src_offset += coord * stride;
//     stride *= ne2;
//   }
//   if (ne3 > 0) {
//     int coord = idx % ne3;
//     idx /= ne3;
//     if (3 == axis) {
//       coord = ne3 - 1 - coord;
//     }
//     src_offset += coord * stride;
//     stride *= ne3;
//   }
//   dst[tid] = src[src_offset];
// }

// void ggml_cuda_op_flip(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
//   const int32_t dim = ((int32_t *)dst->op_params)[0];

//   const ggml_tensor *src0 = dst->src[0];
//   GGML_ASSERT(src0->type == GGML_TYPE_F32);
//   GGML_ASSERT(dst->type == GGML_TYPE_F32);
//   GGML_ASSERT(ggml_is_contiguous(src0));
//   GGML_ASSERT(ggml_is_contiguous(dst));

//   const float *src0_d = (const float *)src0->data;
//   float *dst_d = (float *)dst->data;

//   cudaStream_t stream = ctx.stream();

//   const int64_t ne0 = src0->ne[0];
//   const int64_t ne1 = src0->ne[1];
//   const int64_t ne2 = src0->ne[2];
//   const int64_t ne3 = src0->ne[3];

//   int total_elems = ggml_nelements(src0);

//   int threads = 256;
//   int blocks = (total_elems + threads - 1) / threads;

//   flip_f32<<<blocks, threads, 0, stream>>>(src0_d, dst_d, total_elems,
//                                                   dim, ne0, ne1, ne2, ne3);
// }



#include "flip.cuh"

static __global__ void flip_f32_cuda(const float * __restrict__ src,
                                     float * __restrict__ dst,
                                     const int64_t ne00,
                                     const int64_t ne01,
                                     const int64_t ne02,
                                     const int64_t ne03,
                                     const int     dims) {
    const int64_t idx        = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t n_elements = ne00 * ne01 * ne02 * ne03;

    if (idx >= n_elements) {
        return;
    }
    // Calculate output indices
    const int64_t i0 = idx % ne00;
    const int64_t i1 = (idx / ne00) % ne01;
    const int64_t i2 = (idx / (ne00 * ne01)) % ne02;
    const int64_t i3 = idx / (ne00 * ne01 * ne02);

    // Calculate source indices (flipped if needed) - dims is axis index (0..3)
    const int64_t src_i0 = (dims == 0) ? (ne00 - 1 - i0) : i0;
    const int64_t src_i1 = (dims == 1) ? (ne01 - 1 - i1) : i1;
    const int64_t src_i2 = (dims == 2) ? (ne02 - 1 - i2) : i2;
    const int64_t src_i3 = (dims == 3) ? (ne03 - 1 - i3) : i3;

    // Calculate memory offsets
    const int64_t dst_offset = i3 * (ne00 * ne01 * ne02) + i2 * (ne01 * ne00) + i1 * ne00 + i0;
    const int64_t src_offset = src_i3 * (ne00 * ne01 * ne02) + src_i2 * (ne01 * ne00) + src_i1 * ne00 + src_i0;

    dst[dst_offset] = src[src_offset];
}

void ggml_cuda_op_flip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    int dims = dst->op_params[0];

    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) dst->src[0]->data;
    float *             dst_d  = (float *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(dst->src[0], dst));

    cudaStream_t stream = ctx.stream();

    int64_t sz         = (ne00 * ne01 * ne02 * ne03);
    int64_t num_blocks = (sz + CUDA_FLIP_BLOCK_SIZE - 1) / CUDA_FLIP_BLOCK_SIZE;

    flip_f32_cuda<<<num_blocks, CUDA_FLIP_BLOCK_SIZE, 0, stream>>>(
        src0_d, dst_d, ne00, ne01, ne02, ne03, dims);
}
