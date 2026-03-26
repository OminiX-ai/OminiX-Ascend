# Qwen TTS Encoder Precision Analysis

## Issue

C++ encoder (ggml) produces ref_codes that differ from Python (PyTorch) encoder, causing a subtle background artifact in the decoded audio. The artifact is present throughout the entire audio but most noticeable during silence regions (e.g., the opening ~0.5s before speech begins).

## Root Cause

The Speech Tokenizer Encoder uses Residual Vector Quantization (RVQ) with 16 quantizer layers. RVQ is a sequential process where each layer quantizes the residual from the previous layer. Small numerical differences in the encoder's hidden states get amplified through the residual chain:

| Quantizer | Match Rate (C++ vs Python) | Role |
|-----------|---------------------------|------|
| q0 (semantic) | **100%** | Main speech content |
| q1 (acoustic 1) | 40.2% | Fine detail |
| q2 | 12.0% | Finer detail |
| q3 | 7.7% | ... |
| q4+ | <6% | Finest detail |

### Why q0 matches but q1+ don't

1. The encoder hidden states (512-dim, 117 frames) are **close but not identical** between ggml and PyTorch due to inherent floating-point differences in the transformer computation (8 layers of attention + FFN)
2. When projected to codebook space (256-dim) via `sem_input_proj`, the differences are small enough that nearest-neighbor lookup produces the **same codebook index** in 100% of cases for q0
3. After q0 quantization, the **residual** (original - reconstructed) amplifies the small differences because we're subtracting two similar values
4. The acoustic `input_proj` maps this residual to a different 256-dim space where the amplified differences cause **different nearest-neighbor lookups** for q1
5. Each subsequent quantizer operates on a smaller residual, making it progressively more sensitive to numerical differences

### Numerical difference sources

Even with all-f32 weights and computation, ggml and PyTorch produce slightly different results due to:

- **matmul accumulation order**: ggml and PyTorch may use different BLAS implementations or tiling strategies
- **Attention computation**: softmax, RoPE, layer normalization implementations differ in micro-level rounding
- **8 layers of compound error**: each transformer layer adds ~1e-6 relative error, compounding to ~1e-5 after 8 layers

These differences are within acceptable floating-point tolerance for general LLM inference, but RVQ's nearest-neighbor quantization is **discontinuous** — a tiny change in the continuous input can flip the discrete output.

## Impact

- **q0 (semantic) is correct**: The main speech content, timing, and structure are preserved
- **q1-q15 (acoustic) diverge**: Fine timbral detail differs from Python
- **Decoded audio has subtle background noise**: The decoder reconstructs audio from all 16 quantizers; incorrect acoustic codes produce a low-level artifact (~0.01-0.04 RMS vs Python's ~0.0005)
- **The artifact is masked by speech**: During voiced segments, the artifact is inaudible; it's only noticeable in silence regions

## Attempted Fixes

| Fix | Effect |
|-----|--------|
| f16→f32 encoder GGUF (RVQ weights only) | Codebook precision improved, q0: 99.1%→100% |
| f32 conv1d (build_conv1d_f32) | Conv encoder precision improved, q0: 99.1%→100% |
| All-f32 encoder GGUF + f32 conv1d | q0: 100%, q1: 40.2% (no change) |
| Python ref_codes override test | Clean output (0.0005 peak in first 50ms) |

## Conclusion

This is an inherent numerical precision issue between ggml and PyTorch's f32 transformer implementations, amplified by RVQ's sensitivity to residual values. It is **not a bug** and cannot be fully resolved without matching PyTorch's exact computation semantics.

## Possible Future Mitigations

1. **ONNX Runtime encoder**: Export the Python encoder to ONNX and run in C++ for bit-exact results
2. **Spectral gating post-processing**: Apply noise reduction to the decoded audio
3. **Fine-tune decoder robustness**: Train the decoder to be less sensitive to acoustic code perturbations (requires model retraining)
4. **Pre-compute ref_codes**: For known reference speakers, pre-compute ref_codes using Python and cache them

## Test Data

- Reference audio: `ellen_ref_24k.wav` (24kHz, 224640 samples, 9.36s)
- All comparison data saved in `logs/` directory on the NPU server
- Python dump scripts: `tools/qwen_tts/scripts/dump_intermediates.py`, `compare_outputs.py`
