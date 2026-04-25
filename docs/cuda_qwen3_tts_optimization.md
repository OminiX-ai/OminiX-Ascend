# Qwen3-TTS CUDA/GB10 Optimization Log

Date: 2026-04-24 America/Los_Angeles
Remote: `user1@163.192.33.32:6222`
Work dir: `/home/user1/qwen3_tts_cuda`

## Mission

Optimize Qwen3-TTS on NVIDIA GB10 toward an 80+ fps shipping target, using
the Ascend 910B 32.2 fps arc as the reference for architecture and gates.

## Phase 0 - Environment Baseline

Status: **YELLOW**

The GB10 environment is ready for Phase 1 llama.cpp CUDA benchmarking and
PyTorch CUDA experiments. TensorRT-LLM is installed, but its Python import is
blocked by a PyTorch internal ABI mismatch; this is the one Phase 0 item that
needs a follow-up before Phase 2 compile work can start.

### Remote Hardware / OS

- Host: `zgx-3675`
- OS: Ubuntu 24.04.4 LTS, kernel `6.17.0-1014-nvidia`
- Arch: `aarch64`
- GPU: NVIDIA GB10, driver `580.142`, CUDA driver API `13.0`
- CUDA toolkit: `/usr/local/cuda-13.0`, `nvcc 13.0.88`
- Memory: 119 GiB unified CPU/GPU memory visible to Linux
- Disk: 916 GiB root volume, ~834 GiB free at start

### Access

- SSH key auth installed with the existing local Ed25519 public key.
- Subsequent commands work with:
  `ssh -p 6222 -o BatchMode=yes user1@163.192.33.32 ...`

### Python / CUDA Environments

Main CUDA PyTorch env:

- Path: `/home/user1/qwen3_tts_cuda/.venv`
- Python: 3.12.3
- `torch==2.13.0.dev20260424+cu130`
- `torchaudio==2.11.0.dev20260424+cu130`
- `torchvision==0.27.0.dev20260424+cu130`
- CUDA smoke: `torch.cuda.is_available() == True`
- Device: `NVIDIA GB10`, capability `(12, 1)`
- BF16 GEMM smoke: 4096x4096, 20 iterations, 1.764 ms/iter
- FP8 dtype smoke: `torch.float8_e4m3fn` available

vLLM env:

- Path: `/home/user1/qwen3_tts_cuda/.venv_vllm`
- `vllm==0.19.1`
- `torch==2.10.0+cu130`
- CUDA smoke: BF16 2048x2048 GEMM, 0.192 ms/iter
- Note: PyTorch warns that this build advertises support through SM 12.0
  while GB10 is SM 12.1, but CUDA operations succeeded.

TensorRT-LLM env:

- Path: `/home/user1/qwen3_tts_cuda/.venv_trtllm_pypi`
- `tensorrt==10.14.1.48.post1`
- `tensorrt-llm==1.2.1`
- User-space OpenMPI installed via PyPI package `openmpi==5.0.10`
- Local `libpython3.12.so` symlink added under
  `/home/user1/qwen3_tts_cuda/lib`
- Current blocker:
  `ImportError: .../tensorrt_llm/libs/libth_common.so: undefined symbol: _ZN3c104impl12PyObjectSlotD1Ev`
- Attempts:
  - `torch==2.9.1+cu130`: failed on
    `_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`
  - `torch==2.10.0+cu130`: fixed the CUDA-check symbol, failed on
    `c10::impl::PyObjectSlot`
  - `torch==2.11.0+cu130`: same `PyObjectSlot` failure
- Interpretation: public aarch64 CUDA Torch wheels do not match the
  PyTorch internal ABI expected by the NVIDIA TensorRT-LLM 1.2.1 wheel.
  Phase 2 likely needs either an NVIDIA NGC-matched stack, a source build
  against the selected Torch wheel, or a pinned TRT-LLM version with a
  matching public aarch64 Torch ABI.

### llama.cpp CUDA

- Repo: `/home/user1/qwen3_tts_cuda/llama.cpp`
- Commit: `0adede8`
- Build: `/home/user1/qwen3_tts_cuda/llama.cpp/build-cuda`
- CMake flags:
  - `-DGGML_CUDA=ON`
  - `-DCMAKE_CUDA_ARCHITECTURES=121`
  - `-DCMAKE_BUILD_TYPE=Release`
  - `-DLLAMA_BUILD_TESTS=OFF`
- Built targets: `llama-cli`, `llama-bench`
- Device enumeration:
  `NVIDIA GB10, compute capability 12.1, VMM: yes, VRAM: 122502 MiB`

Small smoke on Q8_0 talker GGUF with Flash Attention:

```text
model: gguf_q8_0/qwen3_tts_talker.gguf
model_type: qwen3vl 1.7B Q8_0
ngl: 99
flash_attn: true
n_prompt=16: 1169.78 tok/s
n_gen=16: 132.916 tok/s
```

This is only a load/execution smoke, not the canonical TTS fps benchmark.
Phase 1 must run the actual talker generation path and report frame/audio
wall-clock.

### Weights / Assets

Downloaded from `cgisky/qwen3-tts-custom-gguf`:

- `/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_talker.gguf`
  - 1,511,314,656 bytes
- `/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_predictor.gguf`
  - 151,124,320 bytes
- `/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_assets.gguf`
  - 406,374,528 bytes
- `/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/onnx/qwen3_tts_decoder.onnx`
  - 456,760,558 bytes
- `/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/tokenizer/tokenizer.json`
  - 11,423,986 bytes
- Preset speaker JSONs under `preset_speakers/`

Note: the model card mentions `onnx_int8`, but the repo file list currently
contains only `onnx/` files.

## Next Phase Scope

Phase 1 should produce the first real CUDA number to beat:

1. Identify the lightest working Qwen3-TTS C++/Python runner compatible with
   the downloaded GGUF package.
2. Run canonical prompts matching the Ascend reports as closely as possible.
3. Report audio frames/sec, token wall-clock, generated frame count, and an
   eye/ear-check verdict.
4. Keep llama.cpp measurements separate from full TTS pipeline measurements,
   because the smoke above only validates decoder-token execution.

TensorRT-LLM follow-up before Phase 2:

1. Check whether an NVIDIA NGC PyTorch/TensorRT-LLM wheel stack is available
   without Docker, or whether the user-space machine allows container runtime.
2. If not, source-build TensorRT-LLM against the selected CUDA Torch aarch64
   wheel.
3. Re-run import smoke and a minimal Qwen/Qwen2 builder smoke before investing
   in Qwen3-TTS graph conversion.
