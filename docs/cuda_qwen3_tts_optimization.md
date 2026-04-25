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

## Phase 2 - vLLM Pivot

Date: 2026-04-25 America/Los_Angeles

Status: **YELLOW/GREEN**

The TRT-LLM path remains blocked on PyTorch internal ABI mismatches, so Phase 2
pivoted to vLLM 0.19.1. vLLM can load and serve the Talker GGUF through a
derived Qwen3 HF config, and CUDA graph mode reaches llama.cpp-class raw decode
throughput on GB10. Full end-to-end TTS fps is still pending because the
current local asset bundle has the decoder ONNX and GGUF assets, but no ready
Python runner or complete ONNX encoder/speaker-encoder path to wire Talker
tokens into audio.

### Setup Fixes

- `vllm==0.19.1` imported only after adding CUDA 12 runtime libraries inside
  `.venv_vllm`:
  `pip install nvidia-cuda-runtime-cu12`
- Runtime library path used for vLLM:
  `LD_LIBRARY_PATH=.venv_vllm/.../torch/lib:.venv_vllm/.../nvidia/cuda_runtime/lib:.venv_vllm/.../nvidia/cu13/lib`
- Torch/Triton helper compilation needed Python development headers. `sudo`
  was not available non-interactively, so headers were extracted into:
  `/home/user1/qwen3_tts_cuda/sysroot/python312-dev`
- Compile path uses:
  `CPATH=/home/user1/qwen3_tts_cuda/sysroot/python312-dev/usr/include:/home/user1/qwen3_tts_cuda/sysroot/python312-dev/usr/include/python3.12`

### Model Loading Path

Direct GGUF loading failed because the Talker GGUF reports architecture
`qwen3vl`, which vLLM/Transformers did not accept from GGUF metadata:

```text
ValueError: GGUF model with architecture qwen3vl is not supported yet.
```

The official HF config for `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` uses a
Qwen3-TTS model type rather than a plain vLLM-supported Qwen3 CausalLM config,
so the run used a derived HF config for the GGUF Talker:

- Config path:
  `/home/user1/qwen3_tts_cuda/vllm_hf/qwen3_tts_talker_gguf_config/config.json`
- Shape:
  `vocab_size=3072`, `hidden_size=2048`, `intermediate_size=6144`,
  `num_hidden_layers=28`, `num_attention_heads=16`,
  `num_key_value_heads=8`, `head_dim=128`,
  `max_position_embeddings=32768`, `rope_theta=1000000.0`
- Architecture override: `Qwen3ForCausalLM`

vLLM's early speculator probe still tried to parse the GGUF before
`--hf-config-path` was applied, so the CLI entrypoint is wrapped by:

```text
/home/user1/qwen3_tts_cuda/scripts/vllm_qwen3_tts_cli.py
```

This wrapper skips the early speculator probe and then delegates to vLLM's
normal CLI. The model must be driven with raw Talker token IDs. Text tokenizer
IDs are invalid for this 3072-token audio-code vocabulary; for example,
`"Hello"` encodes to token ID `9707`, which is out of range for the Talker
embedding table.

### Serve Commands

CUDA graph FP16 server:

```bash
V=/home/user1/qwen3_tts_cuda/.venv_vllm
SYS=/home/user1/qwen3_tts_cuda/sysroot/python312-dev/usr/include
. "$V/bin/activate"
export LD_LIBRARY_PATH="$V/lib/python3.12/site-packages/torch/lib:$V/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$V/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$SYS:$SYS/python3.12:${CPATH:-}"
export VLLM_NO_USAGE_STATS=1

python /home/user1/qwen3_tts_cuda/scripts/vllm_qwen3_tts_cli.py serve \
  /home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_talker.gguf \
  --served-model-name qwen3-tts-talker \
  --host 127.0.0.1 \
  --port 18000 \
  --hf-config-path /home/user1/qwen3_tts_cuda/vllm_hf/qwen3_tts_talker_gguf_config \
  --skip-tokenizer-init \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.50
```

FP8 KV cache variant adds:

```bash
--kv-cache-dtype fp8
```

Eager control adds:

```bash
--enforce-eager
```

API smoke request:

```json
{
  "model": "qwen3-tts-talker",
  "prompt": [1],
  "max_tokens": 16,
  "temperature": 0,
  "stop_token_ids": [],
  "return_token_ids": true
}
```

Smoke output token IDs:

```text
[2157, 2150, 498, 498, 498, 498, 1311, 1489, 1489, 1489, 1489, 1247, 999, 613, 613, 613]
```

### Talker Microbenchmarks

Benchmark script:

```text
/home/user1/qwen3_tts_cuda/scripts/vllm_qwen3_tts_api_bench.py
```

Raw-token benchmark settings:

- Prompt token IDs: `[1]`
- Decode length: 128 tokens for throughput
- Streaming probe: 64 tokens
- `temperature=0`
- `ignore_eos=True`
- `stop_token_ids=[]`
- `return_token_ids=True`
- `max_num_seqs=1`
- `max_num_batched_tokens=512`

These are Talker-token API measurements, not complete audio synthesis fps.

| Path | Mean decode | Warm stream TTFT | Notes |
| --- | ---: | ---: | --- |
| llama.cpp Q8_0 CUDA baseline | 132.916 tok/s | n/a | Phase 1 smoke, `n_gen=16` |
| vLLM eager FP16 GGUF | 110.03 tok/s | 49.99 ms | Control path, no CUDA graphs |
| vLLM CUDA graph FP16 GGUF | 131.78 tok/s | 31.69 ms | 99.1% of llama.cpp baseline |
| vLLM CUDA graph + FP8 KV | 130.95 tok/s | 11.80 ms | 98.5% of llama.cpp baseline |
| Ascend shipping reference | 32.2 fps | n/a | Full TTS reference, not token-only |
| MLX equivalent | not measured | n/a | No equivalent run available |

Best current vLLM Talker result:

```text
vLLM graph FP16: 131.78 tok/s
vs llama.cpp baseline: 132.916 tok/s
vs vLLM eager: +19.8%
vs Ascend 32.2 fps reference: about 4.1x if compared only as raw Talker token rate
```

The Ascend comparison is directional only until the codec path is wired and
measured as real synthesized-audio fps.

### Memory / Compile Receipts

Eager FP16 server log:

```text
/home/user1/qwen3_tts_cuda/logs/vllm_serve_talker_eager_18000.log
```

- Model loading: 1.43 GiB, 0.805 s
- Available KV cache memory: 56.77 GiB
- GPU KV cache size: 531,488 tokens
- Engine init: 2.98 s

CUDA graph FP16 server log:

```text
/home/user1/qwen3_tts_cuda/logs/vllm_serve_talker_graph_18000.log
```

- Model loading: 1.43 GiB, 0.840 s
- `torch.compile`: 13.18 s
- Estimated CUDA graph memory: 0.07 GiB
- Available KV cache memory: 56.41 GiB
- GPU KV cache size: 528,144 tokens
- Engine init: 17.98 s

CUDA graph + FP8 KV server log:

```text
/home/user1/qwen3_tts_cuda/logs/vllm_serve_talker_graph_kvfp8_18000.log
```

- Attention backend: FLASHINFER
- Model loading: 1.43 GiB, 0.849 s
- `torch.compile`: 4.61 s, with compile-cache reuse
- Estimated CUDA graph memory: 2.35 GiB
- Available KV cache memory: 54.78 GiB
- GPU KV cache size: 1,025,680 tokens
- Engine init: 23.17 s
- Saved benchmark JSON:
  `/home/user1/qwen3_tts_cuda/logs/vllm_talker_graph_kvfp8_api_bench.json`

FP8 KV nearly doubles KV capacity versus FP16 KV at this sequence cap
(`1,025,680 / 528,144 = 1.94x`) but did not improve single-stream throughput
for the 128-token decode probe.

### Lever Delta

| Lever | Result |
| --- | --- |
| CUDA 12 runtime in vLLM venv | Unblocked vLLM import against `libcudart.so.12` |
| User-space Python headers | Unblocked Triton/Inductor helper builds |
| Derived Qwen3 HF config | Unblocked GGUF model load despite `qwen3vl` GGUF architecture tag |
| Raw prompt token IDs | Avoided out-of-range tokenizer IDs for Talker vocab |
| CUDA graphs / vLLM compile | 110.03 -> 131.78 tok/s, +19.8% |
| FP8 KV cache | 528k -> 1.026M KV tokens, no speed gain in single stream |
| FP8 weights | Not applied; current checkpoint is GGUF Q8_0 and vLLM reports `quantization=gguf` |
| Single-stream batch tuning | `max_num_seqs=1`, `max_num_batched_tokens=512`, `max_model_len=512` |

### Codec Wiring Status

The intended low-latency path is:

```text
vLLM token stream -> audio-code token buffer -> codec decoder on GPU -> audio
```

The vLLM side is ready: `/v1/completions` accepts list-of-int prompts and can
stream generated token IDs. The codec side was not completed in this Phase 2
run because the available local bundle contains `qwen3_tts_decoder.onnx` plus
GGUF assets, but no ready Qwen3-TTS pipeline runner was present, and the model
card's broader ONNX encoder/speaker-encoder layout was not present in the
downloaded files. Next work is to fetch the complete official HF asset set or
implement the bridge from `qwen3_assets.gguf`/predictor output into the ONNX
decoder input contract.

### Next Measurements

To close Phase 2 as an end-to-end TTS result:

1. Fetch or reconstruct the complete codec path.
2. Use a canonical prompt and reference speaker audio.
3. Record total synthesis wall time, first-token latency, first-audio latency,
   steady-state audio fps, Talker decode tok/s, and peak GPU memory.
4. Compare complete TTS fps against Ascend 32.2 fps and any available MLX
   result. Keep the current 131.78 tok/s Talker number as a raw-token ceiling,
   not as an audio fps claim.
