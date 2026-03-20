# OminiX-Ascend 性能对比

## DeepSeek OCR 2

| 模型 | Backend | 速度 |
|------|---------|------|
| Deepseek OCR 2 | Ours | **33 t/s** |
| Deepseek OCR 2 | Torch-npu | 5 t/s |

## Qwen3 & Qwen3.5

Qwen3.5 为新模型，torch-npu 支持不太好，因此加速比更高。

| 模型 | Torch-npu | Our framework |
|------|-----------|---------------|
| Qwen3-8B | 28 t/s | **42 t/s** |
| Qwen3.5-9B | 8.9 t/s | **28 t/s** |

## 大模型对比（vs vLLM）

vLLM 的支持不是非常稳定（新模型不太支持），不过速度会比 torch-npu 平均高一些。

| 模型 | Our framework | vLLM |
|------|---------------|------|
| Qwen3-32B | **10 t/s** | 显存不够 |
| GLM-4-32B | **10 t/s** | 显存不够 |
| Qwen3-14B | 21 t/s | 23 t/s |
| Ministral-3-14B | **21 t/s** | 4 t/s |

## ggml 自动优化加速（Qwen3-8B）

通过自动 compiler 优化和算子融合（完全自动化），加速超过 **50%**：

| 环境 | 精度/格式 | 框架/配置 | Generate 速度 |
|------|-----------|-----------|---------------|
| CANN 8.3 + torch_npu 2.8 | BF16 | 原生推理 | 11 t/s |
| CANN 8.3 | BF16 | vllm-ascend | 28 t/s |
| CANN 8.5 | Q8_0 | ggml Baseline | 27.15 t/s |
| CANN 8.5 | Q8_0 | ggml V cache 优化 | 37.70 t/s |
| CANN 8.5 | Q8_0 | ggml FA + Fusion（推荐） | **42.82 t/s** |

## Qwen Image 2512（大型 Diffusion 模型）

| 阶段 | Text encoder | Diffusion 10步 | VAE decoder | 端到端 |
|------|-------------|----------------|-------------|--------|
| Ours | 0.09s | 14.70s | 0.37s | **15.20s** |

## 结果对齐（Llama2-7B）

| 环境 | 框架/配置 | Generate 速度 |
|------|-----------|---------------|
| CANN 8.3 + torch_npu 2.8 | 原生推理 | 17 t/s |
| CANN 8.3 | vllm-ascend | 31 t/s |
| CANN 8.5 | ggml | **50 t/s** |

> 其他模型：Qwen-ASR 已经支持，Qwen-TTS 正在做整体测试。
