# OminiX-Ascend

OminiX-Ascend 是 [OminiX](https://github.com/OminiX) 框架在华为 Ascend NPU 上的适配与扩展，基于 llama.cpp 项目构建，为昇腾硬件提供高性能的多模态 AI 推理能力。

## 项目定位

OminiX 是一个以 **ggml** 张量计算库为核心的多模态 AI 推理框架。OminiX-Ascend 作为其在 Ascend 平台上的实现，具备以下特性：

- **以 ggml 为基础**：ggml 是 OminiX 的底层张量计算引擎，负责计算图的构建与执行。OminiX-Ascend 通过智能体（Agent）流水线将多种 AI 模型统一转换、优化、编译并部署为原生 C++ 推理程序，让计算任务更高效的在 Ascend NPU 上运行。
- **以 llama.cpp 为推理入口**：llama.cpp 构建于 ggml 之上，提供完整的模型加载、推理调度和应用层接口。OminiX-Ascend 通过 llama.cpp 的工具链（`llama-cli`等）作为 C++ 推理的执行入口，用户可以直接使用这些工具完成模型推理任务。

## 架构概览

![OminiX 框架图](media/OminiX_framework.png)

OminiX 通过 **Analyzer → Planner → GGUF Converter → C++ Model Design → Validator** 流水线完成模型转换与部署，并通过 Reflector/Debug Loop 对编译错误和数值不匹配问题进行迭代自纠正。

## Model Zoo

OminiX 框架支持多种模态的模型推理：

| 模态 | 支持的模型 |
|------|-----------|
| **LLM（大语言模型）** | 由 llama.cpp 原生支持，涵盖 LLaMA、Qwen、Mistral、ChatGLM等 |
| **图像生成** | Qwen Image 2512、SD1.5、SD2.1、SDXL、SD3、Flux2.2 |
| **视频生成 & 世界模型** | Wan2.2、Cosmos |
| **VLM & VLA（视觉语言/动作模型）** | Moxin-VLA、OpenVLA、Qwen-VL、Vote、Pi-0.5、AntVLA |
| **语音识别** | Qwen ASR、Qwen TTS、Whisper、GPT-SoVITS、Fun-ASR |

## 性能对比

详细的性能测试数据请查看 **[Benchmarks](docs/benchmarks.md)**，以下是部分亮点：

| 模型 | Our framework | 对比方案 | 加速比 |
|------|---------------|----------|--------|
| Deepseek OCR 2 | **33 t/s** | Torch-npu 5 t/s | 6.6x |
| Qwen3-8B | **42 t/s** | Torch-npu 28 t/s | 1.5x |
| Qwen3.5-9B | **28 t/s** | Torch-npu 8.9 t/s | 3.1x |
| Llama2-7B | **50 t/s** | vllm-ascend 31 t/s | 1.6x |

## CANN 后端

CANN（Compute Architecture for Neural Networks）后端是 OminiX-Ascend 的核心组件，位于 `ggml/src/ggml-cann/`：

- **支持的数据类型**：FP16、Q8_0、Q4_0
- **支持的硬件**：Ascend 910B、Ascend 310P
- **关键特性**：
  - 基于 ACLNN 的高性能算子实现
  - 异步流执行与设备内存池管理
  - 权重 NZ 格式转换优化
  - CPU + NPU 混合推理

## 快速开始

### 环境要求

- 华为 Ascend NPU 硬件（Atlas 300T A2 / Atlas 300I Duo）
- CANN Toolkit（8.x 版本）
- CMake >= 3.14
- C++ 编译器（支持 C++17）

### 编译构建

```bash
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# CMake 配置与构建
cmake -B build \
    -DGGML_CANN=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DSOC_TYPE=ascend910b
cmake --build build --config Release -j$(nproc)
```

### 运行推理

```bash
# 使用 llama-cli 进行交互式推理
./build/bin/llama-cli -m model.gguf -ngl 99
```

## 目录结构

```
OminiX-Ascend/
├── ggml/                  # ggml 张量计算库
│   ├── src/ggml-cann/     #   └── CANN 后端实现（核心）
│   └── include/           #   └── 公共头文件
├── src/                   # llama.cpp 主源代码
│   └── models/            #   └── 50+ 种模型架构实现
├── include/               # llama.cpp 公共 API（llama.h）
├── tools/                 # 应用工具
│   ├── cli/               #   └── llama-cli 命令行推理
│   ├── server/            #   └── llama-server REST API
│   └── quantize/          #   └── 模型量化工具
├── gguf-py/               # Python GGUF 文件操作库
├── examples/              # 示例应用
├── docs/                  # 文档
│   └── backend/CANN.md    #   └── CANN 后端详细文档
└── .devops/               # Docker 配置
    └── cann.Dockerfile    #   └── CANN 构建镜像
```

## 相关文档

- [CANN 后端详细指南](docs/backend/CANN.md)
- [构建文档](docs/build.md)
- [多模态支持](docs/multimodal.md)
- [模型量化指南](docs/quantize.md)

## 许可证

本项目基于 MIT 许可证开源。
