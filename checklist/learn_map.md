# vLLM / SGLang 二次开发 · C++ 学习清单
（含 llama.cpp 级进阶，纯 Markdown，可直接存为 `.md` 文件）

---

## 🧭 前置：先判断你要不要碰 C++

二次开发分三层，C++ 介入程度不同：

| 改动位置 | 语言 | 要 C++ 吗 |
|---|---|---|
| 调度策略、continuous batching、prefix cache、RadixAttention 树、API 层 | Python | ❌ 不用 |
| 在现有算子基础上加变体，调已写好的 kernel | Python + 少量 C++ 封装 | ⚠️ 只要看懂 pybind 入口 |
| 自研 / 改 CUDA kernel（新 attention、新量化、新 MoE） | C++/CUDA | ✅ 要，且 CUDA C 比 C++ 本体更重要 |

> 💡 大部分第一次贡献从第一层开始，根本碰不到 C++。招聘 JD 里"熟练掌握 C++ / CUDA C，能独立开发调优 Kernel"是**推理优化岗**的要求，不是二次开发标配。

---

## 📂 两个项目的 C++ 长什么样

### vLLM `csrc/`（~270 文件）
- `torch_bindings.cpp` —— PyTorch 扩展注册入口（Python↔C++ 唯一大门）
- `cache_kernels.cu` / `activation_kernels.cu` / `layernorm_kernels.cu` / `pos_encoding_kernels.cu` / `sampler.cu`
- `attention/` —— PagedAttention v1/v2
- `quantization/` —— AWQ/GPTQ/Marlin/FP8
- `moe/` —— MoE 对齐、路由
- `cutlass_extensions/` —— GEMM 调 CUTLASS
- C++ 本体（`.cpp`）很少，主要是 `.cu`；C++ 角色 = 薄封装 + pybind 注册，重活在 CUDA kernel

### SGLang `sgl-kernel/csrc/`
- `common_extension.cc` —— Python 绑定入口
- `attention/` —— cascade / lightning decode
- `gemm/` —— FP8 GEMM
- `moe/`、`quantization/`、`allreduce/`、`kvcacheio/`、`grammar/`
- 编译用 CMake + scikit-build-core，产出 `common_ops.so`，Python 侧 `from sgl_kernel import common_ops`

**共性**：C++ 只出现在 `csrc/`，框架主体（调度、Worker、KV 管理、RadixAttention）全是 Python。

---

## 🎯 阶段 1：C++ 最小集（≈ 1-2 周，每天 1h）

目标：**能读懂 `.cpp` 注册文件和 `.cu` 里嵌的 C++，不用能手写复杂工程**。

### 要学的

| 模块 | 具体点 | 为什么 |
|---|---|---|
| 基础语法 | `class`, `namespace`, 引用 `&`, `const`, `.h`/`.cc` 分离 | 读 `torch_bindings.cpp` 必备 |
| 模板 | `template <int TBLOCK>` 简单形参模板 | kernel 启动常把 block size 当模板参数 |
| `torch::Tensor` API | `.data_ptr<T>()`, `.sizes()`, `.stride()`, `torch::kCUDA` | csrc/ 里替代 STL 容器 |
| RAII + 智能指针 | `unique_ptr`/`shared_ptr` 能认出来 | vLLM `cumem_allocator.cpp` 用到 |
| 编译链接 | `#include <torch/extension.h>`、`.cu` 和 `.cpp` 分开编 | 看懂 setup.py / CMakeLists.txt |

### 不用学（省 80% 时间）

- ❌ STL 深坑（vector/map 在 csrc 用得极少）
- ❌ 继承多态（kernel 基本是扁平函数）
- ❌ C++17/20 花活（concepts、ranges、coroutine）
- ❌ 移动语义 / 内存池（那是 llama.cpp 级的题）

### 验证方式

打开 `vllm/csrc/torch_bindings.cpp` 前 50 行，能指着说"这行 include 了 attention header，这个类包在 `namespace vllm` 里"——过关。

---

## 🌉 阶段 2：PyTorch C++ Extension + 算子注册（≈ 1 周，关键桥）

### vLLM 侧：`csrc/torch_bindings.cpp`

4 个命名空间：`ops`（核心推理） / `cache_ops`（KV Cache） / `cuda_utils` / `custom_ar`

注册模式（`TORCH_LIBRARY_EXPAND` 逐步替代老的 `PYBIND11_MODULE`）：

```cpp
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("paged_attention_v1(Tensor! out, Tensor query, ...) -> ()");
    ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);
}
```

Python 侧 `torch.ops.vllm.*` → `_custom_ops.py` 包一层 → 上层调用，是"Python↔C++ 三跳"。

### SGLang 侧：`sgl-kernel/csrc/common_extension.cc` + CMake

两条硬规矩（SGLang 特有）：
1. `m.def` **必须带 schema 字符串**（为了 `torch.compile` 能捕获），不能直接 `m.def("rmsnorm", &rmsnorm)`
2. C++ 原生 `int`/`float` 要转 `int64_t`/`double`（Python 类型映射），SGLang 在 `sgl_kernel_torch_shim.h` 给 `make_pytorch_shim` 自动处理
3. Python 侧调的时候**不能用 kwargs，必须 positional args**

构建：CMake + scikit-build-core + ninja，产出 `common_ops.so`。

### 验证方式

`pip install -e vllm` 后，Python 里 `print(torch.ops.vllm.paged_attention_v1)` 能打到 C++，知道它在 `torch_bindings.cpp` 哪行注册的——过关。

---

## ⚡ 阶段 3：CUDA C（≈ 2-3 周，比 C++ 本体重要）

csrc/ 里 `.cu` 数量 >> `.cpp`，真要动 kernel 这块是主体。

### 必会

- `__global__` / `__device__` / `__host__` 修饰符
- 三维网格：`threadIdx.x`, `blockIdx.x`, `blockDim.x`, `gridDim.x`
- shared memory：`extern __shared__ float smem[]`（PagedAttention 核心）
- warp-level：`__shfl_sync`, `__syncwarp`（attention 规约常用）
- kernel launch：`my_kernel<<<grid, block, shmem, stream>>>(args...)`
- 内存布局：row-major、padding、alignment（不然写出来慢 3 倍）

### 推荐练手路径

1. 写个 `add_kernel.cu` 走通 `CUDAExtension` + `setup.py` 全流程
2. 改 vLLM `activation_kernels.cu`（SiLU/GELU，短，几百行）——加个自己的激活
3. 再碰 `layernorm_kernels.cu` → `pos_encoding_kernels.cu` → `cache_kernels.cu`
4. **最后才碰 `attention/paged_attention_v1.cu`**（几百行 + 复杂 indirect lookup，新人直接看会劝退）

### 调试工具

- `torch.autograd.profiler.profile(use_cuda=True)` 看 kernel 耗时
- `nsys profile python xxx.py`
- `TORCH_USE_CUDA_DSA=1` 开 CUDA device-side assert

---

## 🔧 阶段 4：第三方库集成（按需，可延后）

| 库 | 在 vLLM/SGLang 哪 |
|---|---|
| CUTLASS | `csrc/cutlass_extensions/` —— 分块 GEMM |
| FlashAttention | SGLang 通过 CMake FetchContent 拉，在 `attention/` |
| FlashInfer | SGLang 也拉了，cascade attention 用 |
| DeepGEMM | SGLang Hopper 专用路径 |
| Marlin | vLLM 4-bit GEMM，`csrc/quantization/marlin/` |

> 💡 这层模板嵌套深度恐怖，初期**直接用 `torch.ops.vllm.*` / `torch.ops.sgl_kernel.*` 调现成的**就行。

---

## 🏗️ 阶段 5（进阶）：llama.cpp 级 C++/C 工程师

> 📌 定位校准：vLLM/SGLang csrc = "Python 调 CUDA 的薄胶水"，C++ 是胶水；llama.cpp = **零依赖纯 C/C++ 推理框架**，ggml 算力层是 `.c`（C99），C++ 只包壳。要的是**系统级 C + C++ + 体系结构感知**，不是 C++ 语法多炫。

### 5.1 C 语言底子（必须先过，ggml 核心是 .c）

- `struct` 前向声明、opaque pointer 模式
- enum dispatch（`enum ggml_op { GGML_OP_MUL_MAT, ... }`）
- `.h` / `.c` 分离、`#include` 文本拼贴
- `void *` 泛型指针 + 手动 cast（`ggml_tensor.data` 是 `void*`，按 type 转）
- `static` 函数 / 内部链接
- 宏技巧：`offsetof()`、container_of 风格

验证：打开 `ggml/include/ggml.h` 能指着 `struct ggml_tensor` 说清 `ne[4]/nb[4]/src[2]/op` 每个字段干嘛——过关。

### 5.2 C++ 本体（比 vLLM csrc 深，但不追新标准）

llama.cpp 要在 **GCC 7.2（MIPS 君正）** 上能编，所以 `constexpr`/`std::filesystem` 这种都得有 fallback shim，风格偏 **"C with classes"**。

| 模块 | 具体点 | 在 llama.cpp 哪用 |
|---|---|---|
| 模板（中等） | `template<typename T>`、特化、SFINAE 基础 | 量化 dispatch、SIMD 抽象 |
| constexpr / consteval | 编译期计算 | 量化参数、对齐值（GCC7 要 `constexpr→const` 降级） |
| type traits | `enable_if`, `is_same` | 后端选型、模板分支 |
| 移动语义 / RAII | `std::move`、析构保释放 | `llama_model` / `llama_context` 生命周期 |
| std::atomic / memory_order | relaxed / acquire-release / seq-cst | 多线程采样、batch 调度、stop signal |
| optional / variant | C++17 结构化绑定 | sampling 管线、参数解析 |
| lambda + std::function | 回调 | server 请求处理 |

❌ 不用：继承多态深坑、虚函数、RTTI、concepts、ranges、coroutine（ggml/llama 几乎不用）。

### 5.3 内存管理（这档重头戏，vLLM csrc 完全不碰）

**① Arena / Bump Allocator —— `ggml_ctx`**

一次性 `malloc`/`mmap` 一大块，之后只做指针递增，推理期零 `malloc/free`。

**② 图级离线内存规划 —— `ggml_gallocr`**

不是"现问现给"，是"先扫整张计算图，排好偏移表，运行期照表填"——70 层 transformer 中间激活不复用显存爆炸，gallocr 压到"任一时刻同时存活峰值"。

流程：正向遍历分配 → 反向扫图标记生命周期终点 → 正向回收 `free_blocks`（按 offset 有序链表，可合并降碎片）→ 对齐 `GGML_MEM_ALIGN`。

**③ 虚拟内存管理（VMM）**：SYCL/CUDA 后端 pool with VMM，显存不够自动 swap。

要学：`mmap`/`munmap`/`madvise`/`mlock`；arena vs pool；对齐与 padding；缓存行友好布局。

### 5.4 SIMD Intrinsics（llama.cpp 性能灵魂）

ggml 在 `ggml-cpu.c` 用**宏统一不同架构**，上层算子只调 `GGML_F32x4_*` 这组宏，加新架构只补宏定义：

| 架构 | 指令集 | llama.cpp 用法 |
|---|---|---|
| ARMv8 | NEON 128-bit | 全平台兜底 |
| ARMv9 | SVE | 可扩展向量 |
| ARM | I8MM (smmla) | int8 矩阵乘 |
| x86 | AVX2 256-bit | 主流桌面 |
| x86 | AVX-512 + BF16/INT8 | 服务器 |
| x86 | AMX | 至强进阶 |
| Apple | AMX（私有） | M1/M2/M3 比纯 NEON 快 3-4x |

还要懂：**量化 + SIMD 耦合**——Q4_0/Q5_K 这些 block-wise quant 的 unpack + dot 在 SIMD 里手撸，是 llama.cpp 性能 real secret。

练手：读 `ggml-cpu.c` 里 `ggml_compute_forward_mul_mat` 的 "arch + quant type" 两维 dispatch。

### 5.5 并发与多线程

- OpenMP：`#pragma omp parallel for`（batch 内 token 并行）
- pthread：底层线程池（server、batch scheduling）
- `std::atomic` + 6 种 memory_order（stop signal、多 batch 共享 state）
- 无锁队列（server request queue）
- NUMA 感知（多路 ARM Neoverse N2 要绑核）

要学：`std::thread`、`mutex`/`condition_variable`、false sharing、CPU affinity。

### 5.6 跨平台工程化（分水岭）

vLLM csrc 只要 Linux + CUDA + 较新 gcc/clang。llama.cpp 要过：

- 三编译器：gcc（含 gcc7 老 libstdc++）、clang、MSVC（Win arm64 内联汇编有限制）
- 宏守卫地狱：`#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)` 每加一 arch 补一整套
- 属性双写：`__attribute__((aligned(N)))` vs `__declspec(align(N))`
- WASM / 嵌入式 / MIPS / RISC-V 都能编 → 不能依赖"现代 OS 才有"的东西
- CMake 多预设：`arm64-windows-llvm-release`、`-G Xcode` 等多配置生成器

> ⚠️ 这档 C++ 工程师和 vLLM csrc 级最大气质差：**写代码时脑子里要有"这行在 MIPS gcc7 + 无 NEON + 512MB RAM 上能不能编"的雷达**。

### 5.7 文件格式 & 零拷贝加载

- **GGUF 格式**：header + metadata + tensor list + tensor data，纯 C parser
- **mmap 零拷贝**：`.gguf` 直接 map 到 VA，tensor->data 指进去，不拷 RAM
- **权重 / 计算 buffer 分离**：`WEIGHTS`（常驻）vs `COMPUTE`（可复用），即 `n_gpu_layers` 能把部分层留 CPU 的底层

要学：`mmap`/`madvise(MADV_RANDOM)`/`mprotect`、文件格式设计（对齐、可扩展 header、向后兼容）。

---

## 🆚 两档对照表

| 维度 | vLLM/SGLang csrc 级 | llama.cpp 级 |
|---|---|---|
| C++ 角色 | 薄胶水（pybind + torch::Tensor） | 主力语言（零依赖推理框架） |
| C 语言 | 不用 | **必须**（ggml 是 .c） |
| 内存 | PyTorch 兜底 | 自撸 arena + 图级离线规划 |
| SIMD | 不碰（CUDA 管） | **核心**（NEON/AVX2/AVX-512/SVE 手撸） |
| 量化 | 调 Marlin/GPTQ 现成 | 自撸 Q4_K/Q5_K block-wise + SIMD unpack |
| 跨平台 | Linux + CUDA | ARM/x86/Apple/MIPS/RISC-V/WASM/MSVC |
| 编译器容忍 | gcc 9+ / clang 10+ | gcc 7 libstdc++ 都要能编 |
| 并发 | Python asyncio 主 | OpenMP + pthread + atomic + NUMA |
| 文件格式 | safetensors（借第三方） | GGUF 自设计 + mmap 零拷贝 |
| 典型改法 | 改 `torch_bindings.cpp` 注册、改 `.cu` | 改 `ggml-cpu.c` SIMD 宏、改 `ggml_gallocr`、加新 quant |

---

## 🛤️ 练手路径总览

### vLLM/SGLang 向（二次开发主流）
1. 读 Nano-vLLM / miniSGLang（纯 Python，几百~几千行，调度+PagedAttention/RadixAttention 全在）
2. 读主框架 Python 侧：`LLMEngine → Scheduler → Worker → model_runner`（vLLM） / `RadixCache + scheduler + tokenizer_manager`（SGLang）
3. 打开 `torch_bindings.cpp` / `common_extension.cc`，对照 Python 侧 `torch.ops.vllm.*` 看"注册—实现—调用"三跳
4. 真要写 kernel：从 `activation_kernels.cu`（vLLM）或 `elementwise/rmsnorm.cu`（SGLang）改起，别一上来怼 PagedAttention

### 第一个练手任务（二选一）
- **路径 A（vLLM）**：在 `activation_kernels.cu` 加 `custom_silu_with_scale`，`torch_bindings.cpp` 注册，`_custom_ops.py` 包一层，3 天跑通
- **路径 B（SGLang）**：抄 `elementwise/rmsnorm.cu` 写 `elementwise/add_constant.cu`，`common_extension.cc` 里 `m.def`（带 schema）+ `m.impl`，`sgl_kernel/` 包一层，`make build` 重编 sgl-kernel

### llama.cpp 向（推理优化 / 嵌入式部署）
1. 读 `ggml/include/ggml.h` 的 `struct ggml_tensor`（`ne/nb/src/op/view_src`）
2. 读 `ggml.c` 前 500 行 —— arena allocator（`ggml_init` / `ggml_alloc`）
3. 读 `ggml-alloc.c` 的 `ggml_gallocr_alloc_graph` —— 两次遍历 + 生命周期 + free_blocks 合并
4. 读 `ggml-cpu.c` 的 `GGML_F32x4_*` 宏段 —— SIMD 抽象层范本
5. 读 `llama.cpp` 的 `llama_decode → build_graph → ggml_graph_compute` 链路
6. 改一个量化格式：在 Q4_K 旁抄 Q3_K 变体，改 quantize/unpack/mul_mat 三处 —— "毕业礼"

---

> 复制代码块里的内容，粘贴到 `xxx.md`，用 Obsidian / Typora / VS Code 打开即可正常渲染。