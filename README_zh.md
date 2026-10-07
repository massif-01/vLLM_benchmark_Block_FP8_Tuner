# vLLM Block FP8 Tuner

**面向 CUDA W8A8 Block FP8 普通线性层的 Triton Kernel 调优工具**

[English](README.md) · [简体中文](README_zh.md)

为 vLLM 的 W8A8 Block FP8 Triton 矩阵乘法内核搜索合适的启动参数，对候选结果进行数值校验，并生成 vLLM 可读取的 JSON 配置。支持利用多张 NVIDIA GPU 并行调优不同的 `M`，由主进程统一合并结果，避免多进程互相覆盖文件。

这是一个**独立维护的调优工具**，并非 vLLM 官方组件。它测量的是独立 GEMM 内核，不是完整模型推理；生成配置也不意味着实际服务一定提速。

> **当前验证状态：** 已审查版本的 CPU、CLI 和 Shell 回归测试通过；真实 CUDA 编译、数值调优、已安装 vLLM Loader 集成和模型服务性能**尚未完成验收**。在目标设备通过 [CUDA 验收流程](docs/CUDA_VALIDATION.md)前，不应把生成结果视为已经具备生产环境验证证据。

## 这个工具做什么

```text
模型配置 ──→ Qwen3 普通线性层的各 TP rank 形状 ──┐
                                                 ├─→ 调优 (M, N, K)
已实际观测的单 rank (N, K) ──────────────────────┘           │
                                                           ▼
                                              测量 Triton 启动参数候选
                                                           │
                                                           ▼
                                               独立复测与数值正确性校验
                                                           │
                                                           ▼
                                               主进程合并所有 GPU 结果
                                                           │
                                                           ▼
                                               写出 vLLM 格式的 JSON
```

- **理解模型结构：** 针对 `Qwen3ForCausalLM` 和 `Qwen3MoeForCausalLM` 推导真实的 TP 分片形状，处理 GQA、KV 复制以及实际存在的普通 dense/shared MLP。
- **支持手工指定形状：** 对不适合自动推导的场景，使用已观测的单 rank `(N, K)` 直接调优。
- **多 GPU 并行：** 按 `M` 划分工作，每个 worker 返回结果；主进程检查完整性并统一保存。
- **保护已有配置：** 不同的 `M` 可以安全合并，重复 `M` 默认拒绝覆盖；文件更新受锁保护，并使用原子替换。
- **留下验证证据：** 对入围配置使用同一批 FP8 数据和反量化 FP32 参考结果做数值检查；测量、误差和软件/内核身份写入独立报告。

### 维护范围

| 正式支持 | 不属于当前维护范围 |
| --- | --- |
| Compute Capability **8.9 及以上**的 NVIDIA CUDA GPU | 更早的 GPU；ROCm、XPU、CPU 调优 |
| 已安装 vLLM 的 W8A8 Block FP8 **Triton 普通线性层**内核 | Routed/Fused MoE 专家内核调优 |
| Qwen3 Dense 与 Qwen3-MoE 的明确普通线性层 Adapter | 通用架构猜测；Qwen3-Next/VL/Omni、DeepSeek 自动推导 |
| FP8 E4M3FN 输入、FP32 Scale；FP16/BF16/FP32 输出 | INT8 与 AWQ 调优（旧实现已移除） |

**特别说明：** 支持 Qwen3-MoE 的部分普通线性层，**不代表**支持 Routed Expert 调优。能够算出某个 Linear 的形状，也不能证明实际运行的模型采用 Block FP8、或最终选择了 Triton 后端。后两项必须单独核实。

## 环境要求

- 同一 Python 环境中安装**支持 CUDA 的 PyTorch、Triton，以及兼容的 vLLM**。依赖版本以实际安装的 vLLM 要求为准；本仓库不提供替代内核，也不承诺任意版本组合均可使用。
- 真实调优需要 **SM 8.9+** 的 NVIDIA GPU。例如 RTX 4090（SM 8.9）和 H100（SM 9.0）符合硬件门槛；RTX A6000（SM 8.6）和 Jetson AGX Orin（SM 8.7）不符合。
- 当前安装的 vLLM 必须具有相容的 FP8 私有 Triton 内核、配置 Loader 和设备命名接口。升级 vLLM 后，应重新执行环境检查。

`--help` 和**显式形状**的 `--preview` 不依赖 CUDA、PyTorch 或 vLLM。使用 `--model` 预览时，需要能加载模型配置并调用 vLLM 的 dtype resolver；它不会下载模型权重，也不会执行 GPU Kernel 调优。

## 快速开始

以下命令均从仓库根目录运行。

**第一步：检查目标 CUDA 环境**

```bash
bash scripts/environment_check.sh
```

该命令检查必要模块、内核参数签名、设备名称接口和 GPU 计算能力。如果返回非零退出码，说明当前环境尚不能进行正式调优。

**第二步：预览一个已知 GEMM 形状**

```bash
python3 benchmark_w8a8_block_fp8.py \
  --shape 128 256 \
  --out-dtype bfloat16 \
  --batch-size 17 \
  --preview
```

这里 `(N, K) = (128, 256)` 表示权重的两个维度，`M = 17` 表示一次 GEMM 处理的扁平化输入行数。`--preview` 只输出任务计划，**不会**编译或测量 GPU 内核。

**第三步：真正调优这个形状**

```bash
python3 benchmark_w8a8_block_fp8.py \
  --shape 128 256 \
  --out-dtype bfloat16 \
  --batch-size 17 \
  --save-path ./tuned_configs/quickstart
```

即使只指定一个形状、一个 `M`，工具仍会搜索数量较多的启动参数组合。因此先从这个受限任务入手，再考虑完整的 M 网格。仅当调优和校验成功后，才会生成 vLLM JSON 与独立报告。**生成文件不等于已经安装到 vLLM。**

### 根据 Qwen3 模型自动确定形状

例如预览官方 Qwen3-Coder FP8 在目标 TP=4 下的普通线性层形状：

```bash
python3 benchmark_w8a8_block_fp8.py \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --tp-size 4 \
  --preview
```

工具仅加载模型配置，不加载权重。模型模式默认使用 `--out-dtype auto`：由**当前安装的 vLLM dtype resolver** 决定输出类型。在支持 BF16 的目标 CUDA 环境中，这个 BF16 模型预期解析为 `bfloat16`；用户显式指定的 `--out-dtype` 始终优先。

确认计划及目标 Backend 后，可以通过便捷脚本只调一个 `M`：

```bash
BATCH_SIZE=17 bash scripts/tune_qwen3_coder.sh
```

该脚本默认选择 Qwen3-Coder FP8、目标 TP=4、128×128 FP8 Block。`BATCH_SIZE` 只限制这次搜索的 `M`；不设置时将使用默认的完整 M 网格。针对其他受支持的配置，还可使用 `scripts/tune_qwen3.sh` 和 `scripts/tune_custom.sh`。

**自动推导形状不是量化转换。** 例如 `--model Qwen/Qwen3-8B` 可以分析未量化模型的普通线性层维度，但不会把它变成 FP8 Checkpoint，也不能证明部署时正在使用本 Triton Kernel。

## 选择正确的调优模式

| | `--model MODEL` | `--shape N K` |
| --- | --- | --- |
| `(N, K)` 的来源 | 受支持的 Qwen3 模型配置 + 目标 `--tp-size` | 已实际观测的单个目标 TP rank 形状 |
| 输出 dtype | 默认 `auto`，交给安装版本的 vLLM 解析 | **必须明确指定** `float16`、`bfloat16` 或 `float32` |
| FP8 布局责任 | 校验可识别的 checkpoint metadata 与自动 Adapter 的 Block/对齐限制 | 用户必须确认真实 Runtime 使用相同量化布局和 Triton 内核 |
| 适用场景 | 受支持 Qwen3 的普通线性层 | 未支持的架构或已有真实维度的特定 GEMM |

这里有两个容易混淆的参数：

- **`M` 是 GEMM 输入的扁平化 Token 行数**，不一定等于推理服务的并发请求数。
- **`--tp-size` 是模型目标 Tensor Parallel 的分片数量**，不是用于调优的 GPU 数量。调优 GPU 数由可用设备和待调 M 数量决定。

模型模式的 `auto` dtype **与运行平台有关**：如果在纯 CPU 环境预览，vLLM 可能解析出不同于目标 NVIDIA CUDA 环境的 dtype。因此必须在最终调优设备上确认；显式形状模式无法推断模型 dtype，未填写 `--out-dtype` 将直接报错，而不会擅自选用 FP16。

自动 Adapter 有意保持保守：它会检查 Q/KV 分片或复制、融合投影宽度、dense/shared MLP 是否真的存在，以及 FP8 Block 对齐条件。无法识别的架构、影响目标投影的量化排除项或不匹配的布局会直接拒绝，而不是猜出一个看似合理的 Shape。自动 Adapter 当前仅支持**方形 FP8 Block**；使用显式形状也不意味着可以忽略真实 Runtime 的 Scale/Layout 约束。

## 输出文件与配置复用

调优成功后，输出目录示意如下：

```text
tuned_configs/quickstart/
├── N=128,K=256,device_name=GPU_NAME,dtype=fp8_w8a8,block_shape=[128,128].json
├── N=128,K=256,device_name=GPU_NAME,dtype=fp8_w8a8,block_shape=[128,128].json.lock
└── reports/
    └── <UTC时间>-<运行ID>.json
```

根目录下的 `.json` 是 **vLLM Loader 使用的配置**：以 `M` 为键，映射到 Triton 启动参数。`reports/` 中的文件记录模型/形状计划、时间、数值误差、随机种子、GPU 和软件/Kernel 身份。`.lock` 只用于文件写入协调，**不需要安装到 vLLM**。

- 再次调优时，如果新旧 `M` **互不重叠**，工具会保留此前已经保存的值并追加新条目。
- 如果某个 `M` 已存在，默认报错；显式 `--overwrite` **只替换本次指定的重叠 `M`**，不会丢弃其他条目。
- 参与调优的 GPU 必须是相同型号。不同 `M` 采用确定性的负载均衡策略分配，主进程在写入前检查是否存在遗漏、重复和意外结果。
- 单个 JSON 的更新使用文件锁与原子替换；**多个形状之间不是事务**。后续文件保存失败时，已成功写入的文件可能保留，但整次任务不会报告成功。

**务必按运行环境隔离配置。** vLLM 官方文件名包含 `(N, K)`、GPU 名称、FP8 类型与 Block 大小，却**不包含**模型、TP、输出 dtype 或 vLLM/Kernel 版本。不同输出类型、内核版本或其他不兼容的调优环境应使用不同的 `--save-path`。当前工具不会根据历史 Provenance 自动阻止两个同名文件的跨运行合并。

### 安全安装与验证

生成的 JSON 只有放入安装版本的 vLLM 配置目录后，才有机会被对应 Loader 消费。**优先在独立虚拟环境或容器中验收**：覆盖配置可能改变该环境内服务进程使用的内核参数，不能未经备份就覆盖已有文件。

下面的命令只安装**一份此前不存在的同名配置**；如果检测到同名文件，会拒绝覆盖：

```bash
CONFIG_DIR=$(python3 -c 'from pathlib import Path; from vllm.model_executor.layers.quantization.utils import fp8_utils; print(Path(fp8_utils.__file__).resolve().parent / "configs")')
SOURCE=$(find ./tuned_configs/quickstart -maxdepth 1 -type f -name 'N=*.json' -print -quit)

if [ -z "$SOURCE" ]; then
  echo "没有找到生成的配置文件" >&2
else
  mkdir -p "$CONFIG_DIR"
  TARGET="$CONFIG_DIR/$(basename "$SOURCE")"
  if [ -e "$TARGET" ]; then
    echo "拒绝覆盖已有配置：$TARGET" >&2
    echo "请先备份，再由你明确决定是否替换。" >&2
  else
    cp "$SOURCE" "$TARGET"
    echo "已安装：$TARGET"
  fi
fi
```

如果目标位置**原本就有文件**，应先备份，再明确决定是否替换。以下命令仅适用于这一情况，且需先执行上方代码以获得 `SOURCE` 和 `TARGET`：

```bash
BACKUP_DIR=$(mktemp -d "$PWD/tuned_configs/install-backup.XXXXXX")
cp -p "$TARGET" "$BACKUP_DIR/"
echo "原配置已备份到：$BACKUP_DIR"
cp -i "$SOURCE" "$TARGET"   # 仅在确认提示后替换。

# 验收结束后，恢复原文件：
cp -p "$BACKUP_DIR/$(basename "$TARGET")" "$TARGET"
```

如果该文件**在安装前不存在**，回滚时只删除本次新增的文件（`rm -- "$TARGET"`）。对于多个 Shape，应逐个检查和安装，不要把整个输出目录或 `reports/` 批量复制进去。安装和回滚后都应重新启动受影响的服务进程；修改 vLLM 安装目录可能需要相应的文件写入权限。

完成安装后，使用**新的 Python 进程**，并保持与生成配置时相同的 Shape、`M`、Block 布局和输出 dtype：

```bash
python3 benchmark_w8a8_block_fp8.py \
  --shape 128 256 \
  --out-dtype bfloat16 \
  --batch-size 17 \
  --save-path ./tuned_configs/quickstart \
  --verify-installed
```

这个检查会比较已安装 Loader 返回的配置，并调用公开 Triton wrapper 进行数值检查。但它**不能证明完整推理服务选择了 `TritonFp8BlockScaledMMKernel`**：服务端也可能使用 CUTLASS、DeepGEMM 或 FlashInfer。还需要单独确认真实模型的 Backend 选择和性能。Loader 有配置缓存，因此有意安装或回滚后，应重新启动相关服务进程。

完整的硬件验收步骤（包括 FP16/BF16 Smoke、模型配置预览，以及单次与连续多次调用的对照）参见 [CUDA 验收文档](docs/CUDA_VALIDATION.md)。

## 调优与正确性判断方式

本工具直接调用**已安装 vLLM 的私有 Triton 内核**，不会复制一份旧 Kernel 作为备用实现。每组 `(M, N, K)` 的流程是：

1. 生成合成的 FP8 E4M3FN 激活和权重，以及 FP32 Scale；同一 Shape 的候选配置使用同一批输入。
2. 对候选配置预热并使用 CUDA Event 测时。默认**每个 Event 调用一次 GEMM**；显式设置 `--calls-per-event >1` 才测量连续多次调用的平均值。这种负载可能具有不同的缓存行为和排名。
3. 根据测量中位数筛选前三名，并在默认配置能够运行时加入对照；对这些候选进行独立复测。
4. 使用**相同 FP8 数据与 Scale** 反量化后的 FP32 矩阵乘法作为参考进行数值校验（`rtol=0.02`、`atol=0.02`），拒绝 NaN/Inf。
5. 保存复测中最快、且数值通过的候选和测量报告。可识别的 Triton 资源不足候选会跳过；其他编译、运行和数值错误不会被静默吞掉。

这证明的是有限条件下的**内核数值合理性与调优结果**，不是端到端模型精度、生产输入分布代表性、全局最优配置或真实服务吞吐提升。只有部署时实际选中了对应 Triton 路径，调优结果才有可能发挥作用。

## 常用参数

| 参数 | 含义 |
| --- | --- |
| `--model MODEL` / `--shape N K` | 互斥的形状来源；`--shape` 可以重复 |
| `--tp-size TP` | 模型目标 TP，默认 `1` |
| `--out-dtype auto\|float16\|bfloat16\|float32\|half` | 模型模式默认 `auto`；显式 Shape 模式必填；`half` 等价于 `float16` |
| `--block-n N`、`--block-k K` | FP8 量化 Block 大小，默认均为 `128` |
| `--batch-size M` | 只调一个 `M`；省略则搜索默认 18 个 M 点 |
| `--save-path DIR` | 输出 JSON 和报告，默认 `./tuned_configs` |
| `--overwrite` | 允许替换本次重叠的 `M`，不影响其他 M |
| `--preview` | 仅展示计划，不执行 GPU 内核 |
| `--check-environment` | 检查 CUDA/vLLM/Triton 接口与设备能力 |
| `--verify-installed` | 比较 Loader 配置并调用公开 wrapper 检查 |
| `--measurements N`、`--calls-per-event N` | CUDA Event 轮数（默认 `5`）和每轮调用数（默认 `1`） |
| `--trust-remote-code` | 显式允许模型仓库代码执行；**默认关闭** |

Shell Wrapper 另支持 `BATCH_SIZE`、`OUT_DTYPE`、`SAVE_PATH`、`PYTHON`、`TRUST_REMOTE_CODE=1` 等环境变量。批量任务中的 `SAVE_PATH` 或 `--save-path` 指定的是**批量根目录**；不同模型和 TP 会自动隔离到子目录。参见 [`examples/tune_qwen3_models.sh`](examples/tune_qwen3_models.sh)。

## 测试与常见问题

```bash
python3 -m pytest -q
# 具备兼容 CUDA GPU 的机器还应执行：
python3 -m pytest -q tests/test_gpu.py
```

已审查 PR 的 Python 3.10/3.13 CPU/Shell 检查通过，但缺少 CUDA/vLLM 的机器上 GPU 模块会跳过。**GPU 测试被跳过不等于硬件验收通过。** 详细证据参见 [验证记录](docs/VALIDATION.md)。

| 遇到的问题 | 建议检查 |
| --- | --- |
| `--shape` 提示缺少 `--out-dtype` | 明确指定目标服务实际采用的输出 dtype；仅凭矩阵维度无法推断。 |
| 模型架构或排除项被拒绝 | Adapter 无法证明目标普通 FP8 Linear 确实存在。确认真实 Backend/Layout 后，再使用已观测的单 rank `--shape`。 |
| 已存在同一个 `M` | 换输出目录、调不同的 `M`，或明确使用 `--overwrite`。 |
| 内核/helper/signature 检查失败 | 安装的 vLLM 可能与本工具不兼容；应确认目标版本，而非静默换用另一个 Kernel。 |
| JSON 安装后服务没有变化 | 核对完整文件名、M 选择、安装目录、进程是否重启，以及服务是否真的选择 Triton 后端。 |

## 项目状态与许可

目前正式维护的是 **CUDA W8A8 Block FP8** 主路径。本仓库不再提供旧 INT8/AWQ 调优实现及其四个 Python 入口；外部对这些已移除入口或模块的调用将中断。[历史源码与 AWQ 文档](https://github.com/massif-01/vLLM_benchmark_Block_FP8_Tuner/tree/392a13ad7e8b1e34b3e9e44576c957e841174697)可通过删除前的固定 commit 追溯。

真正的 FP8 兼容入口 [`benchmark_w8a8_block_fp8_qwencoder.py`](benchmark_w8a8_block_fp8_qwencoder.py)和 DeepSeek 迁移提示脚本 [`scripts/tune_deepseek_v3.sh`](scripts/tune_deepseek_v3.sh)继续保留。后者提供迁移指引，不提供 DeepSeek 自动形状推导。

仓库曾针对一个固定的 vLLM 上游版本核对源码契约；这既不代表与所有版本兼容，也不能替代 CUDA 实测。固定版本和待验收事项参见 [验证记录](docs/VALIDATION.md)。

本项目遵循 [Apache License 2.0](LICENSE)。代码来源与版权说明见 [NOTICE](NOTICE) 及各源码文件头。
