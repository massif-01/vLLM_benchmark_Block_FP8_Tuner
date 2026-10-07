# CUDA vLLM W8A8 Block FP8 调优工具

[English](README.md) | [中文](README_zh.md)

本项目是 **CUDA-only vLLM W8A8 Block FP8 Triton kernel tuning tool**。正式维护入口为 `benchmark_w8a8_block_fp8.py`：调优普通二维 linear GEMM，校验 finalists 的数值结果，完整合并请求的 M，并安全保存 vLLM 格式配置。不承诺必然提速，也不代表优化了整个模型。

## 范围与证据

| 能力 | 当前范围与验证等级 |
| --- | --- |
| 设备 | NVIDIA CUDA，原生 FP8 compute capability >= 8.9；需要 CUDA PyTorch、vLLM 和 Triton |
| 内核 | 已安装 vLLM 的 `_w8a8_triton_block_scaled_mm`；A/B 为 FP8 E4M3FN，scale 为 FP32；输出 FP16、BF16 或 FP32 |
| 自动 dense adapter | `Qwen3ForCausalLM`：GQA QKV、attention output、融合 gate/up、down projection |
| 部分 MoE adapter | `Qwen3MoeForCausalLM`：attention，以及根据真实 layer schedule 确认存在的普通 dense/shared MLP |
| 排除 | routed experts / fused MoE、LM head、router 和 shared-expert gate、Next/VL/Omni/嵌套配置、DeepSeek 自动推导、未知架构 |
| 证据 | CPU shape/保存/入口及源码契约测试通过；本机 GPU 编译、数值/性能和 serving 验证仍为**未验证** |

自动 shape detection 只针对显式支持的 architecture。未知架构明确失败，不回退到无关尺寸；不以模型名包含 “Qwen” 判断 adapter。支持 MoE 模型的部分普通 linear **不等于调优 routed expert kernel**。

Adapter 确认的是普通 linear 的维度，不证明每层都量化，也不证明 serving 引擎选用了本 Triton backend。有 checkpoint 量化元数据时，自动推导仅接受 dynamic `fp8`，且 `weight_block_size` 必须匹配。同时检查 `ignored_layers` 和 `modules_to_not_convert`。明确非目标 exclusion（LM head、embedding、layernorm、Q/K norm、MoE router `mlp.gate`、shared-expert gate、routed-expert 子树和 bias 参数）允许；checkpoint/fused runtime 名称中的目标 projection、宽泛父级范围及无法识别的 pattern 仍 fail closed，错误会指出具体 exclusion 和 `--shape N K` 路径。小分类器只识别字面名称/分段 `*`，不实现任意正则 pattern。无量化元数据时，预览/报告会明确提示仍需确认实际 FP8 和 backend。其他量化格式只有在真实到达相同 kernel/layout 时，才可用显式观测 shape。

仓库保留历史/实验性 INT8 和 AWQ 脚本，但它们**不属于当前维护和验证的工作流**，不推荐作为入口；不维护其兼容性、正确性和 runtime 配置消费能力。本项目不提供 ROCm/XPU 支持。

## 快速开始

使用与已安装 vLLM 匹配的 CUDA 环境。CPU 规划/测试使用 Python 3.10+；GPU Python、PyTorch、Triton 要求遵循该 vLLM build。不内嵌旧 kernel 来绕过导入失败。

```bash
# 无 CUDA/PyTorch/vLLM 也可查看帮助
python3 benchmark_w8a8_block_fp8.py --help

# 已观测的每个 TP rank (N,K)：仅 CPU 预览
python3 benchmark_w8a8_block_fp8.py --shape 768 2048 --batch-size 17 --preview

# 架构预览需要 vLLM 配置加载能力，但不初始化 CUDA
python3 benchmark_w8a8_block_fp8.py --model Qwen/Qwen3-8B --tp-size 4 --preview

# CUDA/import/symbol/signature/device 检查，失败返回非零
bash scripts/environment_check.sh

# 小规模真实调优：包含默认配置对照、finalist 复测和 correctness
python3 benchmark_w8a8_block_fp8.py --shape 128 256 --batch-size 17 --save-path ./tuned_configs

# Qwen3 的普通 linear；先确认真实 FP8 checkpoint/backend
bash scripts/tune_qwen3.sh Qwen/Qwen3-8B 4 128 128
```

微基准直接生成合成的已量化 FP8 A/B 和 scales，不包含激活量化、模型加载、attention 或 serving 开销。使用未量化模型标识的示例只推导其 linear 维度，不会把 checkpoint 转换为 FP8。

## 参数与语义

| 参数 | 含义 / 默认值 |
| --- | --- |
| `--model` | 显式 adapter 支持的模型标识/本地配置；与 `--shape` 互斥 |
| `--shape N K` | 可重复指定已观测的**每个 rank**普通 linear 权重尺寸；已做 TP 切分，不再除以 TP |
| `--tp-size`, `-tp` | 目标 vLLM tensor-parallel world size，默认 `1`；不是调优 GPU 数量 |
| `--batch-size` | 单个 GEMM M / flattened token rows；省略则为 `1,2,4,8,16,24,32,48,64,96,128,256,512,1024,1536,2048,3072,4096` |
| `--block-n`, `--block-k` | checkpoint/runtime 的量化布局，默认 `128,128`；不小于 32 的 2 的幂，不是任意调优 tile |
| `--out-dtype` | 默认 `float16`，`half` 为别名，另有 `bfloat16`、`float32`；须匹配目标 runtime |
| `--input-type` | 只允许 `fp8` |
| `--save-path` | 默认当前目录下 `./tuned_configs`；单模型 Shell wrapper 默认仓库根目录 `tuned_configs/` |
| `--overwrite` | 显式替换重叠 M；保留其他已有 M |
| `--trust-remote-code` | 默认关闭；显式授权执行模型仓库代码 |
| `--preview` | 输出 shape、来源层、M、TP、布局和输出 dtype，不加载 CUDA |
| `--check-environment` | 检查 CUDA 依赖/设备及实际需要的 FP8 symbol/helper/signature |
| `--verify-installed` | 比较保存文件与已安装 loader 结果并运行其公开 Triton wrapper；不验证 serving backend 选择 |
| `--seed` | 非负合成输入种子，默认 `0` |
| `--measurements` | CUDA event 测量轮数，默认 `5` |
| `--calls-per-event` | 每个 event 内实际 kernel 调用数，默认 `1`；显式 >1 测量 repeated-call 平均值 |

M 是 GEMM 的 M 维度，不一定等于并发 serving request 数。TP 改变目标模型切分尺寸；调优 worker 数独立取 `min(可见 CUDA GPU 数, 请求 M 数)`。只有一个 M 时没有可并行的 M 任务，因此使用一个 GPU。以 M 为成本 proxy，用 deterministic LPT/greedy 分配，各 worker 内 M 按升序记录。参与 GPU 必须同型号。重复 shape 只调一次；worker 空任务、缺失、重复或意外结果都会在保存前中止。

QKV 使用 `local_q = Q_heads / TP`、`local_kv = max(1, KV_heads / TP)`，`N = (local_q + 2*local_kv)*head_dim`、`K = hidden_size`。Attention output 的 `K = local_q*head_dim`，不必等于 `hidden_size/TP`。严格验证 Q/KV 分片或复制关系。普通融合 gate/up 使用 `N = 2*intermediate_size/TP`，仅在对应架构实际构造该 MLP 时生成。不会从 `moe_intermediate_size` 推测 routed expert 尺寸。也会检查 runtime K 分组和 fused partition block 对齐。自动 adapter 当前要求方形 FP8 block：核对的 `Fp8Config` 用 block N 生成激活分组，而目标 GEMM 预期 block K。非方形布局必须使用显式观测的 `--shape`，并确认真实调用方提供正确 scales。

Wrapper 接受 `MODEL TP BLOCK_N BLOCK_K`，四个位置参数后可跟额外 CLI flag。支持环境变量 `PYTHON`、`SAVE_PATH`、`OUT_DTYPE`、`INPUT_TYPE`、`BATCH_SIZE` 和显式 `TRUST_REMOTE_CODE=1`。所有 wrapper 默认关闭 remote code。例如预览：

```bash
bash scripts/tune_custom.sh Qwen/Qwen3-8B 4 128 128 --preview
```

批量示例 `examples/tune_qwen3_models.sh` 将模型/TP 任务隔离到 `tuned_configs/batch/<模型名中的斜杠替换为下划线>/tp_<TP>/`。在此 batch runner 中，`SAVE_PATH` 或 `--save-path` 指定的是**根目录**，CLI 优先。例如 Qwen3-8B TP=4 输出到 `tuned_configs/batch/Qwen_Qwen3-8B/tp_4/`。重跑同一任务仍需显式 `--overwrite`。安装时从目标任务目录复制配置，不要从 batch 根目录复制。

## 测量与保存

每个候选先编译并预热五次，再以预分配输出 buffer 进行多轮 eager CUDA event 测量。默认每个 event 仅执行一次 GEMM，直接换算微秒，不额外除以调用数。显式 `--calls-per-event >1` 测量 repeated-call 平均值，换算为 `elapsed_ms * 1000 / calls_per_event`。重复调用会改变 workload/cache 行为，可能改变配置排名，不能与单次调用 latency 混为一谈。搜索按中位数排序；成功候选中的前三名，以及源码核对的默认配置（可运行时），在同一批 tensor 上独立复测，并与同一 A/B/scales 反量化后的 FP32 matmul 比较（`rtol=0.02`、`atol=0.02`，要求有限值）。复测中位数最低者胜出，默认配置也可能胜出。默认配置出现 `OutOfResources` 时不再重试，报告记录 `baseline.status=unavailable` 及原因；可运行的默认配置记录 `baseline.status=validated` 和独立复测结果。仅跳过 Triton `OutOfResources` 候选；未知编译/runtime/数值错误直接失败。这是 sanity check，不证明模型精度或全局最优。

成功运行另存 `reports/<UTC timestamp>-<id>.json`，记录 seed、软件/kernel 源码 hash、设备、shape/来源层、输出 dtype、量化布局、finalist 样本和误差、默认配置对照、资源不足计数。官方 config 只存 M 到 launch config 的映射。

文件名调用**已安装 vLLM 的设备名 helper**，格式严格为：

```text
N={N},K={K},device_name={normalized device},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json
```

每个文件从读取到合并/保存均持有文件锁；先写同目录临时 JSON，再 flush/fsync/close，最后 atomic replace。默认合并不重叠的 M；重叠 M 在调优前拒绝。`--overwrite` 仅替换请求的 M。已有 JSON 损坏或不兼容时，即使指定 `--overwrite` 也拒绝。不静默把完整文件缩成单个 M。`.lock` 文件是正常产物，安装时只复制顶层 `.json`。原子性按文件提供，不是跨所有 shape 的事务；后续文件写失败时，先前完整文件可能已保留，但不会输出成功。文件系统须支持 advisory lock 和 atomic replace。

官方文件名不区分输出 dtype、TP、模型或 vLLM 版本。不同软件/kernel 身份或输出 dtype 应使用不同 `--save-path`；仅合并针对相同 runtime/layout 的运行。同名不能证明对不同 vLLM kernel 仍有性能兼容性。

## 在目标 CUDA 主机安装与验证

使用实际命令输出的目录。单模型 wrapper 默认仓库根目录 `tuned_configs/`。使用 Python 默认目录时，在仓库根目录执行：

```bash
CONFIG_DIR=$(python3 -c 'from pathlib import Path; from vllm.model_executor.layers.quantization.utils import fp8_utils; print(Path(fp8_utils.__file__).resolve().parent / "configs")')
cp ./tuned_configs/*.json "$CONFIG_DIR/"

# 新进程：真实已安装 loader、请求 M 覆盖以及公开 Triton wrapper
python3 benchmark_w8a8_block_fp8.py --shape 128 256 --batch-size 17 \
  --save-path ./tuned_configs --verify-installed
```

只有 N、K、规范化 device name、`fp8_w8a8`、block shape 与 loader 的真实调用匹配，且 serving 选中普通 linear Triton backend 时，配置才生效。Loader 选择最接近的已保存 M；报告仅记录真实测量的 M。安装后应重启 serving，因为 loader 缓存配置。Backend 可能选 CUTLASS/DeepGEMM/FlashInfer 等其他内核；复制 JSON 不能证明被使用。应在选定 vLLM build 上核验 backend 和真实模型调用，再以相同条件比较 serving 性能。本仓库目前不声明端到端 serving 性能收益。

源码契约于 2026-10-08 对照 vLLM main commit [`c741bfca70cfb777e2016f827eae31f6e215fe9f`](https://github.com/vllm-project/vllm/tree/c741bfca70cfb777e2016f827eae31f6e215fe9f)。这是源码兼容性参考，**不是** GPU 实测版本保证。[验证记录](docs/VALIDATION.md)包含固定 kernel/adapter/loader 来源和剩余硬件验收项。

真实 CUDA 验收是独立于 CPU CI 的**合并 gate**；硬件 gate 未满足前 PR 保持 Draft。完整 preflight、小 shape 调优、installed-loader/public-wrapper、官方模型 preview 以及 1/10 calls 对照命令见 [CUDA 验收步骤](docs/CUDA_VALIDATION.md)。

## 测试与旧入口迁移

```bash
python3 -m pytest -q
# 可选真实 CUDA：编译/调优、3种输出 dtype、M=1/17/64、N尾块、官方 loader/wrapper
python3 -m pytest -q tests/test_gpu.py
```

可选 GPU 套件在临时目录隔离已安装 loader 的配置路径，不修改已安装 vLLM 源码。本机为 **Not executed — CUDA GPU unavailable**（同时未安装 PyTorch/vLLM）。CPU unit/源码契约及 Shell 子进程结果见 [docs/VALIDATION.md](docs/VALIDATION.md)，不编造 GPU 提速数据。

`benchmark_w8a8_block_fp8_qwencoder.py` 改为 deprecated 薄 wrapper，要求与主入口相同的显式来源参数。`scripts/tune_deepseek_v3.sh` 现在非零退出并提示改用观测的 `--shape N K`。旧 `...qwen3_30b.py` 和 `...qwen3omni_talker.py` 实际运行 INT8 W8A8，与文件名不符；保留历史标记，不重定向成 FP8。[README_AWQ.md](README_AWQ.md) 是历史文档，其自定义 JSON 在核对的 vLLM AWQ runtime 中没有自动消费者。

## 仓库结构

```text
benchmark_w8a8_block_fp8.py                # 正式 CUDA tuner/CLI
fp8_tuning.py                             # CPU shape/合并/保存 helper
benchmark_w8a8_block_fp8_qwencoder.py       # Deprecated FP8 薄 wrapper
benchmark_w8a8_block_int8.py                # 历史，非维护
benchmark_awq_w4a16.py                     # 历史，非维护
benchmark_w8a8_block_fp8_qwen3_30b.py        # 历史 INT8，文件名有误导
benchmark_w8a8_block_fp8_qwen3omni_talker.py # 历史 INT8，文件名有误导
scripts/                                  # 维护入口/环境检查，及 DeepSeek 停用提示
examples/tune_qwen3_models.sh              # 批量 runner，任一失败最终非零
tests/                                   # CPU、Shell、源码契约、可选 GPU 测试
docs/VALIDATION.md                        # 修复证据与第二轮自查
.github/workflows/tests.yml               # CPU/Shell CI
README.md / README_zh.md / README_AWQ.md
LICENSE / NOTICE
```

完整 Apache License 2.0 文本见 [LICENSE](LICENSE)；版权与来源说明保留于 [NOTICE](NOTICE) 和 SPDX 文件头。
