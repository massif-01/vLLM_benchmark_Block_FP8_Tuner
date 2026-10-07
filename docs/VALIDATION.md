# 本轮修复与验证记录

核验日期：2026-10-08（Asia/Shanghai）。仓库起点 HEAD：`03af15286e5e2c643f8bbcd61bc64bbc43dfbd90`；开始时工作区干净。本地修复验收时，修改保留在工作区，尚未 commit/push。随后用户授权将这些改动通过独立分支 PR 提交到自己的仓库供进一步审查；不直接提交到默认分支、不合并 PR。未修改任何 vLLM checkout，未创建 vLLM upstream issue/PR。

初始修复以用户 Goal 为最高优先级；后续独立 Review 修复以对应的新指令为准，不重新执行原 Goal。实施说明和审查报告仅作为参考/证据：CUDA-only 是产品边界；只维护 W8A8 Block FP8 主路径；INT8/AWQ 不做功能开发。

## 独立核对来源

GitHub API 查询当前 vLLM main 后，固定以下 commit 下载并检查相关源码：`c741bfca70cfb777e2016f827eae31f6e215fe9f`。没有用旧报告的 vLLM SHA 替代本次核验。

- [FP8 kernel、loader、默认配置、block shape 校验](https://github.com/vllm-project/vllm/blob/c741bfca70cfb777e2016f827eae31f6e215fe9f/vllm/model_executor/layers/quantization/utils/fp8_utils.py)：确认 launch 参数/stride、每个 K tile 单 scale group、文件名、closest-M 选择、缓存和默认配置。
- [设备名 helper](https://github.com/vllm-project/vllm/blob/c741bfca70cfb777e2016f827eae31f6e215fe9f/vllm/utils/platform_utils.py)：统一空白/斜杠；正式代码直接调用已安装版本，不自行替代。
- [Qwen3](https://github.com/vllm-project/vllm/blob/c741bfca70cfb777e2016f827eae31f6e215fe9f/vllm/model_executor/models/qwen3.py)、[Qwen2 普通 MLP](https://github.com/vllm-project/vllm/blob/c741bfca70cfb777e2016f827eae31f6e215fe9f/vllm/model_executor/models/qwen2.py)、[TP linear](https://github.com/vllm-project/vllm/blob/c741bfca70cfb777e2016f827eae31f6e215fe9f/vllm/model_executor/layers/linear.py)：核对 GQA/KV 复制、非 hidden-size 等式的 head_dim、融合 gate/up 和 row-parallel down。
- [Qwen3 MoE](https://github.com/vllm-project/vllm/blob/c741bfca70cfb777e2016f827eae31f6e215fe9f/vllm/model_executor/models/qwen3_moe.py)：依据 `mlp_only_layers`、`decoder_sparse_step`、`num_experts` 确认 dense 层是否存在；shared MLP 仅在 sparse 层且 shared size >0 时存在；routed experts 进入 FusedMoEFactory，未纳入 tuner。
- [FP8 量化配置](https://github.com/vllm-project/vllm/blob/c741bfca70cfb777e2016f827eae31f6e215fe9f/vllm/model_executor/layers/quantization/fp8.py)、[Triton regular-linear backend](https://github.com/vllm-project/vllm/blob/c741bfca70cfb777e2016f827eae31f6e215fe9f/vllm/model_executor/kernels/linear/scaled_mm/triton.py)、[block-linear 基类](https://github.com/vllm-project/vllm/blob/c741bfca70cfb777e2016f827eae31f6e215fe9f/vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py)：配置和实际后端选择是不同的条件；自动 shape 不证明 serving 采用 Triton。

`tests/fixtures/vllm_fp8_contract.py` 只包含固定来源的 helper/loader/default/signature 契约摘录，供 CPU 测试。不是部署内核，不是导入 fallback，也不是实际已安装 vLLM 集成验证。

## 问题映射及处理边界

| 审查线索 | 本轮结果 / 依据 |
| --- | --- |
| B01 attention shape | 修复：Q、KV、head_dim、TP 合法性，TP>KV 的精确复制；QKV 和 o_proj 分别推导；TP1/2/4/8 回归 |
| B02 config 字段不等于真实层 | 修复：仅两个明确 architecture adapter；MoE layer schedule 决定普通 dense/shared MLP；不读取 `moe_intermediate_size` 生成 routed shapes；未知/嵌套架构拒绝 |
| B03 专用入口身份 | qwencoder FP8 改为主入口薄 wrapper；另两个误名文件实际 INT8，醒目标记历史/非维护，不改量化身份、不 redirect 成 FP8 |
| B04 INT8 兼容性 | 不开发：用户排除正式维护范围，标记非维护，撤出主文档推荐 |
| B05/B06 AWQ 消费/搜索问题 | 不开发：标记非维护，历史文档删除“复制 JSON 即自动消费”承诺；未恢复 AWQ 能力 |
| B07 多 GPU 文件覆盖 | 主脚本 worker 返回、主进程统一写文件本来正确，保留；严格校验每个 worker 的 shape/M 集，防 missing/duplicate/unexpected，保留同型号 GPU 约束 |
| B08 跨运行/中断保存 | 修复：默认合并 disjoint M、拒绝 overlap；显式 overwrite 仅替换请求 M；读取坏文件拒绝；文件锁+同目录 temp/fsync/replace；并发/故障注入测试 |
| B09 十倍计时与正确性 | 修复：真实 calls/event 与微秒换算；五次 warmup、多轮中位数、成功候选前三名+可运行 default 独立复测；资源不足 default 不重试，报告记录 unavailable/原因；同量化 tensor/scales 的 FP32 reference；有限值和 tolerance gate；GPU 尚未执行 |
| B10 fallback/退出码 | 修复：无默认架构、无 catch 后 DeepSeek fallback；wrapper 非零传播；batch runner 按模型/TP 隔离输出，继续其余任务但任一失败最终非零；预览不打印调优成功 |
| B11 remote code | 修复：配置加载和全部 wrapper 默认关闭，CLI 或 `TRUST_REMOTE_CODE=1` 显式启用 |
| B12 参数/layout/preflight | 修复：正数、合法 TP、K grouping、fused block 对齐、自动 adapter 方形 block 限制（当前 Fp8Config 的激活分组约束）、quant metadata；只 gate CUDA FP8 实际 imports/helper/signature/native FP8 SM；GPU 编译仍需实测 |
| B13 文件名/环境身份/消费 | 文件名和 schema 用固定官方 helper/loader CPU 契约核对；调用已安装 helper；报告记录软件/kernel 身份；新增真实 `--verify-installed` 路径但本机未执行 |
| B14 重复调优/可恢复性 | shape 去重，worker 数不超过 M 数，单 M 直接单 GPU；M 使用 deterministic LPT/greedy 分配；重跑可分 M 合并；未建立 checkpoint/resume 系统，不扩展 scope |
| B15 文档 | README/README_zh 对齐；撤下营销式模型列表/任意 custom 支持；明确 legacy、TP/M、backend gate、输出路径、覆盖策略、证据级别；AWQ 文档历史说明 |
| B16 测试基线 | 新增 CPU/Shell/source-contract/可选 CUDA suite，以及 CPU GitHub Actions（Python3.10/3.13）；本机只执行 Python3.13，本地验收时远程 CI 尚未运行；PR 的最新 CI 状态以 GitHub Checks 为准 |
| B17 License/卫生 | 使用完整官方 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)，保留 SPDX/来源并增 NOTICE；取消忽略所有 JSON；tree 对应实际结构 |

## 真实测试结果

本机：macOS，Python `3.13.13`，未安装 `torch`、`vllm`、`triton`、`transformers`；无可用 NVIDIA CUDA 环境。

| 检查 | 结果 |
| --- | --- |
| `python3 -m pytest -q` | **160 passed, 1 skipped**；约16秒。138项 CPU/unit/源码契约和22项 Shell/迁移子进程测试；GPU 模块因 PyTorch 缺失整组 skip |
| CPU shape/layout | TP1/2/4/8、head_dim 与 hidden 不同、KV replication、非法 TP/dimensions、fused gate/up、MoE dense/shared 存在性、routed 排除、未知/嵌套架构、配置加载失败、quant metadata/block/ignored layers |
| CPU 调度/保存 | GPU>M、单 M、empty/duplicate 分配、完整/缺失 worker merge、单/多 worker exception、heterogeneous GPU 拒绝、shape 去重、成功完整保存+报告 |
| CPU 数据安全 | disjoint merge、overlap 拒绝、显式覆盖保留其他 M、坏 JSON/重复 key 拒绝、dump/fsync/replace 故障不破坏原文件、8进程并发合并无丢失 |
| CPU 计时/loader 契约 | event 中真实调用数与微秒算术；默认配置资源不足不重试、可运行默认独立复测/可胜出、全部候选失败/未知错误/correctness 错误传播；固定官方 default/signature/helper/loader；空白与斜杠规范化；真实主代码 cached-loader 源码身份读取 |
| Shell | 单任务/env wrapper 传播 exit7、remote code 默认关闭/显式开启/非法值拒绝、batch 8任务继续并累计失败、batch 全成功；实际 shape/merge/writer 链路验证8个模型/TP任务隔离（默认根目录、环境变量、两种CLI形式）、重跑拒绝和显式 overwrite；DeepSeek 退出2、deprecated FP8 wrapper 预览 |
| `python3 -m compileall -q ...` | 主入口/helper/薄wrapper/测试编译通过 |
| `bash -n scripts/... examples/...` | 全部6个 Shell 文件语法检查通过 |
| `git diff --check` | 通过 |
| 主 CLI `--help`、显式 shape `--preview` | exit0；无需 CUDA/PyTorch/vLLM |
| 本机 `scripts/environment_check.sh` | exit1：`Error: No module named 'torch'`，正确暴露环境不满足 |
| 本机单 shape 实际 tuning 命令 | exit1：同上，没有调优成功信息或实际 GPU config |
| GPU compile/correctness/tuning | **Not executed — CUDA GPU unavailable** |
| 已安装 vLLM loader + public Triton wrapper | **Not executed — CUDA GPU unavailable**；CPU 官方源码 fixture 不能代替此项 |
| 实际 FP8 模型 backend/serving 对照性能 | **Not executed — CUDA GPU unavailable** |
| GitHub Actions / Python3.10 环境 | 本地验收时尚未运行；已新增 workflow，PR 的最新远程 CI 状态以 GitHub Checks 为准 |

CPU 实测没有被包装成 GPU 或性能验收。没有生成、提供或宣称真实 GPU 提速数据。

## 第二轮自查

从实际最终实现、测试和两份 README 重新检查以下项：

1. 自动 shape 只保留两个 adapter；无 generic/name substring/fallback；QKV、o_proj、fused MLP、MoE schedule 对应固定源码。
2. Routed experts、router、LM head、shared gate 未被误写为正式优化范围；shared MLP 仍要求运行时真的选到 regular-linear Triton。
3. Python exception、pool.map failure、缺失结果、Shell failure、report 写失败均不能进入成功输出；help/preview 不称 tuning success。
4. 保存前全 worker 结果覆盖校验；跨运行 lock 保护 read/merge/write；原子失败保留旧 JSON。原子性是每个文件，不承诺跨 shape 事务。
5. 文件名直接复用 installed helper；CPU fixture loader 能读取生成 schema；实际 GPU loader/wrapper 验证仍列为未执行。
6. README 说明真实 CLI、默认目录、TP/M、dtype/layout、量化 metadata、锁文件、overlap 政策、closest M、loader cache/backend 条件。
7. INT8/AWQ 从正式支持和 Quick Start 撤下，未开发其 kernel/调度/消费链路。
8. 没有增加、适配或重构 ROCm/XPU；CUDA-only 明确保留。
9. 仅增加一个 CPU helper、必要测试/报告/CI；未建立通用架构平台、数据库、大型 benchmark 框架或上游工作流。
10. 测试覆盖本轮发现的风险。复查中另外修正了 Bash3 空数组 nounset、cached-loader `__code__` 读取、K grouping/fused partition 对齐、当前 Fp8Config 非方形 block 的激活分组限制，并补相应回归。

## Reviewer 发现后的定点修复

上一轮 73 项测试没有覆盖两个真实流程缺陷，不能作为这两项已正确的证据。本次按用户授权仅修复这两项：

- **Batch 输出冲突**：Qwen3-8B TP=4 的 gate/up `(6144,4096)` 与 TP=1 的 QKV 重合，TP=8 的 gate/up `(3072,4096)` 与 TP=2 的 QKV 重合。旧示例共享输出目录，即使首次运行也会失败。现在每个模型/TP 使用独立目录；`SAVE_PATH`/`--save-path` 是 batch 根目录（CLI优先）。新增子进程回归只 mock GPU worker/依赖，真实运行 shape、merge、writer，验证全部8任务、重跑拒绝及显式覆盖。
- **资源不足 default 重试**：旧搜索跳过 `OutOfResources` default 后仍把它加入 finalists，导致有效候选被整体失败丢弃。现在仅把可运行 default 加入复测；报告另有 `baseline.status=unavailable` 和错误原因，或 `validated` 与复测结果。新增回归验证不重试、default 在前三名之外仍复测/仍可胜出、没有有效候选及未知/数值错误继续失败。

先新增回归并确认旧实现失败，再修复。最新总计 **83 passed, 1 skipped**，Python 编译、全部6个 Shell 语法检查及 diff 检查通过；GPU 验证状态没有变化。此前“修复完成”的结论应以本次补充记录为准。

## PR #3 独立 Review 定点修复（本次）

保留现有 CPU helper + CUDA CLI 分层、parent aggregation、官方 filename/schema 和 safe persistence；仅修改 exclusion 分类、单-call 默认计时和 M 负载分配。

| Finding | 结果与证据 |
| --- | --- |
| P1 exclusion handling | **Fixed**：合并检查 `ignored_layers` 与 `modules_to_not_convert`；允许明确非目标模块/参数及 scoped `*`，拒绝 target projection、宽泛父级和无法识别的 pattern，指出具体 exclusion 和显式 shape 路径。router `mlp.gate` 与 target `mlp.gate_proj` 按完整组件区分；不加入 router/routed expert tuning |
| P2 calls_per_event | **Fixed**：函数和 parser 默认 1，保留显式 >1 repeated-call 平均测量；已有 1/10 call 微秒算术和手动 event 调用数测试保留，新增默认值断言 |
| P2 multi-GPU balancing | **Fixed**：M 成本 proxy 的 deterministic LPT/greedy，各 bin 内升序；默认18个M/8GPU的旧 loads 为 `[3,12,40,144,224,768,2560,9216]`，新 loads 为 `[4096,3072,2048,1536,1024,512,340,339]`，最大估计负载从9216降至4096；完整/唯一分配与 parent merge 回归通过 |
| Provenance 系统 | **Deferred**：保留独立目录警告；没有新增数据库、metadata framework、schema migration |
| Hardware gate | **Not executed — CUDA GPU unavailable**；保持 Draft，不满足 ready-for-merge 条件 |

官方配置仅下载 JSON，无模型权重/remote code，固定 revision [`dcaee4d4dfc5ee71ad501f01f530e5652438fde0`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/raw/dcaee4d4dfc5ee71ad501f01f530e5652438fde0/config.json)。保存为 `tests/fixtures/qwen3_coder_fp8_config.json`（SHA256 `2705ed03bb322c864470bc282738b4713daca31b54ee4a639fe198bd7b9523f4`）。离线读取其真实145项 exclusions，实际 planner/metadata/layout 校验通过，TP4 为 `(1280,2048)`、`(2048,1024)`；此结果**不是** vLLM CLI/model preview 或 GPU pass。

先新增回归，确认旧实现拒绝官方风格 exclusions、漏查第二字段、默认 calls=10、旧分配仍最大load9216；再修改实现。完整本地套件 **131 passed, 1 skipped**；Python3.13.13 编译、全部6个 Shell 语法、diff 检查通过。Python3.10 本机未安装，使用本次 PR head 的 GitHub CI 验证，最终状态以 Checks 为准。所有原有语义回归保留，原 distribution 的脆弱 exact-list 断言改为同输入的完整/唯一/worker数/deterministic 断言，另增 default-load 与 parent-merge 回归。

第二轮自审确认：官方 exclusions 允许；checkpoint/fused/shared projections 与父级范围仍拒绝；gate/gate_proj 区分；两字段均检查；default calls=1；LPT 分配确定且完整；aggregation/filename/schema/lock/temp/fsync/replace 没有改动；英中 README 一致；INT8/AWQ/ROCm/XPU 未扩展；未增加架构/manifest 系统；GPU 不隐含通过。

目标 GPU 上的 preflight、smoke、installed-loader/public-wrapper、官方 model preview、1/10 calls 对照的准确命令见 [CUDA_VALIDATION.md](CUDA_VALIDATION.md)。本机 actual vLLM CLI model preview 因 vLLM/PyTorch 未安装而未执行；单-call vs repeated-call GPU 的 winner/ranking/median 对照也未执行。

**Ready for independent re-review**：三项范围内代码修复和 CPU/Shell 回归已完成，硬件 gate 仍明确未通过；此结论不等于 ready for merge。

## 第二次独立 Review：model-aware output dtype（本次，pre-GPU）

本次只修 dtype 语义及必要的 CLI/worker/测试/文档联动，澄清 planner `*` convention；未重写上一轮 exclusions/LPT，未修改 parent aggregation、官方 JSON schema、filename 或 persistence。

源码重新核对 vLLM main `32fbfa15e8bacc64182cb1286831bd63d7e4fc12`：

- [`vllm/config/model.py`](https://github.com/vllm-project/vllm/blob/32fbfa15e8bacc64182cb1286831bd63d7e4fc12/vllm/config/model.py)：ModelConfig 调用 `_get_and_verify_dtype(model_id, config, dtype, *, is_pooling_model, revision=None, config_format="hf")`；auto 使用配置转换、平台 dtype policy 和 validity checks。
- [`model_arch_config_convertor.py`](https://github.com/vllm-project/vllm/blob/32fbfa15e8bacc64182cb1286831bd63d7e4fc12/vllm/transformers_utils/model_arch_config_convertor.py)：config dtype/metadata conversion 属于 vLLM，不在 tuner 重写。
- 常规 FP8 exclusion match mode 的 exact 语义与 planner 分类是不同的层；本工具对 scoped `*` 的分类只是 planner 安全分析，不声明 runtime glob 支持。

`resolve_model_out_dtype()` 为薄调用：校验所核对的 API bind，传入 model/config 和 auto，映射返回的 torch float16/bfloat16/float32 为规范字符串；API、返回类型或 resolver 错误均明确失败并提示显式 dtype，不 fallback。没有 ModelConfig/engine/权重实例化或 CUDA 初始化调用。`tests/fixtures/vllm_dtype_contract.py` 保留固定源码的 resolver 和转换 getter 供 CPU 契约测试，注入 fake torch/config/platform，**不是**生产 fallback 或实际安装环境/GPU 验证。

| 输入 | requested | resolved / source |
| --- | --- | --- |
| 官方 Qwen3-Coder BF16 fixture | auto | bfloat16 / vllm-model-auto |
| 同一 fixture 显式 override | float16 | float16 / explicit |
| BF16 dense Qwen3 | auto | bfloat16 / vllm-model-auto |
| FP16 Qwen3 | auto | float16 / vllm-model-auto |
| FP32 Qwen3（当前 CUDA SM80+ dtype policy） | auto | bfloat16 / vllm-model-auto，由 upstream resolver 选择平台首选 dtype；另测FP16-first policy返回float16 |
| 显式 shape | auto | 拒绝，提示 --out-dtype |
| 显式 shape | bfloat16 | bfloat16 / explicit |
| half override | half | float16 / explicit |

Parser 默认 auto，wrapper 未设置 OUT_DTYPE 时不传 dtype，不在模型 wrapper 硬编码 BF16。Args 保留原始 `out_dtype`，新增 `resolved_out_dtype`；plan/report 同时有 requested/resolved/canonical/source。tune 的 benchmark/reference/correctness 仅使用 resolved，verify 使用 plan 的 resolved，不重新读取 auto parser default。

验收：先新增 dtype 回归并确认旧 plan 缺 requested/source、shape 默认猜 FP16，再修复。完整本地 **160 passed, 1 skipped**，compileall、6个 Shell 语法和 diff 检查通过；其中138项 CPU/unit/source-contract、22项 Shell/迁移子进程测试。原 shape-workload 测试明确传 dtype，原 exclusions/LPT/merge/persistence 回归保留；新增9项 wrapper dtype forwarding 和17项 dtype语义/3项 worker-report、tune/reference/correctness、verify 的 CPU flow 回归。

独立实际 CLI 验证（本机没有 PyTorch/vLLM，无 CUDA）：

```text
--shape 128 256 --preview -> exit 1，明确提示 --out-dtype
--shape 128 256 --out-dtype bfloat16 --preview -> exit 0，requested/resolved=bfloat16，source=explicit
```

Actual model preview 因 vLLM config loading 不可用而未执行；官方 fixture 的全 plan/source-contract 回归不能冒充 actual preview。本轮按指令 **CUDA / installed loader / serving — Not executed**，GPU acceptance deliberately deferred。GPU 文档保留显式 FP16 smoke、追加 BF16 smoke，官方 model preview 同时检查 shape 与 BF16/source，另有显式 FP16 override preview。

第二轮自审确认：auto 调用 installed resolver；official/dense BF16、FP16、explicit cast、half 和失败路径均覆盖；shape 不猜 dtype；wrapper 不注入FP16；requested/resolved/source 未丢失；worker/correctness/verify canonical dtype 一致；英中 README 与 CUDA验收条件一致；planner wildcard 注释不冒充 runtime semantics；gate/gate_proj/bias/routed边界、LPT、parent merge、persistence 的功能未改；未执行本轮禁止的 GPU 验收。

**Ready for independent pre-GPU re-review**。PR 继续 Draft；不是 ready to merge。Python3.10/3.13 对本次 head 的最新 CI 状态以 PR Checks 为准。

## 剩余风险与下一阶段条件

1. 必须在目标 NVIDIA CUDA 主机执行 `tests/test_gpu.py`，对具体 vLLM build/设备完成编译、数值验证和至少一次真实调优。当前 tolerance 和 eager event 测量可靠性只有设计/CPU 算术证据，没有 GPU 实测。
2. 安装配置后，在新进程执行 README 的 `--verify-installed`，随后确认真实模型服务选择 `TritonFp8BlockScaledMMKernel` 并消费目标 shapes/config；其他 backend 不一定读本配置。
3. 不同 vLLM kernel、output dtype、设备/layout 的性能兼容性不能仅靠同名 JSON 保证；用独立输出目录并查看报告身份。当前只做语法/layout 合并兼容检查，未做跨版本性能兼容认证。
4. 写入按文件原子；文件系统需支持 advisory lock/atomic replace。报告或后续 shape 保存失败可能留下此前已完成配置，流程返回失败且不声称整体成功。

**Not ready for upstream PR investigation**：代码修复、CPU/Shell 回归和第二轮自查已完成，但缺真实 CUDA kernel/loader/serving 证据，尚不能声称目标 GPU 路径完成验收。没有创建 upstream PR。
