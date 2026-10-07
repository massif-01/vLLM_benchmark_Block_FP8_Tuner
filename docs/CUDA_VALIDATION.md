# CUDA acceptance gate for PR #3 / CUDA 验收步骤

**Current round: Not executed — GPU acceptance deliberately deferred.**

No CUDA benchmark, installed-loader or serving acceptance is executed in this round. The commands below remain a later hardware gate; current local hardware is also unavailable.

CPU/unit, Shell and source-config fixture results do not satisfy this gate. PR #3 stays Draft; these commands are for the target NVIDIA CUDA host. Run from the repository root in the CUDA PyTorch/vLLM environment. No model weights are needed for the explicit-shape smoke.

## 1. Preflight / 环境检查

```bash
bash scripts/environment_check.sh
```

Require success, CUDA availability, compatible vLLM FP8 symbols/helper/signature and native-FP8 GPU capability >=8.9. Stop if preflight fails.

## 2. Small explicit-shape smoke / 小 shape 验收

```bash
python3 benchmark_w8a8_block_fp8.py \
  --shape 128 256 --batch-size 17 --out-dtype float16 \
  --seed 0 --measurements 3 --calls-per-event 1 \
  --save-path ./tuned_configs/smoke/calls_1
```

This must actually complete compilation/search, finalist correctness, winner selection, official JSON and report saving. This shape fits the default 128x128 layout; it is not a full model sweep. A fresh directory is assumed. Reruns with the same M require explicit `--overwrite`.

### Additional BF16 smoke / 追加 BF16 验收

Explicit shape cannot use auto dtype. In the later GPU stage, also execute BF16 once in a separate directory:

```bash
python3 benchmark_w8a8_block_fp8.py \
  --shape 128 256 --batch-size 17 --out-dtype bfloat16 \
  --seed 0 --measurements 3 --calls-per-event 1 \
  --save-path ./tuned_configs/smoke/bf16
```

Require compilation/search, finalist correctness, winner, JSON and report success for BF16 as well. This command is documented only, not executed now.

## 3. Installed loader / public wrapper

Install only the single-call smoke's top-level official JSON, then verify in a **new process**:

```bash
CONFIG_DIR=$(python3 -c 'from pathlib import Path; from vllm.model_executor.layers.quantization.utils import fp8_utils; print(Path(fp8_utils.__file__).resolve().parent / "configs")')
cp ./tuned_configs/smoke/calls_1/*.json "$CONFIG_DIR/"

python3 benchmark_w8a8_block_fp8.py \
  --shape 128 256 --batch-size 17 --out-dtype float16 \
  --seed 0 --save-path ./tuned_configs/smoke/calls_1 --verify-installed
```

Require the official loader to return the same content/M and the public Triton wrapper to pass the numerical reference. This does not prove that a model-serving engine selects this backend.

## 4. Official Qwen3-Coder FP8 planning / 官方模型预览

If a local checkpoint/config directory is already available, set `MODEL_CONFIG` to it. Otherwise this command loads the official repository's config, not its model weights. Remote code remains off.

```bash
python3 benchmark_w8a8_block_fp8.py \
  --model "${MODEL_CONFIG:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}" \
  --tp-size 4 --preview
```

For the reviewed official config, expected regular-linear shapes are `(1280,2048)` and `(2048,1024)`. Its 145 LM-head/layernorm/router exclusions must not cause rejection. Also require `requested_out_dtype="auto"`, `resolved_out_dtype="bfloat16"`, canonical `out_dtype="bfloat16"` and `out_dtype_source="vllm-model-auto"`. Shapes alone are not sufficient. Routed expert, router and shared-gate tuning remain excluded. Record the actual config revision/path and result; a CPU fixture test is not this CLI integration check.

### Explicit override preview / 显式 override 预览

This is also a config-only check, not a CUDA benchmark. With vLLM config loading available:

```bash
python3 benchmark_w8a8_block_fp8.py \
  --model "${MODEL_CONFIG:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}" \
  --tp-size 4 --out-dtype float16 --preview
```

Require `requested_out_dtype="float16"`, `resolved_out_dtype="float16"`, `out_dtype="float16"`, `out_dtype_source="explicit"`; the BF16 checkpoint must not override the user's FP16 request.

## 5. Single-call vs repeated-call comparison / 1 与 10 calls 对照

Keep M/N/K, full search space, seed, output dtype and measurement rounds identical. The single-call run above is the first half. Run the other half in a separate output directory:

```bash
python3 benchmark_w8a8_block_fp8.py \
  --shape 128 256 --batch-size 17 --out-dtype float16 \
  --seed 0 --measurements 3 --calls-per-event 10 \
  --save-path ./tuned_configs/smoke/calls_10
```

Compare the saved reports without replacing the installed single-call config:

```bash
python3 - <<'PY'
import json
from pathlib import Path
results = {}
for calls in (1, 10):
    folder = Path(f'tuned_configs/smoke/calls_{calls}/reports')
    report = json.loads(max(folder.glob('*.json'), key=lambda p: p.stat().st_mtime).read_text())
    assert report['arguments']['calls_per_event'] == calls
    measurement = report['workers'][0]['measurements'][0]
    results[calls] = measurement
    print(f'calls/event={calls}')
    print('winner:', measurement['winner']['config'])
    print('median_us:', measurement['winner']['timing']['median_us'])
    print('remeasured finalist ranking:')
    for entry in sorted(measurement['finalists'], key=lambda r: r['timing']['median_us']):
        print(entry['timing']['median_us'], entry['config'])
print('same winner:', results[1]['winner']['config'] == results[10]['winner']['config'])
PY
```

The report ranks remeasured finalists, not the entire search history. Record winner agreement/disagreement, finalist rankings and median changes. Do not hide different winners or treat repeated-call averages as single-call latency. Preserve reports with device/software/kernel identity and any failed gate; do not fabricate acceptance results.
