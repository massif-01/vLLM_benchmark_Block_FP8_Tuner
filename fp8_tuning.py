# SPDX-License-Identifier: Apache-2.0
# Repository-specific shape planning, result validation and safe persistence.
"""CPU-only helpers. Importing this module never initializes CUDA or vLLM."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512,
                       1024, 1536, 2048, 3072, 4096]
CONFIG_KEYS = {'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'GROUP_SIZE_M',
               'num_warps', 'num_stages'}
SUPPORTED_ARCHITECTURES = ('Qwen3ForCausalLM', 'Qwen3MoeForCausalLM')


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return value


def field(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def unique_shapes(shapes):
    result = []
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError('Each shape must be (N, K)')
        shape = tuple(positive(v, axis) for v, axis in zip(shape, ('N', 'K')))
        if shape not in result:
            result.append(shape)
    if not result:
        raise ValueError('No weight shapes to tune')
    return result


def model_shapes(config, tp_size):
    """Mirror Qwen3 regular linear construction, never infer routed experts."""
    positive(tp_size, 'tp_size')
    architectures = field(config, 'architectures')
    if not isinstance(architectures, (list, tuple)) or len(architectures) != 1:
        raise ValueError('Auto shape detection requires one explicit architecture')
    arch = architectures[0]
    if arch not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f'Auto shape detection does not support architecture {arch!r}; use --shape N K')
    if any(field(config, key) is not None for key in ('text_config', 'thinker_config', 'talker_config')):
        raise ValueError('Nested model configs are not supported by auto shape detection')
    hidden = positive(field(config, 'hidden_size'), 'hidden_size')
    q = positive(field(config, 'num_attention_heads'), 'num_attention_heads')
    kv = positive(field(config, 'num_key_value_heads'), 'num_key_value_heads')
    if q % kv or kv > q:
        raise ValueError('Q heads must be divisible by KV heads')
    head = field(config, 'head_dim')
    if head is None:
        if hidden % q:
            raise ValueError('hidden_size must be divisible by Q heads when head_dim is absent')
        head = hidden // q
    positive(head, 'head_dim')
    if q % tp_size or (kv >= tp_size and kv % tp_size) or (kv < tp_size and tp_size % kv):
        raise ValueError('Illegal TP: Q heads must be divisible by TP; KV heads must partition or replicate exactly')
    local_q, local_kv = q // tp_size, max(1, kv // tp_size)
    shapes = [((local_q + 2 * local_kv) * head, hidden), (hidden, local_q * head)]
    sources = ['self_attn.qkv_proj', 'self_attn.o_proj']

    def mlp(intermediate, source):
        positive(intermediate, source + '.intermediate_size')
        if intermediate % tp_size:
            raise ValueError(f'{source} intermediate_size must be divisible by TP')
        shapes.extend([(2 * (intermediate // tp_size), hidden), (hidden, intermediate // tp_size)])
        sources.extend([source + '.gate_up_proj', source + '.down_proj'])

    if arch == 'Qwen3ForCausalLM':
        mlp(field(config, 'intermediate_size'), 'mlp')
    else:
        layers = positive(field(config, 'num_hidden_layers'), 'num_hidden_layers')
        experts = field(config, 'num_experts')
        if isinstance(experts, bool) or not isinstance(experts, int) or experts < 0:
            raise ValueError('num_experts must be a nonnegative integer')
        step = positive(field(config, 'decoder_sparse_step'), 'decoder_sparse_step')
        only = field(config, 'mlp_only_layers', [])
        if not isinstance(only, (list, tuple)) or any(type(i) is not int or not 0 <= i < layers for i in only):
            raise ValueError('Invalid mlp_only_layers')
        sparse = [i for i in range(layers) if i not in only and experts > 0 and (i + 1) % step == 0]
        if sparse and tp_size > experts:
            raise ValueError('TP cannot exceed the number of routed experts in this model')
        if len(sparse) < layers:
            mlp(field(config, 'intermediate_size'), 'mlp (dense layers)')
        shared = field(config, 'shared_expert_intermediate_size', 0)
        if type(shared) is not int or shared < 0:
            raise ValueError('shared_expert_intermediate_size must be nonnegative')
        if sparse and shared:
            mlp(shared, 'mlp.shared_expert')
    return unique_shapes(shapes), [{'shape': list(s), 'layer': src} for s, src in zip(shapes, sources)]


def load_model_shapes(model, tp_size, trust_remote_code=False, loader=None):
    if loader is None:
        from vllm.transformers_utils.config import get_config
        loader = get_config
    try:
        config = loader(model=model, trust_remote_code=trust_remote_code)
    except Exception as exc:
        raise RuntimeError(f'Cannot load model config for {model!r}: {exc}') from exc
    shapes, sources = model_shapes(config, tp_size)
    return config, shapes, sources


def validate_model_quantization(config, block_n, block_k):
    # Source-audited Fp8Config uses block_n for activation grouping, while the
    # target GEMM expects block_k. Do not promise its non-square runtime path.
    if block_n != block_k:
        raise ValueError('Automatic adapters require square FP8 blocks; use observed --shape for verified runtime layouts')
    quant = field(config, 'quantization_config')
    if not quant:
        return 'No checkpoint quantization metadata; shapes describe regular linears only. Confirm FP8/backend at runtime.'
    if field(quant, 'quant_method') != 'fp8' or field(quant, 'activation_scheme') != 'dynamic':
        raise ValueError('Auto detection requires dynamic W8A8 FP8 checkpoint metadata, or no quantization metadata')
    if field(quant, 'weight_block_size') != [block_n, block_k]:
        raise ValueError('Checkpoint weight_block_size does not match --block-n/--block-k')
    # An ignored regular layer can remove a shape. Refuse rather than guess the mapping.
    ignored = field(quant, 'ignored_layers', []) or field(quant, 'modules_to_not_convert', [])
    if ignored and ignored != ['lm_head']:
        raise ValueError('Auto detection does not map ignored layer patterns; use observed --shape N K')
    return 'FP8 metadata matches; runtime must still select the Triton regular-linear backend.'


def distribute_batch_sizes(batch_sizes, num_gpus):
    positive(num_gpus, 'num_gpus')
    if not batch_sizes or len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError('Batch sizes must be nonempty and unique')
    for m in batch_sizes:
        positive(m, 'M')
    workers = min(num_gpus, len(batch_sizes))
    return [batch_sizes[i * len(batch_sizes) // workers:(i + 1) * len(batch_sizes) // workers]
            for i in range(workers)]


def validate_launch(config, block_k=None):
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError('Invalid Triton config keys')
    for key, value in config.items():
        positive(value, key)
    for key in ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K'):
        value = config[key]
        if value & (value - 1):
            raise ValueError(f'{key} must be a power of two')
    if config['num_warps'] not in (4, 8):
        raise ValueError('num_warps must be 4 or 8')
    if config['BLOCK_SIZE_K'] < 32 or (block_k and block_k % config['BLOCK_SIZE_K']):
        raise ValueError('BLOCK_SIZE_K must divide the quantization block_k (and be >=32)')
    return config


def merge_results(partials, assignments, shapes):
    shapes = unique_shapes(shapes)
    if len(partials) != len(assignments) or not assignments:
        raise RuntimeError('Worker count does not match assignments')
    result = {shape: {} for shape in shapes}
    expected = [m for batch in assignments for m in batch]
    if not expected or len(set(expected)) != len(expected) or any(not b for b in assignments):
        raise RuntimeError('Empty or duplicate worker assignments')
    for partial, assigned in zip(partials, assignments):
        if not isinstance(partial, dict) or set(partial) != set(shapes):
            raise RuntimeError('Worker returned missing or unexpected shapes')
        for shape in shapes:
            configs = partial[shape]
            if not isinstance(configs, dict) or set(configs) != set(assigned):
                raise RuntimeError(f'Worker returned incomplete/unexpected M for {shape}')
            for m, config in configs.items():
                if m in result[shape]:
                    raise RuntimeError(f'Duplicate result for {shape}, M={m}')
                validate_launch(config)
                result[shape][m] = config
    return {s: {m: result[s][m] for m in expected} for s in shapes}


def config_filename(N, K, block_n, block_k, device_id=0):
    # Same helper and format as vLLM get_w8a8_block_fp8_configs; no local fallback.
    from vllm.utils.platform_utils import get_device_name_as_file_name
    name = get_device_name_as_file_name(device_id)
    return f'N={N},K={K},device_name={name},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json'


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                         suffix='.tmp', delete=False) as stream:
            temp = stream.name
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        temp = None
    finally:
        if temp is not None:
            os.unlink(temp)


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_configs(path, block_k):
    with open(path) as stream:
        data = json.load(stream, object_pairs_hook=_json_pairs)
    if not isinstance(data, dict) or not data:
        raise ValueError('Existing config must be a nonempty JSON object')
    result = {}
    for key, value in data.items():
        if not key.isdecimal() or str(int(key)) != key or int(key) <= 0:
            raise ValueError(f'Invalid M key: {key!r}')
        result[int(key)] = validate_launch(value, block_k)
    return result


def save_configs(path, configs, block_k, overwrite=False):
    """Merge disjoint M; overlapping M require explicit --overwrite. Lock read+write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not configs:
        raise ValueError('Cannot save empty configs')
    for m, cfg in configs.items():
        positive(m, 'M')
        validate_launch(cfg, block_k)
    with open(str(path) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read_configs(path, block_k) if path.exists() else {}
        overlap = previous.keys() & configs.keys()
        if overlap and not overwrite:
            raise FileExistsError(f'{path}: existing M {sorted(overlap)}; use --overwrite to replace these M only')
        previous.update(configs)
        atomic_json(path, {str(m): previous[m] for m in sorted(previous)})
    return path


def timing_stats(elapsed_ms, calls_per_event=1):
    positive(calls_per_event, 'calls_per_event')
    if not elapsed_ms or any(not math.isfinite(v) or v <= 0 for v in elapsed_ms):
        raise ValueError('Timing samples must be finite positive milliseconds')
    samples = [v * 1000 / calls_per_event for v in elapsed_ms]
    return {'median_us': statistics.median(samples), 'mean_us': statistics.mean(samples),
            'min_us': min(samples), 'max_us': max(samples), 'samples_us': samples}
