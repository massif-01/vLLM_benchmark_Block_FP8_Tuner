"""Scope exclusions to the maintained regular-linear targets, including HF aliases."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_w8a8_block_fp8 as bench
import fp8_tuning as core


def quantized_model(**exclusion_fields):
    return dict(architectures=['Qwen3MoeForCausalLM'], hidden_size=2048, head_dim=128,
                num_attention_heads=32, num_key_value_heads=4, intermediate_size=6144,
                num_hidden_layers=48, num_experts=128, decoder_sparse_step=1, mlp_only_layers=[],
                quantization_config=dict(quant_method='fp8', activation_scheme='dynamic',
                                         weight_block_size=[128,128], **exclusion_fields))


def plan(config, monkeypatch):
    def loader(model, tp_size, trust_remote_code):
        shapes, sources = core.model_shapes(config, tp_size)
        return config, shapes, sources
    monkeypatch.setattr(bench, 'load_model_shapes', loader)
    return bench.plan(bench.build_parser().parse_args(['--model', 'official-style', '--tp-size', '4']))


def test_official_coder_style_allows_auto_planning(monkeypatch):
    config = quantized_model(modules_to_not_convert=[
        'lm_head', 'model.layers.0.input_layernorm', 'model.layers.0.post_attention_layernorm',
        'model.layers.0.mlp.gate', 'model.layers.0.mlp.shared_expert_gate'])
    result = plan(config, monkeypatch)
    assert result['shapes'] == [(1280,2048),(2048,1024)]
    assert all(item['layer'].startswith('self_attn.') for item in result['sources'])


@pytest.mark.parametrize('pattern', [
    'lm_head', 'model.embed_tokens.weight', '*.input_layernorm', '*.post_attention_layernorm',
    'model.layers.*.self_attn.q_norm', 'model.layers.0.self_attn.k_norm.weight',
    '*.mlp.gate', '*.mlp.shared_expert_gate', 'model.layers.0.mlp.gate.weight',
    '*.bias', 'model.layers.0.self_attn.o_proj.bias', '*.input_layernorm.*',
    '*.mlp.experts.*', 'model.layers.0.mlp.experts.0.gate_proj.weight',
])
def test_explicit_non_targets_allowed(pattern):
    config = quantized_model(modules_to_not_convert=[pattern])
    assert 'matches' in core.validate_model_quantization(config,128,128)


@pytest.mark.parametrize('pattern', [
    'model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.k_proj.weight',
    'model.layers.0.self_attn.v_proj', '*.self_attn.qkv_proj', 'model.layers.0.self_attn.o_proj',
    'model.layers.0.mlp.gate_proj', 'model.layers.0.mlp.up_proj', '*.mlp.gate_up_proj',
    'model.layers.0.mlp.down_proj', 'model.layers.0.mlp.shared_expert.gate_proj',
    'model.layers.0.mlp.shared_expert.up_proj', 'model.layers.0.mlp.shared_expert.gate_up_proj',
    'model.layers.0.mlp.shared_expert.down_proj',
    'model.layers', 'model.layers.0', 'model.layers.0.self_attn', 'model.layers.0.mlp',
    '*', '*gate*', 're:.*gate', 'model.layers.*.mlp.*',
])
def test_targets_and_ambiguous_exclusions_fail_closed(pattern):
    config = quantized_model(modules_to_not_convert=[pattern])
    with pytest.raises(ValueError) as error:
        core.validate_model_quantization(config,128,128)
    assert pattern in str(error.value) and '--shape N K' in str(error.value)


@pytest.mark.parametrize('unsafe_field', ['ignored_layers','modules_to_not_convert'])
def test_both_exclusion_fields_checked(unsafe_field):
    fields = dict(ignored_layers=['lm_head'], modules_to_not_convert=['*.input_layernorm'])
    unsafe = 'model.layers.0.mlp.gate_proj'
    fields[unsafe_field].append(unsafe)
    with pytest.raises(ValueError) as error:
        core.validate_model_quantization(quantized_model(**fields),128,128)
    assert unsafe in str(error.value)


def test_both_safe_fields_and_duplicates_allowed():
    config = quantized_model(ignored_layers=['lm_head','*.mlp.gate'],
                             modules_to_not_convert=['lm_head','*.input_layernorm'])
    assert 'matches' in core.validate_model_quantization(config,128,128)


@pytest.mark.parametrize('invalid', ['lm_head',[None],[''],42])
def test_invalid_exclusion_contract_rejected(invalid):
    with pytest.raises(ValueError):
        core.validate_model_quantization(quantized_model(ignored_layers=invalid),128,128)


def test_official_config_snapshot_passes_plan(monkeypatch):
    # HF config-only snapshot at dcaee4d4dfc5ee71ad501f01f530e5652438fde0.
    # This exercises the real planner with JSON input, not vLLM CLI/GPU integration.
    config = json.loads((Path(__file__).parent/'fixtures/qwen3_coder_fp8_config.json').read_text())
    assert len(config['quantization_config']['modules_to_not_convert']) == 145
    result = plan(config, monkeypatch)
    assert result['shapes'] == [(1280,2048),(2048,1024)]
    assert len(result['sources']) == 2
