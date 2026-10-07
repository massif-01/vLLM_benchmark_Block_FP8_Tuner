import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fp8_tuning as core

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128,
           GROUP_SIZE_M=32, num_warps=4, num_stages=2)


def model(**overrides):
    cfg = dict(architectures=['Qwen3ForCausalLM'], hidden_size=2048,
               num_attention_heads=32, num_key_value_heads=4,
               head_dim=128, intermediate_size=6144)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


@pytest.mark.parametrize('tp,qkv,k', [(1,5120,4096),(2,2560,2048),(4,1280,1024),(8,768,512)])
def test_gqa_and_fused_mlp(tp,qkv,k):
    shapes,_ = core.model_shapes(model(), tp)
    assert shapes == [(qkv,2048),(2048,k),(12288//tp,2048),(2048,6144//tp)]


@pytest.mark.parametrize('override,tp', [({},3), ({},64), ({'num_key_value_heads':3},2),
    ({'intermediate_size':6145},2), ({'head_dim':0},1), ({'hidden_size':0},1)])
def test_illegal_tp_and_dimensions(override,tp):
    with pytest.raises(ValueError): core.model_shapes(model(**override),tp)


@pytest.mark.parametrize('arch', ['Unknown','Qwen3NextForCausalLM','Qwen3VLForConditionalGeneration','DeepseekV3ForCausalLM'])
def test_unsupported_architecture(arch):
    with pytest.raises(ValueError, match='does not support'):
        core.model_shapes(model(architectures=[arch]),1)


def test_nested_config():
    with pytest.raises(ValueError,match='Nested'):
        core.model_shapes(model(text_config={}),1)


def test_model_load_failure_and_remote_default():
    def loader(**kwargs):
        assert kwargs['trust_remote_code'] is False
        raise OSError('offline')
    with pytest.raises(RuntimeError,match='Cannot load model config'):
        core.load_model_shapes('name',1,loader=loader)


def test_moe_excludes_routed_experts_and_absent_dense_mlp():
    cfg=model(architectures=['Qwen3MoeForCausalLM'],num_hidden_layers=8,
              num_experts=32,decoder_sparse_step=1,mlp_only_layers=[],
              moe_intermediate_size=99999)
    shapes,sources=core.model_shapes(cfg,8)
    assert shapes == [(768,2048),(2048,512)]
    assert all('experts' not in x['layer'] for x in sources)
    cfg.mlp_only_layers=[0]
    cfg.shared_expert_intermediate_size=1024
    shapes,_=core.model_shapes(cfg,8)
    assert (1536,2048) in shapes and (256,2048) in shapes
    assert (2048,128) in shapes


def test_duplicates():
    assert core.unique_shapes([(128,256),(128,256),(256,128)]) == [(128,256),(256,128)]


@pytest.mark.parametrize('sizes,gpus', [([1],8),([1,2,4],2),([1,2],8)])
def test_distribution(sizes,gpus):
    assignments = core.distribute_batch_sizes(sizes,gpus)
    assert len(assignments) == min(gpus,len(sizes)) and all(assignments)
    assert sorted(m for batch in assignments for m in batch) == sorted(sizes)
    assert assignments == core.distribute_batch_sizes(sizes,gpus)


@pytest.mark.parametrize('sizes,gpus', [([],1),([1,1],2),([0],1),([1],0)])
def test_bad_distribution(sizes,gpus):
    with pytest.raises(ValueError): core.distribute_batch_sizes(sizes,gpus)


def test_worker_merge():
    shapes=[(128,256),(256,128)]
    workers=[{s:{1:CFG} for s in shapes},{s:{2:CFG,4:CFG} for s in shapes}]
    merged=core.merge_results(workers,[[1],[2,4]],shapes)
    assert all(list(cfg) == [1,2,4] for cfg in merged.values())
    with pytest.raises(RuntimeError): core.merge_results(workers[:1],[[1],[2,4]],shapes)
    workers[1][shapes[0]].pop(4)
    with pytest.raises(RuntimeError): core.merge_results(workers,[[1],[2,4]],shapes)


@pytest.mark.parametrize('partials,assigned', [([{}],[[1]]),([None],[[1]]),([{(128,256):{}}],[[]]),
    ([{(128,256):{1:CFG}},{(128,256):{1:CFG}}],[[1],[1]]),
    ([{(128,256):{2:CFG}}],[[1]])])
def test_bad_worker_results(partials,assigned):
    with pytest.raises(RuntimeError): core.merge_results(partials,assigned,[(128,256)])


def test_existing_merge_and_overwrite(tmp_path):
    path=tmp_path/'config.json'
    core.save_configs(path,{1:CFG,2:CFG},128)
    core.save_configs(path,{64:CFG},128)
    assert set(core.read_configs(path,128)) == {1,2,64}
    original=path.read_bytes()
    with pytest.raises(FileExistsError): core.save_configs(path,{64:CFG},128)
    assert path.read_bytes() == original
    newer=dict(CFG,BLOCK_SIZE_M=128)
    core.save_configs(path,{64:newer},128,overwrite=True)
    assert core.read_configs(path,128) == {1:CFG,2:CFG,64:newer}


@pytest.mark.parametrize('fault', ['replace','fsync','dump'])
def test_atomic_write_failure(tmp_path,fault):
    path=tmp_path/'config.json'
    core.save_configs(path,{1:CFG},128)
    original=path.read_bytes()
    with patch('fp8_tuning.'+ ('json.dump' if fault=='dump' else 'os.'+fault),side_effect=OSError('interrupted')):
        with pytest.raises(OSError): core.save_configs(path,{2:CFG},128)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('content', ['{', '{"1":{},"1":{}}', '{}', '{"01":{}}'])
def test_invalid_existing_config(tmp_path,content):
    path=tmp_path/'config.json'; path.write_text(content)
    with pytest.raises(ValueError): core.save_configs(path,{2:CFG},128)
    assert path.read_text() == content


def test_timing_arithmetic():
    stats=core.timing_stats([1.,2.,3.])
    assert stats['mean_us'] == 2000 and stats['median_us'] == 2000
    assert core.timing_stats([1.,2.,3.],10)['median_us'] == 200


def test_device_helper_delegation():
    helper=lambda device_id: f'NVIDIA_GPU_{device_id}'
    with patch.dict(sys.modules, {'vllm.utils.platform_utils':SimpleNamespace(get_device_name_as_file_name=helper)}):
        assert core.config_filename(128,256,128,128,2) == 'N=128,K=256,device_name=NVIDIA_GPU_2,dtype=fp8_w8a8,block_shape=[128,128].json'


def test_help_and_explicit_preview_without_gpu():
    result=subprocess.run([sys.executable,str(ROOT/'benchmark_w8a8_block_fp8.py'),'--help'],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    result=subprocess.run([sys.executable,str(ROOT/'benchmark_w8a8_block_fp8.py'),
                           '--shape','128','256','--shape','128','256','--batch-size','3','--out-dtype','float16','--preview'],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    preview=json.loads(result.stdout)
    assert preview['shapes'] == [[128,256]] and preview['M'] == [3]


def test_default_batch_load_balance_and_completeness():
    sizes = core.DEFAULT_BATCH_SIZES
    assignments = core.distribute_batch_sizes(sizes,8)
    assert len(assignments) == 8 and all(assignments)
    assert sorted(m for batch in assignments for m in batch) == sorted(sizes)
    assert all(batch == sorted(batch) for batch in assignments)
    assert assignments == core.distribute_batch_sizes(sizes,8)
    old = [sizes[i*len(sizes)//8:(i+1)*len(sizes)//8] for i in range(8)]
    assert max(map(sum,assignments)) < max(map(sum,old))/2


def test_balanced_assignments_preserve_parent_merge():
    sizes = core.DEFAULT_BATCH_SIZES
    assignments = core.distribute_batch_sizes(sizes,8)
    shapes = [(128,256),(256,128)]
    partials = [{shape:{m:CFG for m in batch} for shape in shapes} for batch in assignments]
    merged = core.merge_results(partials,assignments,shapes)
    assert all(set(configs) == set(sizes) for configs in merged.values())
