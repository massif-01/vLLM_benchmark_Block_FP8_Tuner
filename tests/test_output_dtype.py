"""Model auto delegates to vLLM; CPU/source-contract tests do not run CUDA."""
import importlib.util
import json
import logging
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import benchmark_w8a8_block_fp8 as bench
import fp8_tuning as core


@pytest.fixture
def installed_resolver(monkeypatch):
    spec=importlib.util.spec_from_file_location('dtype_contract',ROOT/'tests/fixtures/vllm_dtype_contract.py')
    contract=importlib.util.module_from_spec(spec)
    class Dtype: pass
    torch=SimpleNamespace(float16=Dtype(),bfloat16=Dtype(),float32=Dtype(),dtype=Dtype)
    contract.torch=torch
    spec.loader.exec_module(contract)
    contract.logger=logging.getLogger('dtype_contract')
    # CUDA dtype policy, without any CUDA API or engine. BF16 remains BF16;
    # the upstream resolver (not the tuner) uses the platform's preferred dtype.
    contract.current_platform=SimpleNamespace(supported_dtypes=[torch.bfloat16,torch.float16,torch.float32],is_cpu=lambda:False)
    contract.get_safetensors_params_metadata=lambda *a,**kw:{}
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'vllm.config.model',contract)
    return contract,torch


def hf_config(raw, dtype):
    # Simulate HF loading normalizing its JSON dtype into a torch.dtype value.
    config=SimpleNamespace(**raw)
    config.dtype=dtype
    config.get_text_config=lambda:config
    return config


def full_plan(monkeypatch, config, *extra):
    def loader(model,tp_size,trust_remote_code):
        shapes,sources=core.model_shapes(config,tp_size)
        return config,shapes,sources
    monkeypatch.setattr(bench,'load_model_shapes',loader)
    args=bench.build_parser().parse_args(['--model','test-model','--tp-size','4',*extra])
    return args,bench.plan(args)


def test_official_coder_auto_uses_runtime_bf16(monkeypatch,installed_resolver):
    contract,torch=installed_resolver
    raw=json.loads((ROOT/'tests/fixtures/qwen3_coder_fp8_config.json').read_text())
    assert raw['torch_dtype']=='bfloat16'
    config=hf_config(raw,torch.bfloat16)
    calls=[]
    resolver=contract._get_and_verify_dtype
    def recorded(model_id,config,dtype,*,is_pooling_model):
        calls.append((model_id,config,dtype,is_pooling_model))
        return resolver(model_id,config,dtype,is_pooling_model=is_pooling_model)
    monkeypatch.setattr(contract,'_get_and_verify_dtype',recorded)
    args,planned=full_plan(monkeypatch,config)
    assert planned['shapes']==[(1280,2048),(2048,1024)]
    assert planned['requested_out_dtype']=='auto' and planned['out_dtype']=='bfloat16'
    assert planned['resolved_out_dtype']=='bfloat16' and planned['out_dtype_source']=='vllm-model-auto'
    assert args.out_dtype=='auto' and args.resolved_out_dtype=='bfloat16'
    assert calls==[('test-model',config,'auto',False)]


@pytest.mark.parametrize('requested,resolved',[('float16','float16'),('half','float16'),('float32','float32')])
def test_official_coder_explicit_override_bypasses_resolver(monkeypatch,installed_resolver,requested,resolved):
    contract,torch=installed_resolver
    raw=json.loads((ROOT/'tests/fixtures/qwen3_coder_fp8_config.json').read_text())
    def forbidden(*a,**kw): raise AssertionError('Explicit dtype must not call auto resolver')
    monkeypatch.setattr(contract,'_get_and_verify_dtype',forbidden)
    args,planned=full_plan(monkeypatch,hf_config(raw,torch.bfloat16),'--out-dtype',requested)
    assert args.out_dtype==requested and args.resolved_out_dtype==resolved
    assert planned['requested_out_dtype']==requested and planned['out_dtype']==resolved
    assert planned['out_dtype_source']=='explicit'


@pytest.mark.parametrize('config_dtype,resolved',[('float16','float16'),('bfloat16','bfloat16'),('float32','bfloat16')])
def test_dense_model_auto_uses_vllm_resolution(monkeypatch,installed_resolver,config_dtype,resolved):
    contract,torch=installed_resolver
    raw=dict(architectures=['Qwen3ForCausalLM'],model_type='qwen3',hidden_size=2048,
             num_attention_heads=32,num_key_value_heads=4,head_dim=128,intermediate_size=6144)
    _,planned=full_plan(monkeypatch,hf_config(raw,getattr(torch,config_dtype)))
    assert planned['out_dtype']==resolved and planned['out_dtype_source']=='vllm-model-auto'


def test_shape_auto_fails_with_explicit_guidance():
    args=bench.build_parser().parse_args(['--shape','128','256'])
    assert args.out_dtype=='auto'
    with pytest.raises(ValueError,match='--out-dtype'): bench.plan(args)
    result=subprocess.run([sys.executable,str(ROOT/'benchmark_w8a8_block_fp8.py'),
                           '--shape','128','256','--preview'],capture_output=True,text=True)
    assert result.returncode!=0 and '--out-dtype' in result.stderr


@pytest.mark.parametrize('dtype',['float16','bfloat16','float32','half'])
def test_shape_explicit_preview_without_gpu(dtype):
    result=subprocess.run([sys.executable,str(ROOT/'benchmark_w8a8_block_fp8.py'),
                           '--shape','128','256','--out-dtype',dtype,'--preview'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    planned=json.loads(result.stdout)
    resolved='float16' if dtype=='half' else dtype
    assert planned['requested_out_dtype']==dtype and planned['out_dtype']==resolved
    assert planned['resolved_out_dtype']==resolved and planned['out_dtype_source']=='explicit'


@pytest.mark.parametrize('fault',['signature','raises','unsupported','missing'])
def test_auto_resolver_failure_is_closed(monkeypatch,installed_resolver,fault):
    contract,torch=installed_resolver
    config=hf_config(json.loads((ROOT/'tests/fixtures/qwen3_coder_fp8_config.json').read_text()),torch.bfloat16)
    if fault=='signature':
        monkeypatch.setattr(contract,'_get_and_verify_dtype',lambda config,dtype:torch.float16)
    elif fault=='raises':
        def fail(*a,**kw): raise RuntimeError('resolver broken')
        monkeypatch.setattr(contract,'_get_and_verify_dtype',fail)
    elif fault=='unsupported':
        monkeypatch.setattr(contract,'_get_and_verify_dtype',lambda **kw:object())
    else:
        monkeypatch.delattr(contract,'_get_and_verify_dtype')
    with pytest.raises(RuntimeError) as error: full_plan(monkeypatch,config)
    assert 'Installed vLLM dtype resolver' in str(error.value) and '--out-dtype' in str(error.value)


def test_auto_dtype_reaches_worker_and_saved_report(monkeypatch,installed_resolver,tmp_path):
    contract,torch=installed_resolver
    raw=json.loads((ROOT/'tests/fixtures/qwen3_coder_fp8_config.json').read_text())
    config=hf_config(raw,torch.bfloat16)
    def load(model,tp,trust):
        shapes,sources=core.model_shapes(config,tp)
        return config,shapes,sources
    monkeypatch.setattr(bench,'load_model_shapes',load)
    args=bench.build_parser().parse_args(['--model','coder','--tp-size','4','--batch-size','17','--save-path',str(tmp_path)])
    torch.cuda=SimpleNamespace(device_count=lambda:1,get_device_name=lambda i:'testGPU',set_device=lambda i:None)
    monkeypatch.setattr(bench,'torch',torch)
    monkeypatch.setattr(bench,'load_runtime',lambda:{'test':'CPU dtype flow only'})
    monkeypatch.setattr(bench,'check_devices',lambda ids:None)
    monkeypatch.setattr(bench,'config_filename',lambda n,k,*rest:f'{n}-{k}.json')
    from test_fp8_tuning import CFG
    def worker(task):
        assert task['args'].out_dtype=='auto' and task['args'].resolved_out_dtype=='bfloat16'
        return dict(configs={s:{17:CFG} for s in task['weight_shapes']},measurements=[],identity={},gpu_id=0)
    monkeypatch.setattr(bench,'tune_on_gpu',worker)
    bench.main(args)
    report=json.loads(next((tmp_path/'reports').glob('*.json')).read_text())
    assert report['arguments']['out_dtype']=='auto'
    assert report['arguments']['resolved_out_dtype']=='bfloat16'
    assert report['plan']['requested_out_dtype']=='auto'
    assert report['plan']['out_dtype']=='bfloat16' and report['plan']['out_dtype_source']=='vllm-model-auto'


class Tensor:
    def __sub__(self,x): return self
    def __add__(self,x): return self
    def __mul__(self,x): return self
    def to(self,dtype): return self


def test_tune_reference_and_correctness_only_use_resolved_dtype(monkeypatch,installed_resolver):
    contract,torch=installed_resolver
    config=hf_config(json.loads((ROOT/'tests/fixtures/qwen3_coder_fp8_config.json').read_text()),torch.bfloat16)
    args,planned=full_plan(monkeypatch,config,'--batch-size','17')
    torch.manual_seed=lambda seed:None
    torch.rand=lambda *a,**kw:Tensor()
    torch.float8_e4m3fn=object()
    torch.cuda=SimpleNamespace(current_device=lambda:0)
    monkeypatch.setattr(bench,'torch',torch)
    monkeypatch.setattr(bench,'triton',SimpleNamespace(cdiv=lambda x,y:(x+y-1)//y,
        runtime=SimpleNamespace(autotuner=SimpleNamespace(OutOfResources=RuntimeError))))
    seen=[]
    def measure(*values):
        seen.append(('measure',values[6]))
        return {'median_us':1.0}
    def reference(*values):
        seen.append(('reference',values[-1]))
        return Tensor()
    def check(*values):
        seen.append(('correctness',values[6]))
        return {'max_abs_error':0.0}
    monkeypatch.setattr(bench,'benchmark_config',measure)
    monkeypatch.setattr(bench,'reference_matmul',reference)
    monkeypatch.setattr(bench,'check_correctness',check)
    bench.tune(17,1280,2048,args,[bench.default_config(128,128)])
    assert {kind for kind,dtype in seen}=={'measure','reference','correctness'}
    assert all(dtype is torch.bfloat16 for kind,dtype in seen)
    assert args.out_dtype=='auto'


def test_verify_installed_uses_plan_resolved_dtype(monkeypatch,installed_resolver,tmp_path):
    contract,torch=installed_resolver
    config=hf_config(json.loads((ROOT/'tests/fixtures/qwen3_coder_fp8_config.json').read_text()),torch.bfloat16)
    args,planned=full_plan(monkeypatch,config,'--batch-size','17','--save-path',str(tmp_path))
    torch.manual_seed=lambda seed:None
    torch.rand=lambda *a,**kw:Tensor()
    torch.float8_e4m3fn=object()
    torch.cuda=SimpleNamespace(set_device=lambda i:None)
    torch.isfinite=lambda value:SimpleNamespace(all=lambda:True)
    torch.testing=SimpleNamespace(assert_close=lambda *a,**kw:None)
    monkeypatch.setattr(bench,'torch',torch)
    monkeypatch.setattr(bench,'triton',SimpleNamespace(cdiv=lambda x,y:(x+y-1)//y))
    monkeypatch.setattr(bench,'check_devices',lambda ids:None)
    monkeypatch.setattr(bench,'config_filename',lambda *a:'config.json')
    from test_fp8_tuning import CFG
    monkeypatch.setattr(bench,'read_configs',lambda *a:{17:CFG})
    def loader(*values): return {17:CFG}
    loader.cache_clear=lambda:None
    seen=[]
    def runtime(*values):
        seen.append(values[-1])
        return Tensor()
    monkeypatch.setitem(sys.modules,'vllm.model_executor.layers.quantization.utils.fp8_utils',
        SimpleNamespace(get_w8a8_block_fp8_configs=loader,w8a8_triton_block_scaled_mm=runtime))
    monkeypatch.setattr(bench,'reference_matmul',runtime)
    bench.verify_installed(args,planned)
    assert len(seen)==4 and all(dtype is torch.bfloat16 for dtype in seen)
    assert args.out_dtype=='auto'


def test_fp32_auto_follows_resolver_preference_not_tuner_rule(monkeypatch,installed_resolver):
    contract,torch=installed_resolver
    contract.current_platform.supported_dtypes=[torch.float16,torch.bfloat16,torch.float32]
    raw=dict(architectures=['Qwen3ForCausalLM'],model_type='qwen3',hidden_size=2048,
             num_attention_heads=32,num_key_value_heads=4,head_dim=128,intermediate_size=6144)
    _,planned=full_plan(monkeypatch,hf_config(raw,torch.float32))
    assert planned['out_dtype']=='float16' and planned['out_dtype_source']=='vllm-model-auto'
