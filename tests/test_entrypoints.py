import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import benchmark_w8a8_block_fp8 as bench
import fp8_tuning as core
from test_fp8_tuning import CFG, model


def test_source_audited_device_name_and_loader(tmp_path):
    fixture = ROOT/'tests/fixtures/vllm_fp8_contract.py'
    spec=importlib.util.spec_from_file_location('official_contract',fixture)
    official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    official.__file__ = str(tmp_path/'fp8_utils.py')
    official.logger = logging.getLogger('fixture')
    platform = SimpleNamespace(get_device_name=lambda i: 'NVIDIA  H100 / PCIe\tGPU')
    with patch.dict(sys.modules,{'vllm.platforms':SimpleNamespace(current_platform=platform)}):
        assert official.get_device_name_as_file_name(0) == 'NVIDIA_H100_PCIe_GPU'
        with patch.dict(sys.modules,{'vllm.utils.platform_utils':official}):
            path=tmp_path/'configs'/core.config_filename(128,256,128,128)
            core.save_configs(path,{1:CFG,64:CFG},128)
            assert official.get_w8a8_block_fp8_configs(128,256,128,128) == {1:CFG,64:CFG}
            assert official.get_w8a8_block_fp8_configs(256,256,128,128) is None


@pytest.mark.parametrize('flags', [[],['--shape','0','128'],['--shape','128','256','--out-dtype','float16','--tp-size','0'],
 ['--shape','128','256','--out-dtype','float16','--block-k','48'],['--shape','128','96'],
 ['--shape','128','256','--out-dtype','float16','--batch-size','0'],['--shape','128','256','--out-dtype','float16','--measurements','0'],
 ['--shape','128','256','--out-dtype','float16','--calls-per-event','0'],['--shape','128','256','--out-dtype','float16','--input-type','int8'],
 ['--shape','128','256','--out-dtype','float16','--seed','-1'],['--preview','--check-environment']])
def test_cli_errors(flags):
    p=subprocess.run([sys.executable,str(ROOT/'benchmark_w8a8_block_fp8.py'),'--out-dtype','float16',*flags,'--preview'],capture_output=True,text=True)
    assert p.returncode != 0
    assert 'completed' not in p.stdout.lower()


def test_preview_effective_parameters():
    args=bench.build_parser().parse_args(['--shape','129','256','--shape','129','256',
      '--tp-size','8','--out-dtype','half','--block-n','64','--block-k','64',
      '--batch-size','17','--seed','3','--measurements','7','--calls-per-event','20',
      '--overwrite','--trust-remote-code','--save-path','where','--preview'])
    plan=bench.plan(args)
    assert plan['shapes']==[(129,256)]  # Explicit shapes are already local, no TP division.
    assert plan['M']==[17] and plan['block_shape']==[64,64]
    assert plan['out_dtype']=='float16' and args.trust_remote_code and args.overwrite


def test_checkpoint_quantization_and_ignored_layers():
    quant={'quant_method':'fp8','activation_scheme':'dynamic','weight_block_size':[128,128]}
    cfg=model(quantization_config=quant)
    assert 'matches' in core.validate_model_quantization(cfg,128,128)
    with pytest.raises(ValueError,match='square'):
        core.validate_model_quantization(cfg,64,128)
    for invalid in [dict(quant,quant_method='int8'),dict(quant,activation_scheme='static'),
                    dict(quant,weight_block_size=[64,128]),dict(quant,ignored_layers=['model.layers.0'])]:
        cfg.quantization_config=invalid
        with pytest.raises(ValueError): core.validate_model_quantization(cfg,128,128)


def test_model_fused_partition_alignment():
    args=bench.build_parser().parse_args(['--model','x','--tp-size','8'])
    cfg=model(head_dim=80)
    shapes,sources=core.model_shapes(cfg,8)
    with pytest.raises(ValueError): bench.validate_plan_layout(shapes,sources,args,cfg)
    cfg=model(intermediate_size=6176)
    shapes,sources=core.model_shapes(cfg,8)
    with pytest.raises(ValueError): bench.validate_plan_layout(shapes,sources,args,cfg)


def test_search_space_and_default():
    for block_k in [32,64,128,256]:
        configs=bench.get_configs_compute_bound(block_k)
        assert configs
        for c in configs: core.validate_launch(c,block_k)
        core.validate_launch(bench.default_config(128,block_k),block_k)


def test_event_count_matches_conversion(monkeypatch):
    calls=[]
    class Event:
        def __init__(self,**kwargs): pass
        def record(self): pass
        def synchronize(self): pass
        def elapsed_time(self,end): return 2.0
    fake_torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None,Event=Event))
    A=SimpleNamespace(shape=(1,256),new_empty=lambda *a,**kw:'buffer')
    B=SimpleNamespace(shape=(128,256))
    monkeypatch.setattr(bench,'torch',fake_torch)
    monkeypatch.setattr(bench,'w8a8_block_matmul',lambda *a,**kw:calls.append(kw['output']))
    stats=bench.benchmark_config(A,B,None,None,[128,128],CFG,'dtype',num_iters=3,calls_per_event=4)
    assert len(calls)==5+3*4 and set(calls)=={'buffer'}
    assert stats['median_us']==500.0


@pytest.fixture
def fake_python(tmp_path):
    stub=tmp_path/'python-stub'
    stub.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$CAPTURE"
exit "${STUB_STATUS:-0}"
''')
    stub.chmod(0o755)
    env=dict(os.environ,PYTHON=str(stub),CAPTURE=str(tmp_path/'args'),STUB_STATUS='7',TRUST_REMOTE_CODE='0')
    env.pop('BATCH_SIZE',None)
    env.pop('OUT_DTYPE',None)
    return env


@pytest.mark.parametrize('wrapper,args', [('tune_custom.sh',['test', '2','128','128']),
    ('tune_qwen3.sh',[]),('tune_qwen3_coder.sh',[]),('environment_check.sh',[])])
def test_shell_failure_propagation(wrapper,args,fake_python):
    p=subprocess.run(['bash',str(ROOT/'scripts'/wrapper),*args],env=fake_python,capture_output=True,text=True)
    assert p.returncode==7 and 'completed' not in p.stdout.lower()
    captured=Path(fake_python['CAPTURE']).read_text()
    assert '--trust-remote-code' not in captured and '--out-dtype' not in captured


def test_shell_explicit_trust_and_success(fake_python):
    env=dict(fake_python,STUB_STATUS='0',TRUST_REMOTE_CODE='1',BATCH_SIZE='17')
    p=subprocess.run(['bash',str(ROOT/'scripts/tune_custom.sh'),'test','2','128','128','--overwrite'],env=env,capture_output=True,text=True)
    assert p.returncode==0
    args=Path(env['CAPTURE']).read_text().splitlines()
    assert '--trust-remote-code' in args and '--overwrite' in args
    assert args[args.index('--batch-size')+1]=='17'
    assert args[args.index('--save-path')+1]==str(ROOT/'tuned_configs')


def test_shell_invalid_trust(fake_python):
    env=dict(fake_python,TRUST_REMOTE_CODE='yes')
    p=subprocess.run(['bash',str(ROOT/'scripts/tune_qwen3.sh')],env=env,capture_output=True,text=True)
    assert p.returncode==2
    assert not Path(env['CAPTURE']).exists()


def test_batch_aggregates_all_failures(fake_python):
    p=subprocess.run(['bash',str(ROOT/'examples/tune_qwen3_models.sh')],env=fake_python,capture_output=True,text=True)
    assert p.returncode==1 and '8 task(s) failed' in p.stderr
    assert Path(fake_python['CAPTURE']).read_text().count('--model')==8
    assert 'completed' not in p.stdout.lower()


def test_batch_success(fake_python):
    env=dict(fake_python,STUB_STATUS='0')
    p=subprocess.run(['bash',str(ROOT/'examples/tune_qwen3_models.sh')],env=env,capture_output=True,text=True)
    assert p.returncode==0 and 'completed successfully' in p.stdout


def test_deprecated_entries():
    p=subprocess.run(['bash',str(ROOT/'scripts/tune_deepseek_v3.sh')],capture_output=True,text=True)
    assert p.returncode==2 and 'Deprecated' in p.stderr
    p=subprocess.run([sys.executable,str(ROOT/'benchmark_w8a8_block_fp8_qwencoder.py'),
                      '--shape','128','256','--out-dtype','float16','--preview'],capture_output=True,text=True)
    assert p.returncode==0 and 'Deprecated' in p.stderr
    assert json.loads(p.stdout)['shapes']==[[128,256]]


def test_main_worker_failure_never_saves(monkeypatch,tmp_path,capsys):
    args=bench.build_parser().parse_args(['--shape','128','256','--out-dtype','float16','--batch-size','1','--save-path',str(tmp_path)])
    cuda=SimpleNamespace(device_count=lambda:1,get_device_name=lambda i:'H100',set_device=lambda i:None)
    monkeypatch.setattr(bench,'load_runtime',lambda:{})
    monkeypatch.setattr(bench,'torch',SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(bench,'check_devices',lambda ids:None)
    monkeypatch.setattr(bench,'config_filename',lambda *a:'config.json')
    def fail(task): raise RuntimeError('worker failed')
    monkeypatch.setattr(bench,'tune_on_gpu',fail)
    with pytest.raises(RuntimeError,match='worker failed'): bench.main(args)
    assert not list(tmp_path.glob('*.json')) and 'completed' not in capsys.readouterr().out.lower()


def test_main_heterogeneous_gpus_rejected(monkeypatch):
    args=bench.build_parser().parse_args(['--shape','128','256','--out-dtype','float16'])
    monkeypatch.setattr(bench,'load_runtime',lambda:{})
    monkeypatch.setattr(bench,'check_devices',lambda ids:None)
    monkeypatch.setattr(bench,'torch',SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2,get_device_name=lambda i:f'GPU{i}')))
    with pytest.raises(RuntimeError,match='identical'): bench.main(args)


def test_default_and_runtime_identity_contract(monkeypatch):
    spec=importlib.util.spec_from_file_location('official_contract',ROOT/'tests/fixtures/vllm_fp8_contract.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    for bn,bk in [(32,32),(128,128),(256,128)]:
        assert bench.default_config(bn,bk)==official.default_config([bn,bk])
    # Exercise the actual import/signature/identity path without CUDA. Cached loader
    # wrappers lack __code__; the underlying __wrapped__ function has the source path.
    def dummy(): pass
    from functools import lru_cache
    loader=lru_cache()(dummy)
    kernel=SimpleNamespace(arg_names=official.KERNEL_ARG_NAMES,src='audited kernel',fn=dummy)
    platform=SimpleNamespace(is_cuda=lambda:True)
    torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True),__version__='torch-test',version=SimpleNamespace(cuda='cuda-test'))
    triton=SimpleNamespace(__version__='triton-test')
    mods={'torch':torch,'vllm':SimpleNamespace(__version__='vllm-test'),
          'vllm.platforms':SimpleNamespace(current_platform=platform),
          'vllm.triton_utils':SimpleNamespace(triton=triton),
          'vllm.model_executor.layers.quantization.utils.fp8_utils':SimpleNamespace(
              _w8a8_triton_block_scaled_mm=kernel,get_w8a8_block_fp8_configs=loader,w8a8_triton_block_scaled_mm=dummy),
          'vllm.utils.platform_utils':SimpleNamespace(get_device_name_as_file_name=lambda *a:'GPU'),
          'vllm.transformers_utils.config':SimpleNamespace(get_config=dummy)}
    with patch.dict(sys.modules,mods):
        identity=bench.load_runtime()
        assert identity['vllm']=='vllm-test' and len(identity['kernel_sha256'])==64
        assert identity['loader_file']==str(Path(__file__).resolve())
        kernel.arg_names=kernel.arg_names+['changed']
        with pytest.raises(RuntimeError,match='signature changed'): bench.load_runtime()


def test_main_saves_every_m_and_report(monkeypatch,tmp_path,capsys):
    args=bench.build_parser().parse_args(['--shape','128','256','--out-dtype','float16','--batch-size','17','--save-path',str(tmp_path)])
    monkeypatch.setattr(bench,'load_runtime',lambda:{'vllm':'mock'})
    monkeypatch.setattr(bench,'torch',SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:8,
        get_device_name=lambda i:'H100',set_device=lambda i:None)))
    monkeypatch.setattr(bench,'check_devices',lambda ids:None)
    monkeypatch.setattr(bench,'config_filename',lambda *a:'config.json')
    def worker(task):
        assert task['gpu_id']==0 and task['batch_sizes']==[17]
        return dict(configs={(128,256):{17:CFG}},measurements=[],identity={},gpu_id=0)
    monkeypatch.setattr(bench,'tune_on_gpu',worker)
    bench.main(args)
    assert core.read_configs(tmp_path/'config.json',128)=={17:CFG}
    reports=list((tmp_path/'reports').glob('*.json'))
    assert len(reports)==1
    report=json.loads(reports[0].read_text())
    assert report['files']==[str(tmp_path/'config.json')] and report['plan']['M']==[17]
    assert 'Tuning completed' in capsys.readouterr().out


def test_concurrent_runs_merge_without_lost_updates(tmp_path):
    script='''import sys
from fp8_tuning import save_configs
config=dict(BLOCK_SIZE_M=64,BLOCK_SIZE_N=128,BLOCK_SIZE_K=128,GROUP_SIZE_M=32,num_warps=4,num_stages=2)
save_configs(sys.argv[1],{int(sys.argv[2]):config},128)
'''
    path=tmp_path/'config.json'
    processes=[subprocess.Popen([sys.executable,'-c',script,str(path),str(m)],cwd=ROOT) for m in range(1,9)]
    assert all(p.wait()==0 for p in processes)
    assert set(core.read_configs(path,128))==set(range(1,9))


def test_multiworker_exception_propagates(monkeypatch,tmp_path,capsys):
    args=bench.build_parser().parse_args(['--shape','128','256','--out-dtype','float16','--save-path',str(tmp_path)])
    monkeypatch.setattr(bench,'load_runtime',lambda:{})
    monkeypatch.setattr(bench,'check_devices',lambda ids:None)
    monkeypatch.setattr(bench,'config_filename',lambda *a:'config.json')
    monkeypatch.setattr(bench,'torch',SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2,
        get_device_name=lambda i:'H100',set_device=lambda i:None)))
    class Pool:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def map(self,worker,tasks):
            assert len(tasks)==2
            raise RuntimeError('spawn worker failed')
    monkeypatch.setattr(bench.mp,'get_context',lambda method:SimpleNamespace(Pool=lambda n:Pool()))
    with pytest.raises(RuntimeError,match='spawn worker failed'): bench.main(args)
    assert not list(tmp_path.glob('*.json')) and 'completed' not in capsys.readouterr().out.lower()


@pytest.mark.parametrize('root_option', ['default', 'environment', 'cli', 'cli_equals'])
def test_batch_real_shape_saves_are_isolated(tmp_path, root_option):
    # Isolate a copy of the shell entry points. The Python driver replaces only
    # GPU dependencies/worker and uses the real planner, merge and config writer.
    project = tmp_path/'project'
    for relative in ['examples/tune_qwen3_models.sh', 'scripts/tune_qwen3.sh', 'scripts/tune_custom.sh']:
        target = project/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT/relative).read_text())
    driver = tmp_path/'python-driver'
    driver.write_text(f'''#!{sys.executable}
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, {str(ROOT)!r})
import benchmark_w8a8_block_fp8 as bench
import fp8_tuning as core
configs = {{
 'Qwen/Qwen3-8B': dict(architectures=['Qwen3ForCausalLM'], hidden_size=4096, head_dim=128,
     num_attention_heads=32, num_key_value_heads=8, intermediate_size=12288),
 'Qwen/Qwen3-30B-A3B': dict(architectures=['Qwen3MoeForCausalLM'], hidden_size=2048, head_dim=128,
     num_attention_heads=32, num_key_value_heads=4, num_hidden_layers=48, num_experts=128,
     decoder_sparse_step=1, mlp_only_layers=[], moe_intermediate_size=768),
}}
def load(model, tp, trust):
 config = configs[model]
 shapes, sources = core.model_shapes(config, tp)
 return config, shapes, sources
launch = {CFG!r}
def worker(task):
 return dict(configs={{s:{{m:launch for m in task['batch_sizes']}} for s in task['weight_shapes']}},
             measurements=[], identity={{}}, gpu_id=0)
bench.load_model_shapes = load
bench.load_runtime = lambda: {{'test': 'CPU flow only'}}
bench.check_devices = lambda ids: None
bench.torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1,
                     get_device_name=lambda i:'testGPU', set_device=lambda i:None))
sys.modules['vllm.utils.platform_utils'] = SimpleNamespace(get_device_name_as_file_name=lambda i:'testGPU')
bench.tune_on_gpu = worker
sys.argv = ['benchmark'] + sys.argv[2:]
sys.exit(bench.cli())
''')
    driver.chmod(0o755)
    env = dict(os.environ, PYTHON=str(driver), BATCH_SIZE='1', TRUST_REMOTE_CODE='0', OUT_DTYPE='float16')
    env.pop('SAVE_PATH', None)
    extra = []
    root = project/'tuned_configs'/'batch'
    if root_option == 'environment':
        root = tmp_path/'custom root'
        env['SAVE_PATH'] = str(root)
    elif root_option in ('cli', 'cli_equals'):
        root = tmp_path/'cli root'
        env['SAVE_PATH'] = str(tmp_path/'unused environment root')
        extra = ['--save-path', str(root)] if root_option == 'cli' else ['--save-path=' + str(root)]
    command = ['bash', str(project/'examples/tune_qwen3_models.sh'), *extra]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'completed successfully' in result.stdout
    for model_id in ['Qwen_Qwen3-8B', 'Qwen_Qwen3-30B-A3B']:
        for tp in [1,2,4,8]:
            output = root/model_id/f'tp_{tp}'
            files = list(output.glob('*.json'))
            assert len(files) == (4 if model_id == 'Qwen_Qwen3-8B' else 2)
            assert all(set(core.read_configs(path,128)) == {1} for path in files)
            assert len(list((output/'reports').glob('*.json'))) == 1
    # Isolation must not disable explicit overwrite protection for a rerun.
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 1 and '8 task(s) failed' in result.stderr
    assert 'completed successfully' not in result.stdout

    result = subprocess.run([*command, '--overwrite'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'completed successfully' in result.stdout


def test_single_call_is_default_benchmark_semantics():
    import inspect
    args = bench.build_parser().parse_args(['--shape','128','256','--out-dtype','float16'])
    assert args.calls_per_event == 1
    assert inspect.signature(bench.benchmark_config).parameters['calls_per_event'].default == 1


@pytest.mark.parametrize('wrapper,positionals', [('tune_custom.sh',['model']),
    ('tune_qwen3.sh',[]),('tune_qwen3_coder.sh',[])])
@pytest.mark.parametrize('dtype',[None,'float16','bfloat16'])
def test_wrapper_only_forwards_explicit_dtype(wrapper,positionals,dtype,fake_python):
    env=dict(fake_python,STUB_STATUS='0')
    if dtype is not None: env['OUT_DTYPE']=dtype
    result=subprocess.run(['bash',str(ROOT/'scripts'/wrapper),*positionals],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    args=Path(env['CAPTURE']).read_text().splitlines()
    if dtype is None:
        assert '--out-dtype' not in args
    else:
        assert args[args.index('--out-dtype')+1]==dtype
