"""Optional hardware checks. A skip is NOT evidence that GPU validation passed."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
torch=pytest.importorskip('torch',reason='Not executed — CUDA GPU/PyTorch unavailable')
pytest.importorskip('vllm',reason='Not executed — vLLM unavailable')
import benchmark_w8a8_block_fp8 as bench
from fp8_tuning import read_configs, save_configs

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Not executed — GPU unavailable')


@pytest.mark.parametrize('dtype',['float16','bfloat16','float32'])
@pytest.mark.parametrize('m',[1,17,64])
def test_cuda_tuning_and_runtime_loader(dtype,m,tmp_path,monkeypatch):
    bench.load_runtime()
    bench.check_devices([0])
    args=bench.build_parser().parse_args(['--shape','129','256','--out-dtype',dtype,
                                         '--batch-size',str(m),'--measurements','3'])
    candidates=[bench.default_config(128,128),dict(bench.default_config(128,128),BLOCK_SIZE_M=16)]
    cfg,report=bench.tune(m,129,256,args,candidates)
    assert report['winner']['correctness']['max_abs_error'] >= 0
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    # Isolated official loader path; no installed vLLM source/config is mutated.
    monkeypatch.setattr(fp8_utils,'__file__',str(tmp_path/'fp8_utils.py'))
    args.save_path=str(tmp_path/'configs')
    filename=bench.config_filename(129,256,128,128)
    save_configs(tmp_path/'configs'/filename,{m:cfg},128)
    try:
        bench.verify_installed(args,bench.plan(args))
    finally:
        fp8_utils.get_w8a8_block_fp8_configs.cache_clear()
