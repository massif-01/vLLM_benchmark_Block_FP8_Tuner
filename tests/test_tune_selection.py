"""CPU regressions for finalist selection; no claim of GPU numerical validation."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_w8a8_block_fp8 as bench


class OutOfResources(Exception):
    pass


class Tensor:
    def __sub__(self, value): return self
    def __add__(self, value): return self
    def __mul__(self, value): return self
    def to(self, dtype): return self


@pytest.fixture
def tuning_runtime(monkeypatch):
    torch = SimpleNamespace(manual_seed=lambda seed: None, rand=lambda *a, **kw: Tensor(),
                            float8_e4m3fn='fp8', float16='fp16',
                            cuda=SimpleNamespace(current_device=lambda: 0))
    triton = SimpleNamespace(cdiv=lambda x, y: (x+y-1)//y,
                            runtime=SimpleNamespace(autotuner=SimpleNamespace(OutOfResources=OutOfResources)))
    monkeypatch.setattr(bench, 'torch', torch)
    monkeypatch.setattr(bench, 'triton', triton)
    monkeypatch.setattr(bench, 'reference_matmul', lambda *a: Tensor())
    args = bench.build_parser().parse_args(['--shape', '256', '256', '--batch-size', '1',
                                          '--block-n', '256', '--block-k', '256'])
    baseline = bench.default_config(256, 256)
    candidate = dict(baseline, BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32)
    return args, baseline, candidate


def test_unavailable_default_does_not_discard_valid_candidate(monkeypatch, tuning_runtime):
    args, baseline, candidate = tuning_runtime
    measured, checked = [], []
    def measure(*values):
        config = values[5]
        measured.append(config)
        if config == baseline:
            raise OutOfResources('shared memory limit')
        return {'median_us': 1.0}
    def check(*values):
        config = values[5]
        checked.append(config)
        assert config != baseline, 'An unavailable default must not be retried'
        return {'max_abs_error': 0.0}
    monkeypatch.setattr(bench, 'benchmark_config', measure)
    monkeypatch.setattr(bench, 'check_correctness', check)
    winner, report = bench.tune(1, 256, 256, args, [candidate])
    assert winner == candidate
    assert measured == [baseline, candidate, candidate] and checked == [candidate]
    assert report['resource_failures'] == 1
    assert report['baseline']['status'] == 'unavailable'
    assert report['baseline']['reason'] == {'type': 'OutOfResources', 'message': 'shared memory limit'}


def test_available_default_is_remeasured_outside_top_three(monkeypatch, tuning_runtime):
    args, baseline, candidate = tuning_runtime
    candidates = [dict(candidate, GROUP_SIZE_M=group) for group in [1, 16, 32]]
    measured, checked = [], []
    def measure(*values):
        config = values[5]
        measured.append(config)
        return {'median_us': 100.0 if config == baseline else float(config['GROUP_SIZE_M'])}
    def check(*values):
        checked.append(values[5])
        return {'max_abs_error': 0.0}
    monkeypatch.setattr(bench, 'benchmark_config', measure)
    monkeypatch.setattr(bench, 'check_correctness', check)
    winner, report = bench.tune(1, 256, 256, args, candidates)
    assert winner == candidates[0]
    assert measured.count(baseline) == 2 and checked.count(baseline) == 1
    assert len(report['finalists']) == 4
    assert report['baseline']['status'] == 'validated'
    assert report['baseline']['measurement']['timing']['median_us'] == 100.0


def test_default_can_still_win(monkeypatch, tuning_runtime):
    args, baseline, candidate = tuning_runtime
    monkeypatch.setattr(bench, 'benchmark_config', lambda *v: {'median_us': 1.0 if v[5] == baseline else 2.0})
    monkeypatch.setattr(bench, 'check_correctness', lambda *v: {'max_abs_error': 0.0})
    winner, report = bench.tune(1, 256, 256, args, [candidate])
    assert winner == baseline and report['baseline']['status'] == 'validated'


def test_no_runnable_configs_still_fails(monkeypatch, tuning_runtime):
    args, baseline, candidate = tuning_runtime
    def fail(*values): raise OutOfResources('no resources')
    monkeypatch.setattr(bench, 'benchmark_config', fail)
    with pytest.raises(RuntimeError, match='No valid configuration'):
        bench.tune(1, 256, 256, args, [candidate])


def test_unknown_search_errors_still_propagate(monkeypatch, tuning_runtime):
    args, baseline, candidate = tuning_runtime
    def fail(*values): raise RuntimeError('unexpected launch failure')
    monkeypatch.setattr(bench, 'benchmark_config', fail)
    with pytest.raises(RuntimeError, match='unexpected launch failure'):
        bench.tune(1, 256, 256, args, [candidate])


def test_correctness_errors_still_propagate(monkeypatch, tuning_runtime):
    args, baseline, candidate = tuning_runtime
    monkeypatch.setattr(bench, 'benchmark_config', lambda *v: {'median_us': 1.0})
    def fail(*values): raise AssertionError('incorrect output')
    monkeypatch.setattr(bench, 'check_correctness', fail)
    with pytest.raises(AssertionError, match='incorrect output'):
        bench.tune(1, 256, 256, args, [candidate])
