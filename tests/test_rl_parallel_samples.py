"""Real RL worker processes with CPU-only MD and CUDA stand-ins."""
from __future__ import annotations

import pytest

pytest.importorskip("ase")
pytest.importorskip("rl_matdesign.utils.structure_samples")

from ase import Atoms
from ase.io import write

from rl_matdesign.structure_sources import StructureSourceSpec
from rl_matdesign.utils.parallel_structures import ParallelStructureBuilder
from rl_matdesign.utils.structure_samples import (
    SAMPLE_INFO_KEY,
    execute_many_seeded_samples,
)


_CPU_TORCH = """
class _Cuda:
    @staticmethod
    def is_available():
        return True
    @staticmethod
    def device_count():
        return 1
cuda = _Cuda()
def set_num_threads(count):
    pass
"""

_CPU_BUILDER = """
from clc_workflow.rl_builder import MDAveragedOptBuilder

class Builder(MDAveragedOptBuilder):
    def __init__(self, cfg, *, seed=None):
        super().__init__(cfg, seed=seed)
        self.fail_index = cfg.get("test_fail_index")

    def _md_average_and_optimize(self, atoms, *, tag, rng):
        if tag == self.fail_index:
            raise RuntimeError("simulated MD failure")
        result = atoms.copy()
        result.info["md_velocity_seed"] = 1 + int(rng.integers(1, 2**31 - 2))
        return result
"""


@pytest.mark.parametrize("fail_index", [None, 1])
def test_clc_worker_handshake_and_serial_parallel_parity(tmp_path, monkeypatch, fail_index):
    modules = tmp_path / "worker_modules"
    modules.mkdir()
    (modules / "torch.py").write_text(_CPU_TORCH)
    (modules / "clc_cpu_builder.py").write_text(_CPU_BUILDER)
    monkeypatch.syspath_prepend(str(modules))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    from clc_cpu_builder import Builder

    template = tmp_path / "POSCAR"
    cell = Atoms(
        "SrFeO3", cell=[4, 4, 4], pbc=True,
        scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
    ).repeat((2, 2, 2))
    write(template, cell, format="vasp", sort=True)
    cfg = {
        "builder": "clc_cpu_builder:Builder",
        "base_poscar": str(template),
        "md": {"model": "unused.pth", "lmp_bin": "unused-lmp"},
        "opt": {"model": "unused.ckpt.pt"},
        "test_fail_index": fail_index,
        "structure_parallel": {
            "devices": [0, 1], "workers_per_device": 1,
            "threads_per_worker": 1, "startup_timeout_s": 30,
            "timeout_s": 30, "min_successful_configs": 1,
        },
    }
    source = StructureSourceSpec("clc", cfg)
    kwargs = dict(n_configs=3, source_spec=source, eval_seed=19)
    a = {"A_dopant": {"Sr": .5, "Ca": .5}, "B_dopant": {"Fe": .5, "Mn": .5}}
    b = {**a, "A_defect": {"defect": "Sr_-0.125"}}
    serial = Builder(cfg)
    expected = execute_many_seeded_samples(serial, [a, b], **kwargs)
    pool = ParallelStructureBuilder(cfg["builder"], cfg, seed=7)
    try:
        actual = pool.build_many_seeded([a, b], **kwargs)
        for parallel_records, serial_records in zip(pool.last_sample_records, serial.last_sample_records):
            # Workers include a traceback; serial execution records the exception.
            assert [dict(record, error=None) for record in parallel_records] == [
                dict(record, error=None) for record in serial_records
            ]
            for parallel_record, serial_record in zip(parallel_records, serial_records):
                if serial_record["status"] == "failed":
                    assert parallel_record["error"].strip().endswith(serial_record["error"])
        reordered = pool.build_many_seeded([b, dict(reversed(list(a.items())))], **kwargs)
        scalar = pool.build_seeded(a, **kwargs)
    finally:
        pool.close()

    def signature(structures):
        return [
            (atoms.get_chemical_symbols(), atoms.positions.tolist(), atoms.cell.tolist(),
             atoms.info["md_velocity_seed"], atoms.info[SAMPLE_INFO_KEY])
            for atoms in structures
        ]

    assert signature(actual[0]) == signature(expected[0]) == signature(reordered[1]) == signature(scalar)
    assert signature(actual[1]) == signature(expected[1]) == signature(reordered[0])
    expected_statuses = ["succeeded", "failed", "succeeded"] if fail_index == 1 else ["succeeded"] * 3
    assert [record["status"] for record in serial.last_sample_records[0]] == expected_statuses
