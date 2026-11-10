"""CLC's explicit RL sample protocol, without LAMMPS or DeepMD inference.

The real defect-site constructor reads a temporary POSCAR. Only the expensive
MD/relax stage is replaced, so these tests exercise atom placement and RNG
continuation across both stages of a declared sample.
"""
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from clc_workflow.rl_builder import MDAveragedOptBuilder
from rl_matdesign.structure_sources import StructureSourceSpec
from rl_matdesign.utils.structure_samples import (
    SAMPLE_INFO_KEY,
    SamplingProtocolError,
    canonical_sample_candidate,
    execute_many_seeded_samples,
    execute_seeded_samples,
    sample_plan,
)


@pytest.fixture
def builder(tmp_path):
    symbols = ["Sr"] * 16 + ["Fe"] * 16 + ["O"] * 48
    positions = [(3 * (i % 5), 3 * ((i // 5) % 5), 3 * (i // 25))
                 for i in range(len(symbols))]
    path = tmp_path / "template.vasp"
    write(path, Atoms(symbols, positions=positions, cell=[18, 18, 18], pbc=True),
          format="vasp")
    return MDAveragedOptBuilder({
        "base_poscar": str(path),
        "md": {"model": "unused.pth", "lmp_bin": "unused-lammps"},
        "opt": {"model": "unused.ckpt.pt"},
    }, seed=12)


@pytest.fixture
def candidate():
    # Deliberately non-canonical order at both dictionary levels.
    return {"B_dopant": {"Mn": .25, "Fe": .5, "Co": .25},
            "A_defect": {"defect": "Sr_-0.125"},
            "A_dopant": {"Sr": .5, "Ca": .25, "Ba": .25}}


@pytest.fixture
def source():
    return StructureSourceSpec("bulk", {
        "builder": "clc_workflow.rl_builder:MDAveragedOptBuilder",
    })


def _mock_md(atoms, *, tag, rng):
    result = atoms.copy()
    result.info["md_tag"] = tag
    result.info["md_seed"] = int(rng.integers(1, 2**31))
    return result


def _signature(atoms):
    return (atoms.numbers.tolist(), atoms.positions.tolist(),
            atoms.cell.tolist(), atoms.info)


def test_sample_rng_continues_from_real_defect_build_into_md(
    builder, candidate, source, monkeypatch,
):
    sample = sample_plan(candidate, n_configs=1, source_spec=source, eval_seed=7)[0]
    rng = np.random.default_rng(sample.seed)
    expected = builder._inner.build(canonical_sample_candidate(candidate),
                                    n_configs=1, rng=rng)[0]
    expected_md_seed = int(rng.integers(1, 2**31))
    observed = []
    native_inner_build = builder._inner.build

    def inner_build(comp, *, n_configs, rng):
        observed.append((comp, n_configs, rng))
        return native_inner_build(comp, n_configs=n_configs, rng=rng)

    def md(atoms, *, tag, rng):
        assert rng is observed[0][2]
        return _mock_md(atoms, tag=tag, rng=rng)

    monkeypatch.setattr(builder._inner, "build", inner_build)
    monkeypatch.setattr(builder, "_md_average_and_optimize", md)
    actual = builder.build_sample(candidate, sample=sample)
    assert isinstance(actual, Atoms)
    assert len(observed) == 1 and observed[0][1] == 1
    assert list(observed[0][0]) == sorted(candidate)
    assert list(observed[0][0]["A_dopant"]) == ["Ba", "Ca", "Sr"]
    assert list(candidate) == ["B_dopant", "A_defect", "A_dopant"]
    assert actual.numbers.tolist() == expected.numbers.tolist()
    np.testing.assert_array_equal(actual.positions, expected.positions)
    assert actual.info["md_seed"] == expected_md_seed
    assert actual.info["md_tag"] == sample.sample_index
    assert actual.info[SAMPLE_INFO_KEY] == sample.record()


def test_candidate_order_and_sample_execution_order_preserve_results(
    builder, candidate, source, monkeypatch,
):
    monkeypatch.setattr(builder, "_md_average_and_optimize", _mock_md)
    plan = sample_plan(candidate, n_configs=3, source_spec=source, eval_seed=17)
    forward = {s.sample_index: _signature(builder.build_sample(candidate, sample=s))
               for s in plan}
    reordered = {key: dict(reversed(list(value.items())))
                 for key, value in reversed(list(candidate.items()))}
    reverse = {s.sample_index: _signature(builder.build_sample(reordered, sample=s))
               for s in reversed(plan)}
    assert forward == reverse
    assert len({repr(value[0]) for value in forward.values()}) > 1
    assert len({value[3]["md_seed"] for value in forward.values()}) == len(plan)


def test_serial_scalar_and_reordered_batch_use_same_sample_results(
    builder, candidate, source, monkeypatch,
):
    monkeypatch.setattr(builder, "_md_average_and_optimize", _mock_md)
    other = {**candidate, "A_dopant": {"Sr": .75, "Ca": .25}}
    kwargs = dict(n_configs=2, source_spec=source, eval_seed=19)
    batch = execute_many_seeded_samples(builder, [candidate, other], **kwargs)
    reverse = execute_many_seeded_samples(builder, [other, candidate], **kwargs)
    scalar = execute_seeded_samples(builder, candidate, **kwargs)
    assert [_signature(s) for s in batch[0]] == [_signature(s) for s in scalar]
    assert [_signature(s) for s in batch[0]] == [_signature(s) for s in reverse[1]]
    assert [_signature(s) for s in batch[1]] == [_signature(s) for s in reverse[0]]


def test_failed_declared_sample_is_recorded_without_replacement(
    builder, candidate, source, monkeypatch,
):
    attempted = []

    def md(atoms, *, tag, rng):
        attempted.append(tag)
        if tag == 1:
            raise RuntimeError("MD failed for declared sample")
        return _mock_md(atoms, tag=tag, rng=rng)

    monkeypatch.setattr(builder, "_md_average_and_optimize", md)
    plan = sample_plan(candidate, n_configs=3, source_spec=source, eval_seed=31)
    results = execute_many_seeded_samples(builder, [candidate], n_configs=3,
                                          source_spec=source, eval_seed=31)
    assert attempted == [0, 1, 2]
    assert [s.info[SAMPLE_INFO_KEY]["sample_index"] for s in results[0]] == [0, 2]
    records = builder.last_sample_records[0]
    assert [record["status"] for record in records] == ["succeeded", "failed", "succeeded"]
    assert records[1] == plan[1].record(
        status="failed", error="RuntimeError: MD failed for declared sample")
    with pytest.raises(RuntimeError, match="MD failed for declared sample"):
        builder.build_sample(candidate, sample=plan[1])
    assert attempted == [0, 1, 2, 1]


@pytest.mark.parametrize("kind", ["not_descriptor", "index", "path", "symbol"])
def test_non_generic_descriptor_is_rejected_before_construction(
    builder, candidate, source, monkeypatch, kind,
):
    sample = sample_plan(candidate, n_configs=1, source_spec=source, eval_seed=7)[0]
    invalid = {"not_descriptor": object(),
               "index": replace(sample, template_index=0),
               "path": replace(sample, template_path="other.vasp"),
               "symbol": replace(sample, site_symbol="Sr")}[kind]

    def forbidden(*args, **kwargs):
        pytest.fail("invalid descriptor must not construct or run MD")

    monkeypatch.setattr(builder._inner, "build", forbidden)
    monkeypatch.setattr(builder, "_md_average_and_optimize", forbidden)
    with pytest.raises(SamplingProtocolError, match="generic StructureSample"):
        builder.build_sample(candidate, sample=invalid)


@pytest.mark.parametrize("bad_raw", [[], [Atoms("H"), Atoms("He")], [object()]])
def test_inner_must_return_exactly_one_atoms_before_md(
    builder, candidate, source, monkeypatch, bad_raw,
):
    sample = sample_plan(candidate, n_configs=1, source_spec=source, eval_seed=7)[0]
    monkeypatch.setattr(builder._inner, "build", lambda *args, **kwargs: bad_raw)

    def forbidden(*args, **kwargs):
        pytest.fail("malformed inner output must not run MD")

    monkeypatch.setattr(builder, "_md_average_and_optimize", forbidden)
    with pytest.raises(SamplingProtocolError, match="exactly one ASE Atoms"):
        builder.build_sample(candidate, sample=sample)


@pytest.mark.parametrize("bad_output", [None, [Atoms("H")]])
def test_md_must_return_exactly_one_atoms(
    builder, candidate, source, monkeypatch, bad_output,
):
    sample = sample_plan(candidate, n_configs=1, source_spec=source, eval_seed=7)[0]
    monkeypatch.setattr(builder, "_md_average_and_optimize",
                        lambda *args, **kwargs: bad_output)
    with pytest.raises(SamplingProtocolError, match="MD/optimization must return"):
        builder.build_sample(candidate, sample=sample)


def test_native_build_still_preserves_partial_failure_behavior(
    builder, candidate, monkeypatch, capsys,
):
    tags = []

    def md(atoms, *, tag, rng):
        tags.append(tag)
        if tag == 1:
            raise RuntimeError("native failed realization")
        return _mock_md(atoms, tag=tag, rng=rng)

    monkeypatch.setattr(builder, "_md_average_and_optimize", md)
    results = builder.build(candidate, n_configs=3, rng=np.random.default_rng(7))
    assert tags == [0, 1, 2]
    assert [s.info["md_tag"] for s in results] == [0, 2]
    assert all(SAMPLE_INFO_KEY not in s.info for s in results)
    assert "1/3 realizations failed" in capsys.readouterr().out


def test_native_build_still_reports_all_failed_realizations(
    builder, candidate, monkeypatch,
):
    tags = []

    def md(atoms, *, tag, rng):
        tags.append(tag)
        raise RuntimeError("native all failed")

    monkeypatch.setattr(builder, "_md_average_and_optimize", md)
    with pytest.raises(RuntimeError, match="all 2 realizations failed"):
        builder.build(candidate, n_configs=2, rng=np.random.default_rng(7))
    assert tags == [0, 1]
