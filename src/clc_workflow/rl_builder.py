"""MDAveragedOptBuilder — real per-episode MD reward input for rl-matdesign.

Built for rl-matdesign's perovskite Level-2 defect-doping scenario
(``configs/perovskite.yaml`` in the rl-matdesign repo), wired in via that
project's FQN builder mechanism: ``builder: clc_workflow.rl_builder:MDAveragedOptBuilder``.

Pipeline, per candidate structure::

    candidate -> DefectSiteBuilder.build()          (rl_matdesign, no physics)
              -> LAMMPS NPT MD, LOCAL subprocess     (this module, via lammps_io)
              -> average the last `average_last_ps`  (lammps_io.average_frames)
              -> DP geometry-optimize (2nd model)     (rl_matdesign.utils.structure.relax_structure)

Everything *after* this builder — scoring the relaxed structure at one or more
``(temperature, pO2_1, pO2_2)`` fparam triples — is handled by rl-matdesign's
existing, already fparam-aware ``dp_property`` backend; no custom predictor is
needed (see the ``properties:`` list in ``configs/perovskite.yaml``).

Why this lives here and not in rl-matdesign: all of the MD/LAMMPS-specific
mechanics (deck rendering, trajectory averaging with vacancy-hop detection) are
reused directly from :mod:`clc_workflow.lammps_io` — the same code stages 2-4 of
this repo's own structures/ pipeline use — rather than reimplemented, per
rl-matdesign's CLAUDE.md convention of staying generic and free of LAMMPS
orchestration.

**Local execution only.** MD runs as a synchronous ``subprocess.run`` of the
LAMMPS binary on the same machine training RL — no dpdispatcher / Bohrium
submission. That machinery is for the offline structures/ pipeline; per-episode
MD needs to be cheap and synchronous, which is only realistic with a fast MD
model (e.g. dpa4c, per the user's own note that it's cheap enough to run for
real every episode).

Requires ``ase`` + ``deepmd-kit`` (the ``rl_reward`` extra) and rl-matdesign
importable on PYTHONPATH in the same environment (this module imports
``rl_matdesign.predictors.builders.defect_site`` and
``rl_matdesign.utils.structure``).

``md.env`` (optional dict, e.g. ``{DP_BACKEND_PLUGIN_PATH: /path/to/lib,
LD_LIBRARY_PATH: ...}``) is merged into the LAMMPS subprocess's environment on
top of the current process's own ``os.environ`` — some deepmd-kit builds need
one of these set to find a backend plugin .so, and setting it here survives a
detached/nohup training run better than relying on the launching shell's env.

**A failed realization does not fail the candidate.** ``build()`` tries every
one of the ``n_configs`` random realizations independently; if MD or the
post-MD relax raises for one (a real failure mode in practice — an unstable
NPT run, a GPU/compute-capability mismatch, an interstitial too close to an
existing atom), that realization is skipped (logged, not silently dropped)
and the rest still run. The reward is computed from whichever realizations
succeeded. Only if *all* of them fail does ``build()`` raise — there is
nothing left to score at that point, and propagating that up (rather than
swallowing it) is what lets a systematic problem (wrong GPU target, bad
model path) surface instead of silently reporting a meaningless reward. A
skipped realization still cost real MD time; this is about not letting one
bad roll crash an entire training run, not about making failures free.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np


class MDAveragedOptBuilder:
    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        from rl_matdesign.predictors.builders.defect_site import DefectSiteBuilder

        # DefectSiteBuilder reads the same top-level cfg (base_poscar, group names,
        # site symbols, interstitial_min_dist, ...) — this builder just adds the
        # `md:` / `opt:` blocks on top.
        self._inner = DefectSiteBuilder(cfg, seed=seed)

        md = dict(cfg.get("md") or {})
        if not md.get("model"):
            raise ValueError(
                "MDAveragedOptBuilder needs 'md.model' — a LAMMPS-ready (frozen) "
                "DeepMD model, e.g. dpa4c.pth."
            )
        if str(md["model"]).endswith((".ckpt.pt", ".ckpt")):
            from .lammps_io import freeze_command

            raise ValueError(
                f"md.model = {md['model']!r} looks like a training checkpoint. No "
                "LAMMPS pair style (Kokkos or stock) can load one directly — only "
                "dp_opt.py/dp_phonon.py and this builder's own `opt.model` step "
                "read a checkpoint via the ASE calculator. Freeze it first:\n"
                f"    {freeze_command('model.pth', md.get('head') or '<HEAD>', md['model'])}\n"
                "then set md.model to the frozen .pth/.pb file"
                + ("" if md.get("head") else " (and md.head to the head you froze)")
                + ". Checked once at builder construction, before any episode runs, "
                "so a misconfigured MD model fails immediately instead of after a "
                "warmup episode already spent real predictor time."
            )
        if not md.get("lmp_bin"):
            raise ValueError(
                "MDAveragedOptBuilder needs 'md.lmp_bin' — path to the LAMMPS binary "
                "on THIS machine (local execution, no Bohrium dispatch)."
            )
        self.md: Dict[str, Any] = md

        opt = dict(cfg.get("opt") or {})
        if not opt.get("model"):
            raise ValueError(
                "MDAveragedOptBuilder needs 'opt.model' — the (possibly different) "
                "DeepMD checkpoint used to relax the MD-averaged structure."
            )
        self.opt: Dict[str, Any] = opt

        self.keep_scratch: bool = bool(cfg.get("md_keep_scratch", False))
        self.scratch_root: Optional[str] = cfg.get("md_scratch_root")
        self._geo_calc = None  # lazily built, reused across calls (avoid reloading the model)

    # ------------------------------------------------------------------

    def build(
        self,
        candidate: Dict[str, Dict[str, Any]],
        *,
        n_configs: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> List["ase.Atoms"]:
        if rng is None:
            rng = np.random.default_rng()
        raw_structures = self._inner.build(candidate, n_configs=n_configs, rng=rng)

        results: List["ase.Atoms"] = []
        failures: List[str] = []
        for i, atoms in enumerate(raw_structures):
            try:
                results.append(self._md_average_and_optimize(atoms, tag=i, rng=rng))
            except Exception as exc:  # noqa: BLE001 - one bad realization must not
                # kill the whole (possibly multi-hour) training run. The GPU time for
                # this attempt was still spent -- it just doesn't get to poison every
                # other realization's reward, or the episode/warmup loop calling this.
                msg = f"{type(exc).__name__}: {exc}"
                failures.append(msg)
                print(
                    f"[rl_builder] structure {i + 1}/{len(raw_structures)} failed "
                    f"MD/relax, skipping it: {msg}"
                )

        if not results:
            raise RuntimeError(
                f"MDAveragedOptBuilder: all {len(raw_structures)} realizations failed "
                f"MD/relax for this candidate -- nothing left to score. First failure: "
                f"{failures[0] if failures else '?'}"
            )
        if failures:
            print(
                f"[rl_builder] {len(failures)}/{len(raw_structures)} realizations "
                f"failed and were skipped; reward computed from the remaining "
                f"{len(results)}."
            )
        return results

    def composition_formula(self, candidate: Dict[str, Dict[str, Any]]) -> Optional[str]:
        fn = getattr(self._inner, "composition_formula", None)
        return fn(candidate) if callable(fn) else None

    # ------------------------------------------------------------------

    @staticmethod
    def _check_structure(structure, min_dist: float) -> None:
        """
        Reject a candidate whose atoms overlap, before any MD time is spent on it.

        A substitution never moves an atom, so this can only fire on the interstitial
        branch of DefectSiteBuilder -- but that is exactly the branch that can place an
        atom on top of another, and a close contact does not announce itself as a bad
        structure.  It announces itself 30 steps into MD as a Kokkos abort inside
        `NBinKokkos::bin_atoms()`, which reads like a GPU problem and is not one.

        Distances use the minimum-image convention via pymatgen's distance_matrix, so a
        contact across a periodic boundary counts.

        The 1.7 A default is calibrated against real output of DefectSiteBuilder on
        perovskite.vasp, not guessed.  The substitution and vacancy branches both leave
        every atom on a template site, so their true minimum is the B-O bond at 1.926 A;
        the interstitial branch with `interstitial_min_dist: 1.5` produces Ca-Ca at
        1.506 A and Ca-Sr at 1.526 A.  1.7 A sits in the gap and separates the two
        cleanly.  Anything much lower -- 1.2 A, say -- passes the broken structures too
        and the check does nothing.
        """
        import numpy as _np

        n = len(structure)
        d = structure.distance_matrix + _np.eye(n) * 1e9
        i, j = _np.unravel_index(d.argmin(), d.shape)
        dmin = float(d[i, j])
        if dmin < min_dist:
            raise ValueError(
                f"candidate structure has overlapping atoms: {structure[i].specie}"
                f"(#{i}) and {structure[j].specie}(#{j}) are {dmin:.3f} A apart, "
                f"below md.min_dist = {min_dist:.3f} A. Formula "
                f"{structure.composition.formula}, {n} atoms. Raise "
                "`interstitial_min_dist` in the rl-matdesign config, or lower "
                "`md.min_dist` if this contact is intentional."
            )

    def _md_average_and_optimize(self, atoms: "ase.Atoms", *, tag: int, rng: np.random.Generator):
        from pymatgen.io.ase import AseAtomsAdaptor

        from .lammps_io import (
            average_frames,
            read_frames_since,
            render_input_lammps,
            write_conf_lmp,
        )

        type_map = sorted({s for s in atoms.get_chemical_symbols()})
        scratch = tempfile.mkdtemp(prefix=f"rl_md_{tag}_", dir=self.scratch_root)
        failed = False
        try:
            conf_path = os.path.join(scratch, "conf.lmp")
            input_path = os.path.join(scratch, "input.lammps")

            structure = AseAtomsAdaptor.get_structure(atoms)
            self._check_structure(structure, float(self.md.get("min_dist", 1.7)))
            write_conf_lmp(conf_path, structure, type_map, source="rl_matdesign candidate")

            nsteps = int(self.md.get("nsteps", 100000))
            timestep_ps = float(self.md.get("timestep_ps", 0.002))
            seed = int(self.md.get("velocity_seed") or (1 + int(rng.integers(1, 2**31 - 2))))
            deck = render_input_lammps(
                type_map=type_map,
                model_path=self.md["model"],
                temperature_k=self.md.get("temperature_k", 300.0),
                nsteps=nsteps,
                timestep_ps=timestep_ps,
                thermo_freq=self.md.get("thermo_freq", 10),
                dump_freq=self.md.get("dump_freq", 100),
                pressure_bar=self.md.get("pressure_bar", 0.0),
                tau_t=self.md.get("tau_t", 0.1),
                tau_p=self.md.get("tau_p", 0.5),
                model_head=self.md.get("head"),
                pair_style=self.md.get("pair_style", "deepmd"),
                atom_style=self.md.get("atom_style", "atomic"),
                plugin=self.md.get("plugin"),
                seed=seed,
            )
            with open(input_path, "w") as f:
                f.write(deck)

            cmd = [self.md["lmp_bin"], *self.md.get("lmp_args", []), "-in", "input.lammps"]
            run_env = {**os.environ, **{str(k): str(v) for k, v in (self.md.get("env") or {}).items()}}
            result = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True, env=run_env)
            if result.returncode != 0:
                raise RuntimeError(
                    f"LAMMPS MD failed (exit {result.returncode}) in {scratch}:\n"
                    f"--- stdout (tail) ---\n{result.stdout[-4000:]}\n"
                    f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
                )

            traj_path = os.path.join(scratch, "traj.lammpstrj")
            average_last_ps = float(self.md.get("average_last_ps", 20.0))
            min_step = max(0, nsteps - int(round(average_last_ps / timestep_ps)))
            frames = read_frames_since(traj_path, min_step)
            avg_frame, stats = average_frames(frames, rms_max=self.md.get("avg_rms_max", 0.5))
            if stats["n_hoppers"]:
                print(
                    f"[rl_builder] MD-average window has {stats['n_hoppers']} atom(s) "
                    f"that changed sites (max_rms={stats['max_rms']:.2f} A) — kept at "
                    "their last-frame position."
                )

            from ase import Atoms as AseAtoms

            averaged = AseAtoms(
                symbols=[type_map[t - 1] for t in avg_frame["types"]],
                positions=avg_frame["coords"],
                cell=avg_frame["cell"],
                pbc=True,
            )

            from rl_matdesign.utils.structure import relax_structure

            if self._geo_calc is None:
                from deepmd.calculator import DP as DPCalculator

                head = self.opt.get("head")
                self._geo_calc = DPCalculator(
                    model=self.opt["model"], **({"head": head} if head else {})
                )
            return relax_structure(
                averaged,
                calc=self._geo_calc,
                fmax=float(self.opt.get("fmax", 0.01)),
                steps=int(self.opt.get("steps", 2000)),
                relax_cell=bool(self.opt.get("relax_cell", True)),
            )
        except BaseException:
            # The failure messages above name `scratch` so the deck, conf.lmp and the
            # partial trajectory can be inspected -- deleting it here would make every
            # one of those messages point at a directory that no longer exists, which is
            # how a reproducible MD blow-up turns into an unreproducible one.
            failed = True
            raise
        finally:
            if failed:
                print(f"[rl_builder] kept scratch dir for inspection: {scratch}")
            elif not self.keep_scratch:
                shutil.rmtree(scratch, ignore_errors=True)
