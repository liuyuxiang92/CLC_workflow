#!/usr/bin/env python3
"""
Stage 6 -- last frame of opt.traj -> optimized_POSCAR (or unconverged_POSCAR).

    clc collect-opt config.yaml [--limit N] [--force] [--nprocs N]

Replaces parse_traj.py and deal_unconverged_POSCAR.py.  Two changes:

  * convergence comes from opt_result.json (written by dp_opt.py), not from a magic
    frame count -- parse_traj.py tested nframes == 801 while deal_unconverged_POSCAR.py
    tested == 401, and both are nsteps/interval+1 for different nsteps;
  * the safe-distance screen keeps the original thresholds and verdict but uses a
    KD-tree instead of an O(N^2) Python loop over a 2x2x2 replication (2560 atoms
    ~ 3.3M pairs per structure).
"""
import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from clc_workflow.clc_config import load_config, MANIFEST_NAME
from clc_workflow.phonon_analysis import min_pair_violation
OK_NAME = "optimized_POSCAR"
BAD_NAME = "unconverged_POSCAR"


def _one(job, force=False, skip_distance_check=False):
    d, rel = job["dir"], job["rel"]
    traj = os.path.join(d, "opt.traj")
    res_path = os.path.join(d, "opt_result.json")
    if not os.path.isfile(traj):
        return {"rel": rel, "status": "no_traj"}

    converged, energy = None, None
    if os.path.isfile(res_path):
        try:
            with open(res_path) as f:
                res = json.load(f)
            converged = res.get("converged")
            energy = res.get("energy")
        except Exception:
            pass
    out_name = OK_NAME if converged else BAD_NAME
    out = os.path.join(d, out_name)
    if not force and os.path.isfile(out):
        return {"rel": rel, "status": "skipped", "written": out_name,
                "converged": converged}

    try:
        from ase.io import read
        from ase.io.trajectory import Trajectory
        with Trajectory(traj, "r") as t:
            nframes = len(t)
            if nframes == 0:
                return {"rel": rel, "status": "empty_traj"}
            atoms = t[-1]
    except Exception as e:
        return {"rel": rel, "status": "failed", "detail": f"read: {e}"}

    if converged is None:
        # no opt_result.json (e.g. a legacy tree): fall back to "assume converged"
        # rather than guessing from the frame count, and flag it
        converged = True
        out_name = OK_NAME
        out = os.path.join(d, out_name)

    if not skip_distance_check:
        try:
            violated, detail = min_pair_violation(
                atoms.get_cell(), atoms.get_positions(), atoms.get_chemical_symbols())
        except Exception as e:
            return {"rel": rel, "status": "failed", "detail": f"distance screen: {e}"}
        if violated:
            return {"rel": rel, "status": "too_close", "detail": detail,
                    "converged": converged, "nframes": nframes}

    try:
        from ase.io import write as ase_write
        ase_write(out, atoms, format="vasp", direct=True, sort=True)
    except Exception as e:
        return {"rel": rel, "status": "failed", "detail": f"write: {e}"}

    return {"rel": rel, "status": "ok", "written": out_name, "converged": bool(converged),
            "energy": energy, "nframes": nframes, "n_atoms": len(atoms)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-distance-check", action="store_true",
                    help="write the frame even if atoms are dangerously close")
    ap.add_argument("--nprocs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = cfg["output_root"]
    man = pd.read_csv(os.path.join(root, MANIFEST_NAME))
    if args.limit:
        man = man.head(args.limit)

    jobs = [{"dir": os.path.join(root, r["path"]), "rel": r["path"]}
            for _, r in man.iterrows()]
    worker = partial(_one, force=args.force,
                     skip_distance_check=args.skip_distance_check)
    if args.nprocs > 1 and len(jobs) > 1:
        with Pool(args.nprocs) as pool:
            results = pool.map(worker, jobs, chunksize=4)
    else:
        results = [worker(j) for j in jobs]

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("[*] " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        n_conv = sum(1 for r in ok if r["converged"])
        print(f"[*] converged {n_conv}/{len(ok)} -> {OK_NAME}; "
              f"{len(ok) - n_conv} -> {BAD_NAME}")
        pd.DataFrame(ok).to_csv(os.path.join(root, "opt_summary.csv"), index=False)
        print(f"[*] summary -> {os.path.join(root, 'opt_summary.csv')}")
    for r in [x for x in results if x["status"] == "too_close"][:5]:
        print(f"    [too close] {r['rel']}: {r.get('detail','')}")
    for r in [x for x in results if x["status"] == "failed"][:5]:
        print(f"    [FAILED] {r['rel']}: {r.get('detail','')}")


if __name__ == "__main__":
    main()
