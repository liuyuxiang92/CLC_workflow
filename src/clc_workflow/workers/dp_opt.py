#!/usr/bin/env python3
"""
Stage 5 worker -- DP geometry optimisation of ONE structure, in its own working directory.

    python dp_opt.py [--structure POSCAR_md_avg] --model model.ckpt.pt
                     [--head DOWNSTREAM_DATA] [--fmax 0.005] [--nsteps 2000]
                     [--pressure-gpa 0] [--traj-interval 5]

Self-contained on purpose: dpdispatcher forwards only this file to the compute node, so
it must not import from CLC_workflow/src.  submit.py bakes the parameters above into the
job command, which keeps the worker stateless and the command self-documenting.

Physics is unchanged from the legacy opt_multi.py -- ASE LBFGS on
UnitCellFilter(atoms, scalar_pressure=P*0.006242), trajectory every `traj-interval`
steps.  What changed:

  * model path and head are arguments, not hardcoded DP(model='model.ckpt.pt',
    head='DOWNSTREAM_DATA');
  * the pointless Pool(processes=1) is gone -- dpdispatcher handles concurrency;
  * the trajectory is a fixed 'opt.traj'.  The old name was
    traj.{poscar_name.split('.')[-1]}, which for a directory called o891 produced
    'traj.o891';
  * convergence is RECORDED in opt_result.json rather than inferred downstream from a
    frame count.  parse_traj.py tested nframes == 801 while deal_unconverged_POSCAR.py
    tested == 401 -- both are nsteps/interval+1 for different nsteps, so they could not
    both be right.
"""
import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", default="POSCAR_md_avg")
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default=None, help="multi-task model head, if any")
    ap.add_argument("--fmax", type=float, default=0.005)
    ap.add_argument("--nsteps", type=int, default=2000)
    ap.add_argument("--pressure-gpa", type=float, default=0.0)
    ap.add_argument("--traj-interval", type=int, default=5)
    ap.add_argument("--traj", default="opt.traj")
    ap.add_argument("--result", default="opt_result.json")
    ap.add_argument("--force", action="store_true", help="rerun even if opt.traj exists")
    args = ap.parse_args()

    if not os.path.isfile(args.structure):
        sys.exit(f"[ERROR] {args.structure} not found in {os.getcwd()}")
    if os.path.isfile(args.traj) and os.path.isfile(args.result) and not args.force:
        print(f"[skip] {args.traj} and {args.result} already exist")
        return

    from ase.io import read, Trajectory
    from ase.optimize import LBFGS
    try:                                   # ASE >= 3.23 moved/renamed the cell filter
        from ase.filters import UnitCellFilter
    except ImportError:
        from ase.constraints import UnitCellFilter
    from deepmd.calculator import DP

    calc = DP(model=args.model, **({"head": args.head} if args.head else {}))

    atoms = read(args.structure)
    atoms.calc = calc
    # legacy conversion: GPa -> eV/A^3
    aim_stress = float(args.pressure_gpa) * 0.006242
    ucf = UnitCellFilter(atoms, scalar_pressure=aim_stress)

    traj = Trajectory(args.traj, "w", atoms)
    opt = LBFGS(ucf)
    opt.attach(traj.write, interval=int(args.traj_interval))

    t0 = time.time()
    status, error = "ok", None
    try:
        opt.run(fmax=float(args.fmax), steps=int(args.nsteps))
    except Exception as e:                 # OOM and friends: record instead of dying silently
        status, error = "error", f"{type(e).__name__}: {e}"
        print(f"[ERROR] optimisation aborted: {error}")
    elapsed = time.time() - t0

    result = {"status": status, "error": error, "structure": args.structure,
              "model": args.model, "head": args.head,
              "fmax_target": float(args.fmax), "nsteps_max": int(args.nsteps),
              "pressure_gpa": float(args.pressure_gpa),
              "elapsed_sec": round(elapsed, 3)}
    try:
        import numpy as np
        forces = atoms.get_forces()
        result["fmax_final"] = float(np.sqrt((forces ** 2).sum(axis=1).max()))
        result["energy"] = float(atoms.get_potential_energy())
        result["n_atoms"] = int(len(atoms))
        result["cell"] = [[float(v) for v in row] for row in atoms.get_cell()]
        result["volume"] = float(atoms.get_volume())
        result["nsteps_taken"] = int(getattr(opt, "nsteps", -1))
        result["converged"] = bool(result["fmax_final"] <= float(args.fmax))
    except Exception as e:
        result["converged"] = False
        result.setdefault("error", f"post-processing failed: {e}")

    with open(args.result, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[*] converged={result.get('converged')} "
          f"fmax_final={result.get('fmax_final')} steps={result.get('nsteps_taken')} "
          f"({elapsed:.1f} s) -> {args.result}")


if __name__ == "__main__":
    main()
