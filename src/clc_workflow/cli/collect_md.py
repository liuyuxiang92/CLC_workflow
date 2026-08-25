#!/usr/bin/env python3
"""
Stage 4 -- time-average the last md.average_last_ps of MD into POSCAR_md_avg, the
starting point for DP opt.

    clc collect-md config.yaml [--limit N] [--force] [--nprocs N]

NOT the last frame.  A snapshot catches every atom at a random phase of its thermal
libration, so it hands the relaxation a geometry with ~0.1 A of noise per atom and a cell
that is one instantaneous fluctuation of the NPT box.  Averaging positions over the last
20 ps leaves only the static distortion -- the contrast is quantified for these materials
in code/md/SrCaFeCoO/feofe_windows.py, whose mean-structure Fe-O-Fe angles sit several
degrees away from the instantaneous ones.  See lammps_io.average_frames for how atoms
that hop sites during the window are kept out of the average.

Set md.average_last_ps to 0 to fall back to the old single-frame behaviour (still written
to POSCAR_md_avg, tagged method=last_frame in the summary) -- for a run whose dump_freq is
too coarse to average.

Frames are located by seeking backwards from EOF, so cost follows the length of the
window rather than the length of the trajectory, and a truncated final frame (killed job)
is skipped in favour of the last complete one.  LAMMPS types are mapped back to elements
through the manifest's per-family type map -- with several families in one tree this can
no longer be a global constant.

Under `fix npt ... aniso` the box stays orthogonal (verified on
code/md/SrCaFeCoO/clc_300k/traj.lammpstrj: `ITEM: BOX BOUNDS pp pp pp`, no xy/xz/yz),
but the triclinic case is handled anyway.
"""
import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from clc_workflow.clc_config import load_config, MANIFEST_NAME
from clc_workflow.lammps_io import (average_frames, read_frames_since,
                                    read_last_frame, write_poscar_from_frame)

OUT_NAME = "POSCAR_md_avg"
MIN_FRAMES = 10          # below this the "average" is barely one, so say so


def _one(job, force=False):
    d, type_map, rel = job["dir"], job["type_map"], job["rel"]
    dt, window_ps, rms_max = job["timestep_ps"], job["window_ps"], job["rms_max"]
    traj = os.path.join(d, "traj.lammpstrj")
    out = os.path.join(d, OUT_NAME)
    if not os.path.isfile(traj):
        return {"rel": rel, "status": "no_traj"}
    if not force and os.path.isfile(out):
        return {"rel": rel, "status": "skipped"}
    try:
        last = read_last_frame(traj)
        if window_ps > 0:
            min_step = last["step"] - int(round(window_ps / dt))
            frame, stats = average_frames(read_frames_since(traj, min_step),
                                          rms_max=rms_max)
            method = "avg"
        else:
            frame, method = last, "last_frame"
            stats = {"n_frames": 1, "step_first": last["step"], "step_last": last["step"],
                     "n_hoppers": 0, "max_rms": 0.0, "mean_rms": 0.0}
        t0, t1 = stats["step_first"] * dt, stats["step_last"] * dt
        comment = (f"{rel} mean of {t0:.1f}-{t1:.1f} ps, {stats['n_frames']} frames, "
                   f"{stats['n_hoppers']} hoppers" if method == "avg"
                   else f"{rel} MD step {frame['step']}")
        write_poscar_from_frame(out, frame, type_map, comment=comment)
    except Exception as e:
        return {"rel": rel, "status": "failed", "detail": str(e)}
    cell = frame["cell"]
    return {"rel": rel, "status": "ok", "method": method,
            "step_first": stats["step_first"], "step_last": stats["step_last"],
            "t_start_ps": round(t0, 3), "t_end_ps": round(t1, 3),
            "n_frames": stats["n_frames"], "n_atoms": frame["n_atoms"],
            "triclinic": frame["triclinic"],
            "a": float(cell[0, 0]), "b": float(cell[1, 1]), "c": float(cell[2, 2]),
            "max_rms": round(stats["max_rms"], 4),
            "mean_rms": round(stats["mean_rms"], 4),
            "n_hoppers": stats["n_hoppers"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--nprocs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = cfg["output_root"]
    man = pd.read_csv(os.path.join(root, MANIFEST_NAME))
    if args.limit:
        man = man.head(args.limit)

    md = cfg["md"]
    window_ps = float(md.get("average_last_ps") or 0.0)
    rms_max = md.get("avg_rms_max")
    rms_max = float(rms_max) if rms_max else None
    dt = float(md["timestep_ps"])
    if window_ps > 0:
        print(f"[*] averaging the last {window_ps:g} ps "
              f"(~{window_ps / (dt * int(md['dump_freq'])):.0f} frames at dump_freq "
              f"{md['dump_freq']}), hopper cutoff "
              f"{'off' if rms_max is None else f'{rms_max:g} A'}")
    else:
        print("[*] md.average_last_ps = 0 -- writing the last frame, not an average")

    jobs = [{"dir": os.path.join(root, r["path"]),
             "type_map": str(r["lammps_type_map"]).split(),
             "rel": r["path"], "timestep_ps": dt, "window_ps": window_ps,
             "rms_max": rms_max} for _, r in man.iterrows()]

    worker = partial(_one, force=args.force)
    if args.nprocs > 1 and len(jobs) > 1:
        with Pool(args.nprocs) as pool:
            results = pool.map(worker, jobs, chunksize=8)
    else:
        results = [worker(j) for j in jobs]

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("[*] " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        df = pd.DataFrame(ok)
        dest = os.path.join(root, "md_avg_summary.csv")
        df.to_csv(dest, index=False)
        print(f"[*] steps reached: min={df['step_last'].min()} max={df['step_last'].max()}")
        print(f"[*] lattice a: {df['a'].min():.3f}-{df['a'].max():.3f} A")
        if window_ps > 0:
            print(f"[*] frames averaged: min={df['n_frames'].min()} "
                  f"max={df['n_frames'].max()}   per-atom rms about the mean: "
                  f"max={df['max_rms'].max():.3f} A")
            thin = df[df["n_frames"] < MIN_FRAMES]
            if len(thin):
                # A short window means the job was killed early or dump_freq is too coarse
                # for md.average_last_ps -- either way the "mean structure" is a few
                # snapshots, and the libration it is supposed to remove is still in there.
                print(f"[warn] {len(thin)} structure(s) averaged fewer than {MIN_FRAMES} "
                      f"frames (min {int(thin['n_frames'].min())}) -- truncated run, or "
                      f"dump_freq too coarse for md.average_last_ps")
                for r in thin.head(5).itertuples():
                    print(f"    {r.rel}: {r.n_frames} frames over "
                          f"{r.t_start_ps:.1f}-{r.t_end_ps:.1f} ps")
            hop = df[df["n_hoppers"] > 0]
            if len(hop):
                # Expected to be empty at 300 K; rises with T and with delta, where
                # vacancy-mediated O diffusion is the point of the material.
                print(f"[warn] {len(hop)} structure(s) had atoms hopping sites during the "
                      f"window ({int(hop['n_hoppers'].sum())} atoms total, max "
                      f"{int(hop['n_hoppers'].max())} in one cell) -- those atoms kept "
                      f"their last-frame position instead of the mean")
                for r in hop.head(5).itertuples():
                    print(f"    {r.rel}: {r.n_hoppers} atoms, max rms {r.max_rms:.2f} A")
        if df["triclinic"].any():
            print(f"[warn] {int(df['triclinic'].sum())} frame(s) had a triclinic box")
        print(f"[*] summary -> {dest}")
    for r in [x for x in results if x["status"] == "failed"][:10]:
        print(f"    [FAILED] {r['rel']}: {r.get('detail','')}")


if __name__ == "__main__":
    main()
