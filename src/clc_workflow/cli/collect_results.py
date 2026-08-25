#!/usr/bin/env python3
"""
Stage 8 -- join the manifest with every artifact into results.csv, then average the
independent SQS sets into results_avg.csv.

    clc collect config.yaml [--nprocs N] [--out results.csv]

results.csv has one row per (structure, temperature): composition, vacancy count, MD
final lattice, optimisation outcome, DP energy, the phonon thermal properties, S_config,
G_state and the stability verdict.

results_avg.csv has one row per (composition, o_tag, temperature) -- the quantity that
actually enters the ΔG analysis -- averaged over the n_sets independent SQS
realisations.  **The mean is taken over the sets that succeeded, not over n_sets.**
With 50 sets of which one has no energy or no phonons, the divisor is 49, and the row
records n_sets_total=50, n_sets_ok=49 and which set failed, so a thin average is
visible rather than silent.  A set is "ok" at a temperature when G_state is computable
there, i.e. both energy.txt and a thermal_properties.yaml entry at that T exist; E and G
therefore always come from the same subset of sets and cannot be averaged over
different samples.

S_config and the state free energy use CLC_workflow/src/thermo.py, the same corrected
implementation the plot script uses, so the two cannot drift apart:

    G_state = E_DFT + F_phonopy - T * S_config
    S_config = |a| R n_fu [2d ln 2d + (1-2d) ln(1-2d)]  with the sign that makes S >= 0

delta comes from nv/n_fu exactly, never from the rounded o#### label.
"""
import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from clc_workflow.clc_config import load_config, MANIFEST_NAME
from clc_workflow.thermo import (KJ_TO_EV, S_config_of_delta,
                                 infer_cell_stoichiometry)

try:
    from yaml import CLoader as _Loader
except Exception:
    from yaml import SafeLoader as _Loader


def read_energy_txt(p):
    try:
        s = open(p).read().strip()
        return float(s.split()[0]) if s else None
    except Exception:
        return None


def read_thermal(p, temps, tol=1e-6):
    """{T: {F, H_vib, S_vib}} in eV / eV/K from a phonopy thermal_properties.yaml."""
    out = {}
    try:
        with open(p) as f:
            data = yaml.load(f, Loader=_Loader)
        entries = data.get("thermal_properties", []) if isinstance(data, dict) else []
        for T in temps:
            for e in entries:
                if "temperature" not in e:
                    continue
                if abs(float(e["temperature"]) - float(T)) <= tol:
                    out[T] = {
                        "F_eV": float(e["free_energy"]) * KJ_TO_EV,
                        "H_vib_eV": float(e["energy"]) * KJ_TO_EV,
                        "S_vib_eV_per_K": float(e["entropy"]) * 1e-3 * KJ_TO_EV,
                    }
                    break
    except Exception:
        pass
    return out


def read_stability(p):
    try:
        lines = [l.strip() for l in open(p).read().strip().splitlines() if l.strip()]
        if not lines:
            return None, None, None
        nums = lines[0].split()
        A = float(nums[0]) if len(nums) > 0 else None
        B = float(nums[1]) if len(nums) > 1 else None
        return A, B, (lines[-1] if len(lines) > 1 else None)
    except Exception:
        return None, None, None


def _one(row, root=".", temps=(670, 1070), s_config_a=2.0, nv_tol=1):
    d = os.path.join(root, row["path"])
    nv, n_fu = int(row["nv"]), int(row["n_formula_units"])
    delta = nv / n_fu                      # exact; never from the o#### tag
    S_conf = S_config_of_delta(delta, n_fu, a_mag=s_config_a)

    # Does the cell on disk hold the composition the manifest ordered?  A relaxation
    # that collapsed, or a POSCAR truncated mid-write, still leaves a readable
    # energy.txt -- and an energy for the wrong cell is far more damaging to the mean
    # than a missing one, because nothing downstream can tell it apart.  Measured on a
    # real 50-set tree: one cell that had lost 142 of its 186 oxygens shifted its
    # point's mean by 21 eV and collapsed the DG fit to R2 = 0.02.
    nv_disk, n_fu_disk = None, None
    st_nfu, _st_nO, st_nv = infer_cell_stoichiometry(d)
    if st_nv is not None:
        nv_disk, n_fu_disk = int(st_nv), int(st_nfu)
    bad_cell = ""
    if nv_disk is not None and (abs(nv_disk - nv) > nv_tol or n_fu_disk != n_fu):
        bad_cell = (f"cell on disk is nv={nv_disk}/n_fu={n_fu_disk}, "
                    f"manifest says nv={nv}/n_fu={n_fu}")

    energy = read_energy_txt(os.path.join(d, "energy.txt"))
    therm = read_thermal(os.path.join(d, "thermal_properties.yaml"), temps)
    A, B, verdict = read_stability(os.path.join(d, "stability.txt"))

    opt = {}
    rp = os.path.join(d, "opt_result.json")
    if os.path.isfile(rp):
        try:
            opt = json.load(open(rp))
        except Exception:
            opt = {}

    a_md = b_md = c_md = None
    # POSCAR_md_final is stage 4's pre-averaging output; kept so older trees still report.
    for name in ("optimized_POSCAR", "POSCAR_md_avg", "POSCAR_md_final"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            try:
                from pymatgen.io.vasp import Poscar
                lat = Poscar.from_file(p).structure.lattice
                a_md, b_md, c_md = (float(v) for v in lat.abc)
            except Exception:
                pass
            break

    # group_dir/comp_dir identify the composition across sets.  Old manifests predate
    # the columns, so fall back to the path, which has always been set/group/comp/o.
    parts = str(row["path"]).replace("\\", "/").split("/")
    group_dir = row.get("group_dir") or (parts[1] if len(parts) > 2 else "")
    comp_dir = row.get("comp_dir") or (parts[2] if len(parts) > 3 else "")

    base = {
        "path": row["path"], "set": row["set"], "family": row["family"],
        "group_dir": group_dir, "comp_dir": comp_dir,
        "a_base": row["a_base"], "a_dopant": row["a_dopant"],
        "b_base": row["b_base"], "b_dopant": row["b_dopant"],
        "x": row["x"], "y": row["y"], "nv": nv, "delta": delta,
        "o_tag": row["o_tag"], "n_formula_units": n_fu,
        "n_atoms": row["n_atoms"], "n_O": row["n_O"],
        "S_config_eV_per_K": S_conf,
        "nv_on_disk": nv_disk, "cell_mismatch": bad_cell,
        "energy_eV": energy,
        "opt_converged": opt.get("converged"),
        "opt_energy_eV": opt.get("energy"),
        "opt_fmax_final": opt.get("fmax_final"),
        "opt_nsteps": opt.get("nsteps_taken"),
        "a_ang": a_md, "b_ang": b_md, "c_ang": c_md,
        "stability_A": A, "stability_B": B, "stability": verdict,
        "has_phonon": bool(therm),
    }

    rows = []
    for T in temps:
        th = therm.get(T, {})
        F = th.get("F_eV")
        H_vib = th.get("H_vib_eV")
        G_state = (energy + F - T * S_conf) if (energy is not None and F is not None
                                               and S_conf is not None) else None
        H_state = (energy + H_vib) if (energy is not None and H_vib is not None) else None

        # Why this set does or does not contribute to the average in results_avg.csv.
        # Only the first missing piece is named -- a directory with no energy.txt
        # usually has no thermal_properties.yaml either, and listing both says nothing
        # extra about what to rerun.
        if bad_cell:
            reason = "wrong cell"
        elif energy is None:
            reason = "no energy"
        elif not therm:
            reason = "no phonon"
        elif F is None:
            reason = f"no phonon at {T:g} K"
        elif S_conf is None:
            reason = "no S_config (n_fu unknown)"
        else:
            reason = ""

        rows.append({**base, "Temperature": T,
                     "F_phonopy_eV": F, "H_vib_eV": H_vib,
                     "S_vib_eV_per_K": th.get("S_vib_eV_per_K"),
                     "G_state_eV": G_state, "H_state_eV": H_state,
                     "ok": (reason == ""), "fail_reason": reason})
    return rows


# --------------------------------------------------------------- averaging over sets
#  Averaged over the sets that succeeded.  Columns that are a property of the
#  composition rather than of the realisation (nv, delta, n_fu, S_config) are carried
#  through unchanged; everything else is a mean with its sample std.
_AVG_COLS = ["energy_eV", "F_phonopy_eV", "H_vib_eV", "S_vib_eV_per_K",
             "G_state_eV", "H_state_eV", "a_ang", "b_ang", "c_ang"]

_CARRY_COLS = ["family", "group_dir", "comp_dir", "a_base", "a_dopant",
               "b_base", "b_dopant", "x", "y", "nv", "delta", "o_tag",
               "n_formula_units", "n_atoms", "n_O", "S_config_eV_per_K"]


def average_over_sets(df):
    """
    Average the independent SQS sets at fixed (composition, o_tag, temperature).

    The divisor is the number of sets that succeeded (n_sets_ok), never n_sets: a
    composition where 1 of 50 sets has no phonons averages over 49.  The failed sets
    are named in `failed_sets` so a thin average can be traced back to the directories
    that need rerunning.
    """
    if df.empty:
        return pd.DataFrame()

    keys = ["group_dir", "comp_dir", "o_tag", "Temperature"]
    out = []
    for kv, g in df.groupby(keys, dropna=False, sort=True):
        ok = g[g["ok"].fillna(False).astype(bool)]
        bad = g[~g["ok"].fillna(False).astype(bool)]
        row = dict(zip(keys, kv))

        src = ok if not ok.empty else g          # identity columns survive total failure
        first = src.iloc[0]
        for c in _CARRY_COLS:
            if c in g.columns:
                row[c] = first[c]

        row["n_sets_total"] = int(len(g))
        row["n_sets_ok"] = int(len(ok))
        row["n_sets_failed"] = int(len(bad))
        row["failed_sets"] = ";".join(sorted(str(s) for s in bad["set"].unique()))
        row["fail_reasons"] = ";".join(sorted(r for r in bad["fail_reason"].dropna().unique() if r))

        for c in _AVG_COLS:
            if c not in g.columns:
                continue
            v = pd.to_numeric(ok[c], errors="coerce").dropna()
            row[f"{c}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{c}_std"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            row[f"{c}_n"] = int(len(v))

        # spread of the realisations, the quantity that says whether 50 sets were needed
        gv = pd.to_numeric(ok["G_state_eV"], errors="coerce").dropna()
        row["G_state_spread_eV"] = float(gv.max() - gv.min()) if len(gv) > 1 else np.nan

        if "stability" in g.columns:
            verdicts = [str(s) for s in src["stability"].dropna().unique()]
            row["stability"] = ";".join(sorted(verdicts))
        out.append(row)

    res = pd.DataFrame(out)
    # keys and _CARRY_COLS overlap (comp_dir, o_tag, ...), so dedupe before reindexing --
    # a repeated name here makes res[cols] return a DataFrame per column, not a Series
    lead, seen = [], set()
    for c in (keys + _CARRY_COLS + ["n_sets_total", "n_sets_ok", "n_sets_failed",
                                    "failed_sets", "fail_reasons"]):
        if c in res.columns and c not in seen:
            lead.append(c); seen.add(c)
    return res[lead + [c for c in res.columns if c not in seen]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default=None, help="default <output_root>/results.csv")
    ap.add_argument("--avg-out", default=None,
                    help="default <output_root>/results_avg.csv (the per-set average)")
    ap.add_argument("--no-avg", action="store_true",
                    help="write only the per-structure results.csv")
    ap.add_argument("--nv-tol", type=int, default=1,
                    help="Mark a structure failed when the cell on disk differs from the "
                         "manifest by more than this many oxygen vacancies (default 1). "
                         "Use a huge value to accept every cell.")
    ap.add_argument("--nprocs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--temps", default=None, help="override analysis.temps, e.g. 670,1070")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = cfg["output_root"]
    man = pd.read_csv(os.path.join(root, MANIFEST_NAME))
    temps = ([float(t) for t in args.temps.split(",")] if args.temps
             else [float(t) for t in cfg["analysis"]["temps"]])
    a_mag = float(cfg["analysis"]["s_config_a"])

    print(f"[*] {len(man)} structures x {len(temps)} temperature(s); |a|={a_mag}")
    worker = partial(_one, root=root, temps=tuple(temps), s_config_a=a_mag,
                     nv_tol=args.nv_tol)
    records = [r.to_dict() for _, r in man.iterrows()]
    if args.nprocs > 1 and len(records) > 1:
        with Pool(args.nprocs) as pool:
            chunks = pool.map(worker, records, chunksize=8)
    else:
        chunks = [worker(r) for r in records]

    df = pd.DataFrame([row for chunk in chunks for row in chunk])
    dest = args.out or os.path.join(root, "results.csv")
    df.to_csv(dest, index=False)

    n_e = int(df["energy_eV"].notna().sum() / max(len(temps), 1))
    n_p = int(df["has_phonon"].sum() / max(len(temps), 1))
    n_g = int(df["G_state_eV"].notna().sum())
    print(f"[*] energies {n_e}/{len(man)}   phonons {n_p}/{len(man)}   "
          f"G_state rows {n_g}/{len(df)}")

    mism = df[df["cell_mismatch"].astype(bool)].drop_duplicates("path")
    if not mism.empty:
        print(f"[warn] {len(mism)} structure(s) hold the wrong cell and are excluded from "
              f"every average (their energy describes a different composition):")
        for _, r in mism.head(10).iterrows():
            print(f"      {r['path']}: {r['cell_mismatch']}")
        if len(mism) > 10:
            print(f"      ... and {len(mism) - 10} more (see cell_mismatch in {dest})")
    if df["stability"].notna().any():
        print("[*] stability: " + "  ".join(
            f"{k}={int(v)}" for k, v in
            df.drop_duplicates("path")["stability"].value_counts().items()))
    print(f"[*] results -> {dest}")

    if args.no_avg:
        return

    avg = average_over_sets(df)
    if avg.empty:
        print("[warn] nothing to average")
        return
    dest_avg = args.avg_out or os.path.join(root, "results_avg.csv")
    avg.to_csv(dest_avg, index=False)

    # A composition averaged over fewer sets than its neighbours carries more noise into
    # the ΔG fit than the row itself shows, so say how many and where.
    n_full = int((avg["n_sets_failed"] == 0).sum())
    n_thin = int(((avg["n_sets_failed"] > 0) & (avg["n_sets_ok"] > 0)).sum())
    n_empty = int((avg["n_sets_ok"] == 0).sum())
    print(f"[*] averaged {len(avg)} (composition, o_tag, T) group(s) over "
          f"{int(avg['n_sets_total'].max())} set(s): "
          f"{n_full} complete, {n_thin} averaged over fewer sets, {n_empty} with no data")
    if n_thin or n_empty:
        short = avg[avg["n_sets_failed"] > 0].sort_values("n_sets_ok")
        for _, r in short.head(10).iterrows():
            print(f"      {r['comp_dir']}/{r['o_tag']} T={r['Temperature']:g}K  "
                  f"n={int(r['n_sets_ok'])}/{int(r['n_sets_total'])}  "
                  f"missing {r['failed_sets']}  ({r['fail_reasons']})")
        if len(short) > 10:
            print(f"      ... and {len(short) - 10} more (see n_sets_ok in {dest_avg})")
    print(f"[*] set average -> {dest_avg}")


if __name__ == "__main__":
    main()
