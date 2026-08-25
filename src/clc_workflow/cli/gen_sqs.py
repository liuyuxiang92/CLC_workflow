#!/usr/bin/env python3
"""
Stage 1 -- generate SQS structures for every (set, family, x, y, nv) and write manifest.csv.

    clc sqs config.yaml [--dry-run] [--limit N] [--sets set_01,set_02]

Replaces run_workflow.py + execute_parrallel_mcsqs.py.  No source-code patching, no
taskset: one flat task list over a multiprocessing.Pool, so load balances properly and
N independent sets are just a config field.

Composition chain runs ONCE PER FAMILY with a single dopant per site, which returns the
(x,y) grid exactly (81 rows for 9x9, no duplicates) -- so filtering.filter_compositions
is not needed at all.  That matters for correctness as well as speed: its
element_combination mode flattens pairs into one allowed set per site, so asking for
(Sr,Ca)-(Fe,Mn) plus (Sr,Ba)-(Fe,Co) would also admit Ca-Co and Ba-Mn.
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from clc_workflow.clc_config import load_config, expand_families, set_names, MANIFEST_NAME
from clc_workflow.composition import generate_valid_combinations
from clc_workflow.neutrality import filter_charge_neutral_compositions
from clc_workflow.sqs_generator import (build_directory_name_for_family, delta_from_nv,
                                        generate_one_structure, integral_site_counts,
                                        poscar_atom_count, site_occupancies,
                                        supercell_site_counts, vacancy_fraction)
from clc_workflow.thermo import o_tag_for_delta

def family_compositions(cfg, fam):
    """(x, y) rows for one family, after the optional charge-neutrality screen."""
    df = generate_valid_combinations(
        formula_extra_part="O3",
        site_vars={"x": "A", "y": "B"},
        site_dopants={"A": {"base": fam.a_base, "dopants": [fam.a_dopant]},
                      "B": {"base": fam.b_base, "dopants": [fam.b_dopant]}},
        site_values={"x": fam.x_values, "y": fam.y_values},
        include_undoped=True,
        save_csv=False,
    )
    n_raw = len(df)
    if fam.screen:
        df = filter_charge_neutral_compositions(
            df, delta_range=tuple(fam.screen_delta_range), debug=False, save_csv=False)
    dup = df[["x", "y"]].duplicated().sum()
    if dup:
        raise RuntimeError(f"{fam.name}: {dup} duplicate (x,y) rows -- unexpected")
    return df.reset_index(drop=True), n_raw


def cation_site_counts(fam, x, y, n_A_sites, n_B_sites):
    """
    Integer occupancies of the two cation sublattices, or a complaint if there are none.

    Returns ((n_a_base, n_a_dop, n_a_vac), (n_b_base, n_b_dop, n_b_vac), problem) with
    problem None when both sublattices land exactly on whole sites.  They have to: mcsqs
    takes concentrations, so a composition the cell cannot hold is rounded to one it can,
    and the tree fills with a compound nobody asked for.  x*n_A must be an integer, and so
    must a_vac*n_A -- 5% of 64 sites is 3.2 atoms, which is why the off-stoichiometric
    list needs 80 A sites rather than 64.
    """
    a_fracs, b_fracs = site_occupancies(x, y, fam.a_vac, fam.b_vac)
    a_counts = integral_site_counts(a_fracs, n_A_sites)
    b_counts = integral_site_counts(b_fracs, n_B_sites)
    problem = None
    if a_counts is None or b_counts is None:
        side = "A" if a_counts is None else "B"
        fracs = a_fracs if a_counts is None else b_fracs
        n = n_A_sites if a_counts is None else n_B_sites
        problem = (f"{side} site: {[round(f, 6) for f in fracs]} x {n} sites = "
                   f"{[round(f * n, 3) for f in fracs]}, not whole atoms")
    return a_counts, b_counts, problem


def build_tasks(cfg):
    """Flat task list plus the manifest rows they will produce."""
    fams = expand_families(cfg)
    sets = set_names(cfg)
    supercell = cfg["sqs"]["supercell"]
    base = cfg["base_poscar"]
    n_A, n_B, n_O_sites, n_atoms, n_fu = supercell_site_counts(supercell, base)
    sq = cfg["sqs"]

    tasks, rows, summary, problems = [], [], [], []
    for fam in fams:
        comps, n_raw = family_compositions(cfg, fam)
        nvs = fam.nv_range(n_O_sites)
        summary.append({"family": fam.name, "a_dopant": fam.a_dopant,
                        "b_dopant": fam.b_dopant, "xy_raw": n_raw,
                        "xy_kept": len(comps), "n_nv": len(nvs),
                        "a_vac": fam.a_vac, "b_vac": fam.b_vac,
                        "per_set": len(comps) * len(nvs)})
        for set_name in sets:
            for _, row in comps.iterrows():
                x, y = float(row["x"]), float(row["y"])
                a_counts, b_counts, problem = cation_site_counts(fam, x, y, n_A, n_B)
                group, comp = build_directory_name_for_family(
                    fam.a_base, fam.a_dopant, x, fam.b_base, fam.b_dopant, y,
                    a_vac=fam.a_vac, b_vac=fam.b_vac)
                if problem and set_name == sets[0]:
                    problems.append(f"{group}/{comp}: {problem}")
                n_a_vac = a_counts[2] if a_counts else 0
                n_b_vac = b_counts[2] if b_counts else 0
                for nv in nvs:
                    delta = delta_from_nv(nv, n_O_sites)
                    o_tag = o_tag_for_delta(delta)
                    out_dir = os.path.join(cfg["output_root"], set_name, group, comp, o_tag)
                    tasks.append({
                        "out_dir": out_dir, "base_poscar": base,
                        "supercell": supercell,
                        "method": sq.get("method", "mcsqs"),
                        # nv is deliberately absent: one seed per (set, composition) is
                        # what nests the vacancy series, see generate_one_random
                        "seed_key": f"{set_name}/{group}/{comp}",
                        "random_seed": sq.get("random_seed", 0),
                        "a_base": fam.a_base, "a_dopant": fam.a_dopant, "x": x,
                        "b_base": fam.b_base, "b_dopant": fam.b_dopant, "y": y,
                        "a_vac": fam.a_vac, "b_vac": fam.b_vac,
                        "nv": nv, "cutoffs": sq["cutoffs"],
                        "timeout_sec": sq["timeout_sec"],
                        "clean_dir": sq["clean_dir"],
                        "skip_existing": sq["skip_existing"],
                        "max_attempts": sq.get("max_attempts", 2),
                    })
                    rows.append({
                        "path": os.path.relpath(out_dir, cfg["output_root"]),
                        "set": set_name, "family": fam.name,
                        "a_base": fam.a_base, "a_dopant": fam.a_dopant,
                        "b_base": fam.b_base, "b_dopant": fam.b_dopant,
                        "x": x, "y": y, "a_vac": fam.a_vac, "b_vac": fam.b_vac,
                        "nv": nv, "delta": delta, "o_tag": o_tag,
                        "vacancy_frac": vacancy_fraction(nv, n_O_sites),
                        # n_A / n_B / n_O are OCCUPIED cations and anions; the *_sites
                        # columns are the lattice they sit on.  The two differ only for a
                        # cation-deficient composition, and n_formula_units stays the
                        # number of A SITES, so delta = nv/n_fu and every per-f.u. energy
                        # keeps the same meaning it had before cation vacancies existed.
                        "n_atoms": n_atoms - nv - n_a_vac - n_b_vac,
                        "n_A": n_A - n_a_vac, "n_B": n_B - n_b_vac,
                        "n_A_sites": n_A, "n_B_sites": n_B,
                        "n_vac_A": n_a_vac, "n_vac_B": n_b_vac,
                        "n_O": n_O_sites - nv, "n_O_sites": n_O_sites,
                        "n_formula_units": n_fu,
                        "lammps_type_map": " ".join(fam.type_map),
                        "group_dir": group, "comp_dir": comp,
                    })
    if problems and cfg["sqs"].get("require_commensurate", True):
        listed = "\n".join(f"        {p}" for p in problems[:12])
        more = f"\n        ... and {len(problems) - 12} more" if len(problems) > 12 else ""
        raise SystemExit(
            f"[ERROR] {len(problems)} composition(s) cannot be realised exactly on a "
            f"{n_atoms}-site supercell\n"
            f"        ({n_A} A, {n_B} B, {n_O_sites} O from {supercell} of {base}):\n"
            f"{listed}{more}\n\n"
            f"        mcsqs is given concentrations, not counts, so it would round these\n"
            f"        silently and generate a different compound.  Either enlarge\n"
            f"        sqs.supercell until the fractions land on whole sites, or set\n"
            f"        sqs.require_commensurate: false to accept the rounding.")
    return tasks, pd.DataFrame(rows), summary


def _pin_initializer(pin):
    """Optional Linux CPU pinning; a no-op everywhere else."""
    if not pin or not hasattr(os, "sched_setaffinity"):
        return
    try:
        import multiprocessing as mp
        ident = mp.current_process()._identity
        idx = (ident[0] - 1) if ident else 0
        ncpu = os.cpu_count() or 1
        os.sched_setaffinity(0, {idx % ncpu})
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the task table and counts, generate nothing")
    ap.add_argument("--audit", action="store_true",
                    help="check every existing POSCAR's atom count against the supercell "
                         "and report; write nothing.  A 'stale' row is a POSCAR built "
                         "from a truncated bestsqs.out -- rerun without --audit to "
                         "rebuild it")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N tasks")
    ap.add_argument("--sets", default=None, help="comma list of set names to restrict to")
    ap.add_argument("--n-procs", type=int, default=None, help="override sqs.n_procs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tasks, manifest, summary = build_tasks(cfg)

    if args.sets:
        keep = {s.strip() for s in args.sets.split(",") if s.strip()}
        idx = [i for i, r in manifest.iterrows() if r["set"] in keep]
        tasks = [tasks[i] for i in idx]
        manifest = manifest.iloc[idx].reset_index(drop=True)

    n_A, n_B, n_O_sites, n_atoms, n_fu = supercell_site_counts(cfg["sqs"]["supercell"],
                                                               cfg["base_poscar"])
    method = str(cfg["sqs"].get("method", "mcsqs")).lower()
    print(f"[*] base cell      : {cfg['base_poscar']}")
    print(f"[*] supercell      : {cfg['sqs']['supercell']} -> {n_atoms} sites "
          f"({n_A} A, {n_B} B, {n_O_sites} O), {n_fu} formula units")
    print(f"[*] method         : {method}" + (
        f"  (seed {cfg['sqs'].get('random_seed', 0)}; vacancies nested within each "
        f"composition)" if method == "random" else
        f"  (mcsqs, timeout {cfg['sqs']['timeout_sec']}s per structure)"))
    print(f"[*] sets           : {', '.join(sorted(manifest['set'].unique()))}")
    print()
    print(f"{'family':<16} {'(x,y) raw':>10} {'kept':>6} {'nv':>4} "
          f"{'a_vac':>6} {'b_vac':>6} {'per set':>9}")
    for s in summary:
        print(f"{s['family']:<16} {s['xy_raw']:>10} {s['xy_kept']:>6} "
              f"{s['n_nv']:>4} {s['a_vac']:>6.4g} {s['b_vac']:>6.4g} {s['per_set']:>9}")
    per_set = sum(s["per_set"] for s in summary)
    print(f"{'TOTAL':<16} {'':>10} {'':>6} {'':>4} {'':>6} {'':>6} {per_set:>9}"
          f"   x {int(cfg['n_sets'])} sets = {per_set * int(cfg['n_sets'])} structures")
    print()

    if args.dry_run:
        show = manifest.head(8)[["path", "x", "y", "nv", "o_tag", "n_A", "n_B",
                                 "n_vac_A", "n_vac_B", "n_O", "n_atoms"]]
        print("first tasks:")
        print(show.to_string(index=False))
        print(f"\n[dry-run] {len(tasks)} tasks; nothing written.")
        return

    if args.limit:
        tasks, manifest = tasks[:args.limit], manifest.iloc[:args.limit].reset_index(drop=True)

    if args.audit:
        for t in tasks:
            t["audit"] = True

    os.makedirs(cfg["output_root"], exist_ok=True)
    n_procs = args.n_procs or int(cfg["sqs"]["n_procs"])
    verb = "auditing" if args.audit else "generating"
    print(f"[*] {verb} {len(tasks)} structures on {n_procs} process(es) ...")

    if n_procs > 1 and len(tasks) > 1:
        with Pool(processes=n_procs, initializer=_pin_initializer,
                  initargs=(cfg["sqs"]["pin_cpus"],)) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(generate_one_structure, tasks), 1):
                results.append(res)
                if i % 50 == 0 or i == len(tasks):
                    print(f"    {i}/{len(tasks)}")
    else:
        results = []
        for i, t in enumerate(tasks, 1):
            results.append(generate_one_structure(t))
            if i % 50 == 0 or i == len(tasks):
                print(f"    {i}/{len(tasks)}")

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n[*] " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for label, want in (("FAILED", "failed"), ("STALE", "stale")):
        bad = [r for r in results if r["status"] == want]
        for r in bad[:10]:
            print(f"    [{label}] {r['out_dir']}: {r['detail']}")
        if len(bad) > 10:
            print(f"    ... and {len(bad) - 10} more {label.lower()}")

    if args.audit:
        stale = sum(1 for r in results if r["status"] in ("stale", "failed"))
        print(f"\n[audit] {stale} structure(s) need rebuilding; nothing was written.\n"
              f"        Rerun without --audit to rebuild exactly those "
              f"(the rest are skipped).")
        return

    # The manifest keeps a row only if its POSCAR exists AND holds the atoms the row
    # claims.  Existence alone is not enough: a POSCAR built from a bestsqs.out that was
    # truncated in its atom block is a complete, readable file of the wrong composition,
    # and every later stage would take it at face value.  n_atoms is exactly the expected
    # count, so this also re-checks structures generated by earlier runs.
    def _usable(row):
        return poscar_atom_count(
            os.path.join(cfg["output_root"], row["path"], "POSCAR")) == int(row["n_atoms"])

    have = manifest.apply(_usable, axis=1)
    manifest_out = manifest[have].reset_index(drop=True)
    dest = os.path.join(cfg["output_root"], MANIFEST_NAME)
    manifest_out.to_csv(dest, index=False)
    dropped = len(manifest) - len(manifest_out)
    note = f"  ({dropped} dropped: missing or wrong atom count)" if dropped else ""
    print(f"[*] manifest: {len(manifest_out)}/{len(manifest)} rows -> {dest}{note}")


if __name__ == "__main__":
    main()
