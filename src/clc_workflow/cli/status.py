#!/usr/bin/env python3
"""
One status view over the whole tree.

    clc status config.yaml [--by set|family|both] [--pending STAGE] [--gaps]

Replaces collect_uncal.py, find_uncalculated_dir.py, check_stability.py,
get_only_unstable_info.py, collect_new_generated_unconverged_POSCAR.sh and
clean_only_unstable_yaml.sh -- all of which existed because there was no single notion of
"what is done".  Those files are left in place; this just makes them unnecessary.

--gaps additionally reports composition series with a hole in their nv sequence.  That
matters because a gap makes an adjacent pair span dnv > 1, and the analysis has to scale
mu accordingly (the old code always added a single 1/2 O2, an error of about 5.2 eV).
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from clc_workflow.clc_config import load_config, MANIFEST_NAME
# label -> (file that marks it done, file that must exist for it to be attemptable)
STAGES = [
    ("sqs",       "POSCAR",              None),
    ("md",        "traj.lammpstrj",      "conf.lmp"),
    ("md_avg",    "POSCAR_md_avg",       "traj.lammpstrj"),
    ("opt",       "opt_result.json",     "POSCAR_md_avg"),
    ("optimized", "optimized_POSCAR",    "opt_result.json"),
    ("phonon",    "stability.txt",       "optimized_POSCAR"),
]


def scan(cfg, man):
    root = cfg["output_root"]
    rows = []
    for _, r in man.iterrows():
        d = os.path.join(root, r["path"])
        rec = {"set": r["set"], "family": r["family"], "path": r["path"],
               "nv": r["nv"], "x": r["x"], "y": r["y"],
               "comp_dir": r.get("comp_dir", ""), "group_dir": r.get("group_dir", "")}
        for label, done, _need in STAGES:
            rec[label] = os.path.isfile(os.path.join(d, done))
        verdict = ""
        sp = os.path.join(d, "stability.txt")
        if rec["phonon"]:
            try:
                lines = open(sp).read().strip().splitlines()
                verdict = lines[-1].strip() if lines else ""
            except Exception:
                verdict = "unreadable"
        rec["verdict"] = verdict
        rows.append(rec)
    return pd.DataFrame(rows)


def report(df, keys):
    labels = [s[0] for s in STAGES]
    for key, block in df.groupby(keys, sort=True):
        head = " / ".join(key) if isinstance(key, tuple) else str(key)
        n = len(block)
        parts = []
        prev = n
        for label, _done, need in STAGES:
            got = int(block[label].sum())
            denom = prev if need else n
            parts.append(f"{label} {got}/{denom}")
            prev = got
        print(f"  {head:<34} " + "  ".join(parts))
        v = block.loc[block["phonon"], "verdict"].value_counts()
        if len(v):
            print(f"  {'':<34} " + "  ".join(f"{k}={int(c)}" for k, c in v.items()))


def find_gaps(df):
    """Composition series whose nv sequence has a hole (breaks adjacent-pair mu)."""
    out = []
    for (s, comp), block in df.groupby(["set", "comp_dir"], sort=True):
        have = sorted(int(v) for v in block.loc[block["phonon"], "nv"])
        if len(have) < 2:
            continue
        missing = [v for v in range(have[0], have[-1] + 1) if v not in set(have)]
        if missing:
            out.append({"set": s, "comp_dir": comp, "n_have": len(have),
                        "nv_range": f"{have[0]}..{have[-1]}",
                        "missing_nv": ",".join(str(v) for v in missing[:12])})
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--by", choices=["set", "family", "both"], default="both")
    ap.add_argument("--pending", choices=[s[0] for s in STAGES], default=None,
                    help="print paths still needing this stage, one per line")
    ap.add_argument("--gaps", action="store_true",
                    help="list composition series with a hole in their nv sequence")
    ap.add_argument("--csv", default=None, help="write the per-structure table here")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    man_path = os.path.join(cfg["output_root"], MANIFEST_NAME)
    if not os.path.isfile(man_path):
        sys.exit(f"{man_path} not found -- run `clc sqs` first")
    df = scan(cfg, pd.read_csv(man_path))

    if args.pending:
        label = args.pending
        # rows lacking this stage's output but holding its prerequisite
        sel = df[~df[label]]
        prereq = _label_for(dict((s[0], s[2]) for s in STAGES)[label])
        if prereq:
            sel = sel[sel[prereq]]
        for p in sel["path"]:
            print(p)
        return

    print(f"[*] {len(df)} structures in {cfg['output_root']}")
    keys = {"set": ["set"], "family": ["family"], "both": ["set", "family"]}[args.by]
    report(df, keys)

    if args.gaps:
        gaps = find_gaps(df)
        print()
        if gaps.empty:
            print("[*] no gaps: every completed composition series is contiguous in nv")
        else:
            print(f"[warn] {len(gaps)} composition series have nv gaps "
                  f"(adjacent pairs there span dnv > 1):")
            print(gaps.head(20).to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n[*] per-structure table -> {args.csv}")


def _label_for(done_file):
    for label, done, _ in STAGES:
        if done == done_file:
            return label
    return None


if __name__ == "__main__":
    main()
