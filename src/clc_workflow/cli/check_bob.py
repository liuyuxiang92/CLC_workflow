#!/usr/bin/env python3
"""
Stage 6.5 -- B-O-B angle distributions of POSCAR_md_avg vs optimized_POSCAR.

    clc check-bob config.yaml [--limit N] [--nprocs N] [--plot] [--dump-angles]
    clc check-bob config.yaml --plot          -> bob_angles.png in each structure dir
    clc check-bob --dirs set_01/SrCa_FeMn/x0.25_y0.50/o0008 ...   (no config)

A structural check between stages 6 and 7.  Every bridging O gets the angle subtended
by its two nearest B cations: 180 deg is the untilted cubic perovskite, and the tilt
(180 - mean) is what the relaxation is allowed to change.  For each structure this
reports the count of bridging O, the mean, sigma, min and max of the distribution, and
the same for the relaxed structure -- so a relaxation that quietly destroyed the
octahedral network is visible here rather than three GPU-hours later in the phonons.

Read the columns as:

  *_n        bridging O actually used.  Should equal n_O; a deficit means some O has
             fewer than two B inside the 2.8 A cutoff -- broken octahedra.
  *_mean     mean B-O-B angle; *_tilt = 180 - mean is the octahedral tilt.
  *_std      width of the distribution.  In a mean MD structure this is static
             disorder (SQS site disorder + the frozen-in tilt pattern), NOT thermal
             libration: the libration has already been averaged out by stage 4.
  *_min/max  the tails.  A min far below the bulk of the distribution is one broken
             bridge, not a tilted cell.
  d_mean     opt - md.  Small and negative is the expected story (relaxing at 0 K
             deepens the tilt slightly); tens of degrees is a failed relaxation.

The angles themselves are recomputed here rather than taken from the MD trajectory, so
this says nothing about the instantaneous-vs-mean-structure difference -- for that see
code/md/SrCaFeCoO/feofe_windows.py, which is where the +2.4 deg libration figure for
these materials comes from.
"""
import argparse
import os
import re
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from clc_workflow.clc_config import load_config, MANIFEST_NAME
from clc_workflow.bob_angles import analyze_poscar, angle_stats, RCUT_DEFAULT
# (column prefix, label, candidate file names in order of preference).  POSCAR_md_final
# is the pre-averaging name stage 4 no longer writes but older trees still hold;
# unconverged_POSCAR is only picked up with --include-unconverged, because comparing a
# relaxation that never converged against the MD mean answers a different question.
SOURCES = [("md",  "POSCAR_md_avg",    ["POSCAR_md_avg", "POSCAR_md_final"]),
           ("opt", "optimized_POSCAR", ["optimized_POSCAR"])]
UNCONVERGED = "unconverged_POSCAR"

MIN_ANGLE_WARN = 120.0   # below this a bridge is bent past anything a perovskite does
DELTA_WARN = 5.0         # |opt - md| mean shift worth listing structure by structure
PAIRS_SHOWN = 6          # cation-pair rows in the pooled report
PLOT_NAME = "bob_angles.png"     # per structure, beside its POSCARs
POOLED_PLOT_NAME = "bob_check.png"


def _resolve(d, names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.isfile(p):
            return p
    return None


def _legend_label(prefix, path):
    """
    Legend text for one source.  `clc collect-md` writes the averaging window into the
    POSCAR comment line ("... mean of 180.0-200.0 ps, 100 frames, 0 hoppers"), so the
    figure can name the window it is showing instead of asserting config.yaml's value,
    which may not be what this particular file was built with.
    """
    base = PLOT_STYLE[prefix][1]
    if prefix != "md":
        return base
    try:
        with open(path) as fh:
            m = re.search(r"mean of ([\d.]+)-([\d.]+) ps", fh.readline())
        if m:
            return f"{base} ({float(m.group(1)):g}-{float(m.group(2)):g} ps)"
        return base + " (last frame)" if "md_final" in os.path.basename(path) else base
    except Exception:
        return base


PLOT_STYLE = {"md":  ("#2b6cb0", "MD average"),
              "opt": ("#c53030", "0 K optimized")}


def _plot_structure(rel, bsite, angles, labels, dest, rcut):
    """
    One figure per structure directory -- the two POSCARs overlaid on one axis.

    Pooling the whole tree into a single histogram hides what this check is for: each
    cell has its own tilt pattern, so a structure that went wrong is its own figure,
    not a bump in a global distribution.  Density (not counts) on the y-axis, because
    a vacancy-bearing cell has fewer bridging O than its neighbours and raw counts
    would make the two curves incomparable for no physical reason.  The KDE is a
    reading aid over the histogram, at a fixed bandwidth so its width is comparable
    between structures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [p for p, _label, _n in SOURCES if p in angles and angles[p][0].size]
    if not present:
        return None
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.axisbelow": True})

    allang = np.concatenate([angles[p][0] for p in present]).astype(float)
    lo, hi = float(allang.min()), float(allang.max())
    pad = max(2.0, 0.08 * (hi - lo))
    bins = np.linspace(np.floor(lo), np.ceil(hi), 22)
    grid = np.linspace(lo - pad, hi + pad, 400)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ymax = 0.0
    for p in present:
        a = angles[p][0].astype(float)
        color, base = PLOT_STYLE[p]
        h, _, _ = ax.hist(a, bins=bins, density=True, color=color, alpha=0.32,
                          edgecolor=color, lw=0.8)
        ymax = max(ymax, float(h.max()))
        try:
            from scipy.stats import gaussian_kde
            ax.plot(grid, gaussian_kde(a, bw_method=0.30)(grid), color=color, lw=2.4,
                    label=f"{labels.get(p, base)}\n"
                          f"$\\langle\\theta\\rangle$ = {a.mean():.2f} $\\pm$ "
                          f"{a.std():.2f}$^\\circ$   (N = {a.size})")
        except Exception:
            # a delta-like distribution (every angle equal) has a singular covariance
            ax.plot([], [], color=color, lw=2.4,
                    label=f"{labels.get(p, base)}\n"
                          f"$\\langle\\theta\\rangle$ = {a.mean():.2f} $\\pm$ "
                          f"{a.std():.2f}$^\\circ$   (N = {a.size})")
        ax.axvline(a.mean(), color=color, ls="--", lw=1.6)

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(0, ymax * 1.30)          # headroom so the legend clears the tallest bar
    ax.set_xlabel(r"B$-$O$-$B angle $\theta$ (deg)")
    ax.set_ylabel("probability density")
    ax.set_title(f"B$-$O$-$B angle distribution (B = {', '.join(bsite)})", fontsize=13,
                 pad=18)
    # provenance line under the title -- which structure, and the cutoff the neighbour
    # list used, since changing it changes N
    ax.text(0.0, 1.012, f"{rel}   B$-$O cutoff {rcut} $\\AA$", transform=ax.transAxes,
            fontsize=8, color="0.35", va="bottom")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(dest, dpi=180)
    plt.close(fig)
    return dest


def _one(job, rcut=RCUT_DEFAULT, dump=False, plot=False, keep_angles=True):
    d, rel, bsite = job["dir"], job["rel"], job["bsite"]
    rec = {"rel": rel, "bsite": "+".join(bsite)}
    angles, labels = {}, {}
    for prefix, _label, names in SOURCES:
        path = _resolve(d, names)
        if path is None:
            rec[prefix + "_file"] = ""
            rec.update(angle_stats(np.zeros(0), prefix + "_"))
            continue
        rec[prefix + "_file"] = os.path.basename(path)
        try:
            R = analyze_poscar(path, bsite, rcut=rcut)
        except Exception as e:
            rec[prefix + "_file"] = os.path.basename(path)
            rec[prefix + "_error"] = str(e)
            rec.update(angle_stats(np.zeros(0), prefix + "_"))
            continue
        a = R["angles"]
        rec.update(angle_stats(a, prefix + "_"))
        rec[prefix + "_n_O"] = R["n_O"]
        rec[prefix + "_n_B"] = R["n_B"]
        rec[prefix + "_n_underco"] = R["n_underco"]
        rec[prefix + "_n_overco"] = R["n_overco"]
        rec[prefix + "_bo_mean"] = float(R["bridge_blen"].mean()) if R["bridge_blen"].size else float("nan")
        if dump and a.size:
            np.savetxt(os.path.join(d, f"bob_angles_{os.path.basename(path)}.txt"),
                       np.c_[a], fmt="%.4f",
                       header=f"B-O-B angle (deg), rcut {rcut} A, B = {'+'.join(bsite)}")
        if keep_angles:
            angles[prefix] = (a.astype(np.float32), R["pairs"], R["axis"])
        labels[prefix] = _legend_label(prefix, path)

    have = [p for p, _l, _n in SOURCES if rec.get(p + "_n", 0) > 0]
    rec["status"] = ("ok" if len(have) == 2 else
                     "one_only" if len(have) == 1 else
                     "missing" if not any(rec.get(p + "_file") for p, _l, _n in SOURCES)
                     else "failed")
    if len(have) == 2:
        rec["d_mean"] = rec["opt_mean"] - rec["md_mean"]
        rec["d_std"] = rec["opt_std"] - rec["md_std"]
        rec["d_n"] = rec["opt_n"] - rec["md_n"]
    if plot and have:
        # written beside the POSCARs it describes, and drawn here in the worker so the
        # figures come out on the same processes that did the angles
        try:
            rec["figure"] = _plot_structure(rel, bsite, angles, labels,
                                            os.path.join(d, PLOT_NAME), rcut)
        except Exception as e:
            rec["plot_error"] = str(e)
    return rec, angles


def _pool_report(name, label, angles, pairs, axes_):
    """Pooled distribution over every structure, the tree-wide version of the figure."""
    if angles.size == 0:
        print(f"[*] {label:<17}: no angles")
        return
    s = angle_stats(angles)
    print(f"[*] {label:<17}: N={s['n']}  mean={s['mean']:.2f}  sigma={s['std']:.2f}  "
          f"min={s['min']:.2f}  max={s['max']:.2f}  median={s['median']:.2f}  "
          f"tilt={s['tilt']:.2f} deg")
    print(f"    {'':<17}  5-95%: {s['p05']:.1f}-{s['p95']:.1f}   "
          f"frac<160: {s['p_lt_160']:.3f}   frac<150: {s['p_lt_150']:.3f}")
    uniq, counts = np.unique(pairs, return_counts=True)
    for p in uniq[np.argsort(-counts)][:PAIRS_SHOWN]:
        a = angles[pairs == p]
        print(f"      {p:<9} N={a.size:<7d} mean={a.mean():7.2f}  sigma={a.std():5.2f}")
    for k in "xyz":
        a = angles[axes_ == k]
        if a.size:
            print(f"      bridge||{k}  N={a.size:<7d} mean={a.mean():7.2f}  sigma={a.std():5.2f}")


def _plot_pooled(pooled, df, dest):
    """
    Optional tree-wide overview (--plot-pooled), NOT the default.

    Pooling every structure's angles into one histogram is only meaningful as a
    population summary -- it cannot show which cell is wrong, which is why --plot
    draws one figure per structure instead.  Panel (a) is that population; (b) and
    (c) are per structure, which is where an outlier is actually identifiable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COL = {"md": "#2b6cb0", "opt": "#c53030"}
    LBL = {"md": "POSCAR_md_avg", "opt": "optimized_POSCAR"}
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.axisbelow": True})
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axs[0]
    both = np.concatenate([pooled[p][0] for p in ("md", "opt") if pooled[p][0].size])
    bins = np.linspace(np.floor(both.min()), 180.0, 60) if both.size else np.linspace(150, 180, 60)
    for p in ("md", "opt"):
        a = pooled[p][0]
        if not a.size:
            continue
        ax.hist(a, bins=bins, density=True, color=COL[p], alpha=0.35, edgecolor=COL[p],
                lw=0.8, label=f"{LBL[p]}\nN={a.size}, {a.mean():.2f}$\\pm${a.std():.2f}$^\\circ$, "
                              f"[{a.min():.1f}, {a.max():.1f}]")
        ax.axvline(a.mean(), color=COL[p], ls="--", lw=1.2)
    ax.axvline(180, color="0.35", ls=":", lw=1.0)
    ax.set_xlabel("B$-$O$-$B angle (deg)")
    ax.set_ylabel("probability density")
    ax.set_title("(a)  pooled over all structures", loc="left", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")

    ax = axs[1]
    ok = df[df["status"] == "ok"]
    ax.plot(ok["md_mean"], ok["opt_mean"], "o", ms=4, color="#2b6cb0", mec="w", mew=0.5,
            alpha=0.75)
    if len(ok):
        lim = [min(ok["md_mean"].min(), ok["opt_mean"].min()) - 1,
               max(ok["md_mean"].max(), ok["opt_mean"].max()) + 1]
        ax.plot(lim, lim, "-", color="0.45", lw=1.2, label="no change")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("mean angle, MD average (deg)")
    ax.set_ylabel("mean angle, optimized (deg)")
    ax.set_title(f"(b)  per structure (N = {len(ok)})", loc="left", fontweight="bold")

    ax = axs[2]
    d = ok["d_mean"].to_numpy() if len(ok) else np.zeros(0)
    if d.size:
        ax.hist(d, bins=40, color="#2b6cb0", alpha=0.55, edgecolor="#2b6cb0", lw=0.8)
        ax.axvline(float(np.mean(d)), color="#c53030", lw=1.8,
                   label=f"mean {np.mean(d):+.2f}$^\\circ$, sd {np.std(d):.2f}$^\\circ$\n"
                         f"range [{d.min():+.2f}, {d.max():+.2f}]")
        ax.legend(fontsize=8)
    ax.axvline(0, color="0.35", ls=":", lw=1.0)
    ax.set_xlabel("mean angle shift, optimized $-$ MD average (deg)")
    ax.set_ylabel("structures")
    ax.set_title("(c)  what the relaxation moved", loc="left", fontweight="bold")

    fig.tight_layout()
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    print(f"[*] figure  -> {dest}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="config.yaml (omit only with --dirs)")
    ap.add_argument("--dirs", nargs="+", default=None,
                    help="check these structure directories instead of the manifest; "
                         "the B site is then taken from --bsite")
    ap.add_argument("--bsite", nargs="+", default=None,
                    help="B-site elements, e.g. --bsite Fe Mn.  Default: b_base and "
                         "b_dopant from the manifest (required with --dirs)")
    ap.add_argument("--rcut", type=float, default=RCUT_DEFAULT,
                    help=f"B-O first-shell cutoff in A (default {RCUT_DEFAULT})")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-unconverged", action="store_true",
                    help="also accept unconverged_POSCAR as the relaxed structure")
    ap.add_argument("--dump-angles", action="store_true",
                    help="write bob_angles_<file>.txt in each structure directory")
    ap.add_argument("--plot", action="store_true",
                    help=f"write {PLOT_NAME} in each structure directory: one panel per "
                         f"POSCAR, that structure's own distribution")
    ap.add_argument("--plot-pooled", action="store_true",
                    help=f"additionally write {POOLED_PLOT_NAME} at the root -- every "
                         f"structure's angles in one histogram, a population summary only")
    ap.add_argument("--csv", default=None, help="summary path (default <root>/bob_summary.csv)")
    ap.add_argument("--nprocs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args(argv)

    if args.include_unconverged:
        SOURCES[1][2].append(UNCONVERGED)

    if args.dirs:
        if not args.bsite:
            sys.exit("[ERROR] --dirs needs --bsite (e.g. --bsite Fe Mn): without the "
                     "manifest there is nothing to say which element is the B site")
        root = os.path.commonpath([os.path.abspath(d) for d in args.dirs]) \
            if len(args.dirs) > 1 else os.path.abspath(args.dirs[0])
        jobs = [{"dir": d, "rel": d, "bsite": tuple(args.bsite)} for d in args.dirs]
    else:
        if not args.config:
            sys.exit("[ERROR] give a config.yaml, or --dirs D [D ...] --bsite Fe Mn")
        cfg = load_config(args.config)
        root = cfg["output_root"]
        man_path = os.path.join(root, MANIFEST_NAME)
        if not os.path.isfile(man_path):
            sys.exit(f"{man_path} not found -- run `clc sqs` first")
        man = pd.read_csv(man_path)
        if args.limit:
            man = man.head(args.limit)
        jobs = []
        for _, r in man.iterrows():
            bsite = tuple(args.bsite) if args.bsite else tuple(
                dict.fromkeys([str(r["b_base"]), str(r["b_dopant"])]))
            jobs.append({"dir": os.path.join(root, r["path"]), "rel": r["path"],
                         "bsite": bsite})

    print(f"[*] {len(jobs)} structures, B-O cutoff {args.rcut} A")
    worker = partial(_one, rcut=args.rcut, dump=args.dump_angles, plot=args.plot)
    if args.nprocs > 1 and len(jobs) > 1:
        with Pool(args.nprocs) as pool:
            out = pool.map(worker, jobs, chunksize=4)
    else:
        out = [worker(j) for j in jobs]
    rows = [r for r, _a in out]

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("[*] " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    df = pd.DataFrame(rows)
    dest = args.csv or os.path.join(root, "bob_summary.csv")
    df.to_csv(dest, index=False)

    pooled = {}
    for prefix, label, _n in SOURCES:
        parts = [a[prefix] for _r, a in out if prefix in a and a[prefix][0].size]
        pooled[prefix] = (np.concatenate([p[0] for p in parts]) if parts else np.zeros(0),
                          np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, "<U8"),
                          np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, "<U1"))
        _pool_report(prefix, label, *pooled[prefix])

    ok = df[df["status"] == "ok"] if "status" in df else df.iloc[0:0]
    if len(ok):
        print(f"[*] per-structure mean angle: md {ok['md_mean'].mean():.2f} "
              f"({ok['md_mean'].min():.2f}-{ok['md_mean'].max():.2f}), "
              f"opt {ok['opt_mean'].mean():.2f} "
              f"({ok['opt_mean'].min():.2f}-{ok['opt_mean'].max():.2f})")
        d = ok["d_mean"]
        print(f"[*] shift opt-md over {len(ok)} paired structures: mean {d.mean():+.2f} "
              f"sigma {d.std(ddof=0):.2f}  range [{d.min():+.2f}, {d.max():+.2f}] deg")

        big = ok[d.abs() > DELTA_WARN]
        if len(big):
            # The relaxation is supposed to deepen the tilt by a degree or two.  A large
            # shift means the mean structure it started from was not a perovskite the
            # optimiser recognised -- usually a stage-4 average over a window with hops.
            print(f"[warn] {len(big)} structure(s) moved more than {DELTA_WARN:g} deg:")
            for r in big.reindex(big["d_mean"].abs().sort_values(ascending=False).index) \
                       .head(8).itertuples():
                print(f"    {r.rel}: {r.md_mean:.2f} -> {r.opt_mean:.2f} "
                      f"({r.d_mean:+.2f} deg)")

    for prefix, label, _n in SOURCES:
        col = prefix + "_n_underco"
        if col not in df:
            continue
        bad = df[df[col].fillna(0) > 0]
        if len(bad):
            # Every O in an intact perovskite bridges two B, vacancies included -- a
            # vacancy removes the O, it does not strand one.  So this is a broken
            # network, or a cell so expanded that 2.8 A no longer reaches the B site.
            print(f"[warn] {label}: {len(bad)} structure(s) have O with fewer than two B "
                  f"inside {args.rcut} A (max {int(bad[col].max())} such O)")
            for r in bad.nlargest(min(5, len(bad)), col).itertuples():
                print(f"    {r.rel}: {int(getattr(r, col))} of "
                      f"{int(getattr(r, prefix + '_n_O'))} O")
        mincol = prefix + "_min"
        bent = df[df[mincol].fillna(180.0) < MIN_ANGLE_WARN]
        if len(bent):
            print(f"[warn] {label}: {len(bent)} structure(s) have a bridge below "
                  f"{MIN_ANGLE_WARN:g} deg (min {bent[mincol].min():.1f} deg)")
            for r in bent.nsmallest(min(5, len(bent)), mincol).itertuples():
                print(f"    {r.rel}: min {getattr(r, mincol):.1f} deg")

    errs = [r for r in rows if any(k.endswith("_error") for k in r)]
    for r in errs[:5]:
        for k in r:
            if k.endswith("_error"):
                print(f"    [FAILED] {r['rel']} {k[:-6]}: {r[k]}")

    print(f"[*] summary -> {dest}")
    if args.plot:
        n_fig = sum(1 for r in rows if r.get("figure"))
        print(f"[*] {n_fig} per-structure figure(s) -> <structure dir>/{PLOT_NAME}")
        for r in [x for x in rows if x.get("plot_error")][:5]:
            print(f"    [plot FAILED] {r['rel']}: {r['plot_error']}")
    if args.plot_pooled:
        _plot_pooled(pooled, df, os.path.join(root, POOLED_PLOT_NAME))


if __name__ == "__main__":
    main()
