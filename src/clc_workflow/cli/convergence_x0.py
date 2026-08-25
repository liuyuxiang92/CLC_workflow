#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 9b -- how stable is the ΔG=0 crossing?

Two convergence sweeps over the SAME chain stage 9 uses (average across sets ->
adjacent-pair ΔG -> OLS ΔG ~ x -> solve ΔG=0 at each pO2), replayed from
state_series_raw.csv so nothing is re-read from the trees:

  A) n_sets sweep.  Redo everything with only n of the sets, n = 1 .. N.  Two
     curves per quantity: the deterministic "first n sets" (set_01..set_n) and
     the mean +- 1σ over RESAMPLES random n-subsets, which is the one that says
     whether the answer has stopped moving.

  B) n_v sweep.  With ALL sets, restrict the regression to the pairs whose
     vacancy count is n_v <= n_max, n_max = 2 .. 32.  x = n_v/64 as everywhere
     else, so this is "fit only the first n_max points of the ΔG line".

Reported for every step: x_1atm, x_0p8atm, x_3e-05atm and
x_swing = x_3e-05atm - x_0p8atm, at each temperature.

--s-config-a re-derives G_state on another S_config branch (-2 physical, +2 as
printed, 0 dropped) before either sweep, so a convergence study is not stuck with
whichever branch happened to write the CSV.  E and F carry no S_config, so this
is exact rather than an approximation.

  clc convergence --raw <...>/state_series_raw.csv --outdir <...>
  clc convergence --raw <...>/state_series_raw.csv --s-config-a 2 ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clc_workflow.cli import plot_heatmap as S

# the three pressures the summary table reports, and the pair whose difference
# is the usable swing (paper: 0.00003 vs 0.8 atm)
PRESSURES = [1.0, 0.8, 0.00003]
P_LO, P_HI = 0.00003, 0.8

QUANTS = ["x_1atm", "x_0p8atm", "x_3e-05atm", "x_swing"]
QLABEL = {
    "x_1atm":     r"$x$ @ 1 atm",
    "x_0p8atm":   r"$x$ @ 0.8 atm",
    "x_3e-05atm": r"$x$ @ 3$\times10^{-5}$ atm",
    "x_swing":    r"$x_{\rm swing}$ (3e-5 $-$ 0.8 atm)",
}


def reject_mislabelled_cells(df_state, tol=1):
    """
    Apply stage 9's nv_tag_tol guard to a raw CSV that predates it.

    A cell whose own vacancy count contradicts the o#### it sits in by more than
    `tol` is not the structure the directory claims, so its energy describes
    nothing here.  The o-tag is a rounded label, hence the tolerance -- but a
    64-f.u. cell recorded with nv = 148 is a corrupt file, and at ~1000 eV off
    the mean it does not widen an error bar, it destroys the fit it enters.

    Returns (cleaned frame, list of (series, o, nv, nv_tag)).
    """
    nv = pd.to_numeric(df_state["nv"], errors="coerce")
    n_fu = pd.to_numeric(df_state["n_fu"], errors="coerce")
    tag = pd.to_numeric(df_state["delta_from_o_tag"], errors="coerce")
    nv_tag = (tag * n_fu).round()
    bad = (nv - nv_tag).abs() > tol
    bad &= nv.notna() & nv_tag.notna()
    if not bad.any():
        return df_state, []
    listed = (df_state.loc[bad, ["series", "o", "nv"]]
              .assign(nv_tag=nv_tag[bad]).drop_duplicates()
              .itertuples(index=False, name=None))
    return df_state[~bad].copy(), list(listed)


# ---------- one pass of the stage-9 chain ----------

def run_chain(df_state, series_subset, muG, muH, temps, comp_info,
              nv_max=None, avg_mode="simple"):
    """
    Average -> pairs -> ΔG=0 vs pO2 -> the headline row, for one choice of sets
    and one fit window.  Returns {(T, quantity): value} plus fit diagnostics.

    `series_subset` is the universe of sets: it is both the filter and the
    n_series_total the averaging reports, so "43 of 50 usable" stays honest when
    the universe is smaller than the tree.
    """
    d = df_state[df_state["series"].isin(series_subset)]
    if d.empty:
        return {}

    bavg = S.average_across_series(d, avg_mode, all_series=sorted(series_subset))
    pairs = S.build_pair_deltas(bavg, muG, muH, strict_mu=True, want_EH=False)
    if pairs.empty:
        return {}

    pairs["family"] = pairs["comp_dir"].map(lambda c: comp_info[c]["family"])
    pairs["A_frac"] = pairs["comp_dir"].map(lambda c: comp_info[c]["A_frac"])
    pairs["B_frac"] = pairs["comp_dir"].map(lambda c: comp_info[c]["B_frac"])

    # the fit window: keep only pairs whose vacancy count is within n_v <= nv_max.
    # nv_lo is the vacancy count of the deficient member, i.e. exactly the x*n_fu
    # the ΔG line is plotted against.
    if nv_max is not None:
        nv_lo = pd.to_numeric(pairs["nv_lo"], errors="coerce")
        pairs = pairs[nv_lo <= nv_max]
        if pairs.empty:
            return {}

    thr = S.compute_pO2_thresholds(pairs, temps, PRESSURES,
                                   variant_label="corrected_a_neg",
                                   a_signed=-2.0, gcol="ΔG_bavg",
                                   x_col="x_pair", fit_hi_min=None)
    if thr.empty:
        return {}
    summ = S.build_simple_summary(thr, PRESSURES, "corrected_a_neg", comp_info)
    if summ.empty:
        return {}

    out = {}
    for _, r in summ.iterrows():
        T = int(r["Temperature_K"])
        for q in QUANTS:
            out[(T, q)] = float(r[q]) if pd.notna(r.get(q)) else np.nan
        out[(T, "R2")] = float(r["R2"]) if pd.notna(r["R2"]) else np.nan
        out[(T, "n_sets_used")] = r.get("n_sets_used", np.nan)

    # Diagnostics the headline table deliberately hides.  build_simple_summary floors a
    # negative crossing at 0, which is right for reporting ("no oxygen is released") but
    # wrong for reading a convergence curve: x0 = -b/m is a RATIO, so wherever the fitted
    # slope m passes through zero x0 runs off to ±∞ and comes back with the other sign.
    # Carrying the unfloored x0 and m makes that pole visible instead of flat zeros.
    for T in temps:
        sub = thr[thr["Temperature"] == T]
        out[(T, "n_points")] = float(sub["n_points"].max()) if not sub.empty else np.nan
        for p in PRESSURES:
            rp = sub[np.isclose(sub["pO2_atm"], p)]
            tag = S.fmt_pressure(p)
            out[(T, f"x0raw_{tag}")] = float(rp["x0"].iloc[0]) if not rp.empty else np.nan
            out[(T, f"x0std_{tag}")] = float(rp["x0_std"].iloc[0]) if not rp.empty else np.nan
        r1 = sub[np.isclose(sub["pO2_atm"], 1.0)]
        out[(T, "m")] = float(r1["m"].iloc[0]) if not r1.empty else np.nan
        out[(T, "se_m")] = float(r1["se_m"].iloc[0]) if not r1.empty else np.nan
        out[(T, "b")] = float(r1["b"].iloc[0]) if not r1.empty else np.nan
    return out


# ---------- sweeps ----------

def sweep_n_sets(df_state, all_series, muG, muH, temps, comp_info,
                 resamples=40, seed=0):
    """n = 1..N sets: deterministic prefix + random-subset mean/std."""
    rng = np.random.default_rng(seed)
    N = len(all_series)
    rows = []
    for n in range(1, N + 1):
        # deterministic: the first n sets in name order
        det = run_chain(df_state, all_series[:n], muG, muH, temps, comp_info)

        # random n-subsets (only one distinct subset exists at n = N)
        reps = 1 if n == N else resamples
        draws = []
        for _ in range(reps):
            sub = list(rng.choice(all_series, size=n, replace=False)) if n < N else all_series
            draws.append(run_chain(df_state, sub, muG, muH, temps, comp_info))

        for T in temps:
            row = {"n_sets": n, "Temperature": T}
            for q in QUANTS:
                row[f"{q}_first"] = det.get((T, q), np.nan)
                vals = np.array([dd.get((T, q), np.nan) for dd in draws], dtype=float)
                vals = vals[np.isfinite(vals)]
                row[f"{q}_mean"] = vals.mean() if vals.size else np.nan
                row[f"{q}_std"] = vals.std(ddof=1) if vals.size > 1 else 0.0
            row["R2_first"] = det.get((T, "R2"), np.nan)
            row["n_points_first"] = det.get((T, "n_points"), np.nan)
            row["n_draws"] = len(draws)
            rows.append(row)
        print(f"    n_sets = {n:2d}/{N}  ({reps} draw(s))", flush=True)
    return pd.DataFrame(rows)


def sweep_nv(df_state, all_series, muG, muH, temps, comp_info,
             nv_lo=2, nv_hi=32):
    """All sets, fit window n_v <= n_max for n_max = nv_lo..nv_hi."""
    rows = []
    for nmax in range(nv_lo, nv_hi + 1):
        res = run_chain(df_state, all_series, muG, muH, temps, comp_info,
                        nv_max=nmax)
        for T in temps:
            row = {"nv_max": nmax, "x_max": nmax / 64.0, "Temperature": T}
            for q in QUANTS:
                row[q] = res.get((T, q), np.nan)
            for p in PRESSURES:
                tag = S.fmt_pressure(p)
                row[f"x0raw_{tag}"] = res.get((T, f"x0raw_{tag}"), np.nan)
                row[f"x0std_{tag}"] = res.get((T, f"x0std_{tag}"), np.nan)
            for k in ("m", "se_m", "b", "R2", "n_points"):
                row[k] = res.get((T, k), np.nan)
            rows.append(row)
        print(f"    n_v <= {nmax:2d}", flush=True)
    return pd.DataFrame(rows)


# ---------- plots ----------

def _style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=9)


def plot_n_sets(df, temps, out_png, dpi=200):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    colors = {temps[0]: "#1f77b4"}
    if len(temps) > 1:
        colors[temps[1]] = "#d62728"

    for ax, q in zip(axes.ravel(), QUANTS):
        for T in temps:
            d = df[df["Temperature"] == T].sort_values("n_sets")
            c = colors.get(T, "#555555")
            ax.fill_between(d["n_sets"], d[f"{q}_mean"] - d[f"{q}_std"],
                            d[f"{q}_mean"] + d[f"{q}_std"],
                            color=c, alpha=0.18, lw=0)
            ax.plot(d["n_sets"], d[f"{q}_mean"], "-", color=c, lw=1.8,
                    label=f"{T} K  random subsets (mean ±1σ)")
            ax.plot(d["n_sets"], d[f"{q}_first"], "--", color=c, lw=1.0,
                    alpha=0.85, label=f"{T} K  first n sets")
            # the converged value = all sets
            full = d[d["n_sets"] == d["n_sets"].max()][f"{q}_mean"]
            if not full.empty and np.isfinite(full.iloc[0]):
                ax.axhline(full.iloc[0], color=c, ls=":", lw=0.9, alpha=0.6)
        ax.set_title(QLABEL[q], fontsize=11)
        ax.set_ylabel("vacancy fraction $x = n_v/64$", fontsize=9)
        _style(ax)

    for ax in axes[1]:
        ax.set_xlabel("number of SQS sets averaged, $n_{\\rm sets}$", fontsize=10)
    axes[0, 0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle("ΔG = 0 crossing vs how many sets are averaged\n"
                 "dotted line = value at all sets; band = spread over random subsets "
                 "of that size", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"[*] wrote {out_png}")


def plot_nv(df, temps, out_png, dpi=200):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    colors = {temps[0]: "#1f77b4"}
    if len(temps) > 1:
        colors[temps[1]] = "#d62728"

    for ax, q in zip(axes.ravel(), QUANTS):
        for T in temps:
            d = df[df["Temperature"] == T].sort_values("nv_max")
            c = colors.get(T, "#555555")
            ax.plot(d["nv_max"], d[q], "o-", color=c, ms=3.5, lw=1.6,
                    label=f"{T} K")
            full = d[d["nv_max"] == d["nv_max"].max()][q]
            if not full.empty and np.isfinite(full.iloc[0]):
                ax.axhline(full.iloc[0], color=c, ls=":", lw=0.9, alpha=0.6)
        ax.set_title(QLABEL[q], fontsize=11)
        ax.set_ylabel("vacancy fraction $x = n_v/64$", fontsize=9)
        _style(ax)

    for ax in axes[1]:
        ax.set_xlabel("widest vacancy count kept in the fit, "
                      "$n_v^{\\max}$  ($x_{\\max} = n_v^{\\max}/64$)", fontsize=10)
    axes[0, 0].legend(fontsize=9, loc="best", framealpha=0.9)
    fig.suptitle("ΔG = 0 crossing vs the fitting window in x = $n_v$/64  "
                 "(all sets; fit uses pairs with $n_v \\leq n_v^{\\max}$)\n"
                 "dotted line = value at the full window", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"[*] wrote {out_png}")


def plot_nv_diagnostics(df, temps, out_png, dpi=200):
    """
    Why the n_v sweep has a discontinuity: x0 = -b/m blows up where the fitted
    slope m crosses zero.  Top panel is the UNFLOORED x0 on a symlog axis (the
    headline table clips it at 0, which turns the pole into a flat stretch of
    zeros that reads like physics and is not); bottom is m with its ±1 s.e. band.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1]})
    colors = {temps[0]: "#1f77b4"}
    if len(temps) > 1:
        colors[temps[1]] = "#d62728"

    ax = axes[0]
    for T in temps:
        d = df[df["Temperature"] == T].sort_values("nv_max")
        c = colors.get(T, "#555555")
        ax.plot(d["nv_max"], d["x0raw_1atm"], "o-", color=c, ms=3.5, lw=1.5,
                label=f"{T} K, 1 atm")
        ax.plot(d["nv_max"], d["x0raw_3e-05atm"], "s--", color=c, ms=3,
                lw=1.2, alpha=0.65, label=f"{T} K, 3e-5 atm")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(0, 0.5, color="#2ca02c", alpha=0.07, lw=0)
    # linear while the crossing stays in range; symlog only if a pole threw it out,
    # so a well-behaved sweep is not squashed into the corner of a log axis
    cols = [c for c in df.columns if c.startswith("x0raw_")]
    vals = pd.to_numeric(df[cols].to_numpy().ravel(), errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.size and (vals.min() < -0.05 or vals.max() > 1.0):
        ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("unfloored crossing $x_0 = -b/m$", fontsize=10)
    ax.set_title("the crossing is a ratio $-b/m$: meaningful only while the slope "
                 "stays clear of 0\n(shaded = the physical window "
                 "$0 \\leq x \\leq 0.5$)", fontsize=11)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    _style(ax)

    ax = axes[1]
    for T in temps:
        d = df[df["Temperature"] == T].sort_values("nv_max")
        c = colors.get(T, "#555555")
        ax.plot(d["nv_max"], d["m"], "o-", color=c, ms=3.5, lw=1.5, label=f"{T} K")
        ax.fill_between(d["nv_max"], d["m"] - d["se_m"], d["m"] + d["se_m"],
                        color=c, alpha=0.18, lw=0)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_ylabel("fitted slope $m$ of $\\Delta G$ vs $x$  (eV)", fontsize=10)
    ax.set_xlabel("widest vacancy count kept in the fit, $n_v^{\\max}$", fontsize=10)
    ax.set_title("$m$ ± 1 s.e.: a band straddling 0 makes the crossing above meaningless",
                 fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    _style(ax)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"[*] wrote {out_png}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True,
                    help="state_series_raw.csv written by stage 9")
    ap.add_argument("--outdir", default=None,
                    help="where the CSVs and PNGs go (default: next to --raw)")
    ap.add_argument("--temps", default=None, help="e.g. 670,1070 (default: all in the file)")
    ap.add_argument("--resamples", type=int, default=40,
                    help="random n-subsets per n_sets step (default 40)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nv-min", type=int, default=2)
    ap.add_argument("--nv-max", type=int, default=32)
    ap.add_argument("--avg-mode", choices=["simple", "boltz"], default="simple")
    ap.add_argument("--s-config-a", type=float, default=None,
                    help="SIGNED a: re-derive G_state = E + F - T*S_config(a) from "
                         "E_state/F_state before the sweeps, so the study can be run "
                         "on a different S_config branch than the one that wrote the "
                         "CSV (-2 physical, +2 as printed, 0 dropped). Omit to use "
                         "G_state exactly as written.")
    ap.add_argument("--nv-tag-tol", type=int, default=1,
                    help="drop cells whose nv contradicts their o#### tag by more "
                         "than this (stage 9's own guard; default 1)")
    ap.add_argument("--keep-mislabelled", action="store_true",
                    help="skip that guard and keep every row as written")
    ap.add_argument("--mu-g", default="670:-5.1768,1070:-5.6699")
    ap.add_argument("--mu-h", default="670:-4.37519,1070:-4.304685")
    ap.add_argument("--elements", default=None,
                    help="A_base,A_dop,B_base,B_dop fallback if the dir name will not parse")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)

    raw = Path(args.raw).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else raw.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df_state = pd.read_csv(raw)
    temps = (sorted(int(t) for t in df_state["Temperature"].unique())
             if args.temps is None
             else [int(t.strip()) for t in args.temps.split(",") if t.strip()])
    df_state = df_state[df_state["Temperature"].isin(temps)]

    if not args.keep_mislabelled:
        df_state, dropped = reject_mislabelled_cells(df_state, tol=args.nv_tag_tol)
        if dropped:
            print(f"[warn] {len(dropped)} cell(s) rejected: nv contradicts the o-tag "
                  f"by more than {args.nv_tag_tol}. Their energy is discarded, so the "
                  f"point averages over the remaining sets:")
            for series, o, nv, nv_tag in dropped[:10]:
                print(f"      {series}/{o}: structure has nv={nv:g}, "
                      f"o-tag implies nv={nv_tag:g}")
            if len(dropped) > 10:
                print(f"      ... and {len(dropped) - 10} more")

    # S_config branch.  G_state in the CSV was written with whatever `a` the stage 9
    # run used; re-deriving it here from E and F -- which do not depend on S_config --
    # makes the sweep exact for another branch rather than an estimate.  Passing the
    # branch the file already used is a no-op, and the printed max shift shows it.
    if args.s_config_a is not None:
        need = {"E_state", "F_state", "delta", "n_fu", "Temperature"}
        missing = need - set(df_state.columns)
        if missing:
            raise SystemExit(f"--s-config-a needs {sorted(missing)} in {raw.name}; "
                             f"regenerate it with a current stage 9")
        before = pd.to_numeric(df_state["G_state"], errors="coerce")
        df_state = S.recompute_state_G(df_state, args.s_config_a)
        shift = (pd.to_numeric(df_state["G_state"], errors="coerce") - before).abs()
        print(f"[*] S_config re-derived at a = {args.s_config_a:+g} "
              f"({S.s_config_label(args.s_config_a)}); "
              f"max |ΔG_state| vs the file = {shift.max():.6g} eV "
              f"({'no change: the file already used this branch' if shift.max() < 1e-9 else 'the file used a different branch'})")

    all_series = sorted(df_state["series"].unique())
    comps = sorted(df_state["comp_dir"].unique())
    comp_info = {c: S.parse_comp_dir(c) for c in comps}
    if args.elements:
        eb = tuple(s.strip() for s in args.elements.split(","))
        for c, v in list(comp_info.items()):
            if v is None:
                a, b = S.parse_composition(c, eb)
                comp_info[c] = {"elements": eb, "a_base": eb[0], "a_dopant": eb[1],
                                "b_base": eb[2], "b_dopant": eb[3],
                                "A_frac": a, "B_frac": b, "a_vac": 0.0, "b_vac": 0.0,
                                "family": f"{eb[0]}{eb[1]}_{eb[2]}{eb[3]}",
                                "formula": c}
    bad = [c for c, v in comp_info.items() if v is None]
    if bad:
        raise SystemExit(f"cannot parse composition dir(s) {bad}; pass --elements")

    muG = S.parse_mu_table(args.mu_g)
    muH = S.parse_mu_table(args.mu_h)

    print(f"[*] {len(all_series)} sets, {len(comps)} composition(s), temps {temps}")
    print(f"[*] pressures: {PRESSURES} atm;  x_swing = x({P_LO:g}) - x({P_HI:g})")

    print("[*] sweep A: n_sets")
    df_sets = sweep_n_sets(df_state, all_series, muG, muH, temps, comp_info,
                           resamples=args.resamples, seed=args.seed)
    p = outdir / "convergence_vs_n_sets.csv"
    df_sets.to_csv(p, index=False)
    print(f"[*] wrote {p}")
    plot_n_sets(df_sets, temps, outdir / "convergence_vs_n_sets.png", dpi=args.dpi)

    print(f"[*] sweep B: n_v window ({args.nv_min}..{args.nv_max}), all {len(all_series)} sets")
    df_nv = sweep_nv(df_state, all_series, muG, muH, temps, comp_info,
                     nv_lo=args.nv_min, nv_hi=args.nv_max)
    p = outdir / "convergence_vs_nv_window.csv"
    df_nv.to_csv(p, index=False)
    print(f"[*] wrote {p}")
    plot_nv(df_nv, temps, outdir / "convergence_vs_nv_window.png", dpi=args.dpi)
    plot_nv_diagnostics(df_nv, temps,
                        outdir / "convergence_vs_nv_window_diagnostics.png", dpi=args.dpi)

    # a short console read-out: the last few steps of each sweep
    pd.set_option("display.width", 160)
    for T in temps:
        d = df_sets[df_sets["Temperature"] == T].tail(3)
        print(f"\n[n_sets tail, T={T} K]")
        print(d[["n_sets"] + [f"{q}_mean" for q in QUANTS] +
                [f"{q}_std" for q in QUANTS]].to_string(index=False, float_format="%.4f"))
        d = df_nv[df_nv["Temperature"] == T]
        print(f"\n[n_v window, T={T} K]")
        print(d[["nv_max", "n_points"] + QUANTS + ["R2"]].to_string(
            index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
