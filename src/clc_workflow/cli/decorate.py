#!/usr/bin/env python3
"""
Decorate one template POSCAR per measured formula and write a K-folded dpdata set.

    clc decorate --poscar NiOOH.vasp --xlsx data.xlsx \
                 --label-col exp --label-name overpotential \
                 --n-configs 5 --kfold 5 --out oer_dataset [--dry-run]

The spreadsheet supplies a composition and a measured number; the POSCAR supplies the
lattice.  For each row the substitutable sites of the template are relabelled to match the
formula, `--n-configs` times with different random decorations, and each decoration becomes
one frame carrying the row's label.

WHY SEVERAL CONFIGURATIONS PER FORMULA.  A formula like Ni0.60Fe0.20Ce0.10Cu0.05Zn0.05O2H1
does not name a structure, it names an ensemble -- the measurement averages over however the
five metals actually sat on the lattice.  One arbitrary decoration would teach the model
that one arrangement is what 242.9 mV means.  Several decorations of the same formula
sharing one label are what tell it which details of a particular arrangement are irrelevant
to the property, exactly as the n_sets SQS realisations do in `clc delta`.

THE DECORATIONS ARE RANDOM, NOT SQS.  Sampling the ensemble is the point here; a single
maximally-disordered cell per formula would be one structure again, and 198 formulas times
n_configs SQS searches is not worth the compute for a property that is being learned from
the label rather than from the short-range order.  `clc sqs` remains the route if you do
want optimised decorations.

THE SPLIT IS BY FORMULA.  Every configuration of one formula shares its label, so a
frame-wise split would score the model on a decoration whose label it already saw.  --kfold
K rotates K held-out sets of whole formulas; --valid-frac holds out one.  Same rule, and the
same fold machinery, as `clc delta`.

Output is deepmd/npy/mixed -- one directory per atom count, with real_atom_types.npy
carrying the per-frame species.  Mixed type is required rather than convenient: every
formula puts different elements on the same sites, so one plain deepmd/npy system, whose
type.raw is fixed for the whole system, cannot hold two of them.

`<label-name>.npy` is written beside coord.npy; the name has to match `property_name` in
the deepmd input.json.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

from clc_workflow.kfold import assign_folds, fold_manifest, write_system

FIXED_DEFAULT = "O,H"


# ------------------------------------------------------------------------- the formula

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿\xa0"), None)
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(text):
    """'Ni0.70Fe0.15Ce0.05Er0.05Tm0.05O2H1' -> {'Ni': .7, ..., 'O': 2.0, 'H': 1.0}."""
    s = str(text).translate(_INVISIBLE).strip().replace(" ", "")
    out = {}
    pos = 0
    for m in _TOKEN.finditer(s):
        if m.start() != pos:
            raise ValueError(f"could not read {s[pos:m.start()]!r} in {text!r}")
        pos = m.end()
        out[m.group(1)] = out.get(m.group(1), 0.0) + (float(m.group(2))
                                                      if m.group(2) else 1.0)
    if pos != len(s):
        raise ValueError(f"trailing {s[pos:]!r} in {text!r}")
    if not out:
        raise ValueError(f"no elements in {text!r}")
    return out


def site_fractions(counts, fixed):
    """The substitutable part of a formula, renormalised to fractions of the sites."""
    sub = {el: v for el, v in counts.items() if el not in fixed}
    if not sub:
        raise ValueError(f"every element of {counts} is in --fixed")
    total = sum(sub.values())
    if total <= 0:
        raise ValueError(f"substitutable amounts sum to {total} in {counts}")
    return {el: v / total for el, v in sub.items()}


def integer_counts(fracs, n_sites):
    """
    Site fractions -> whole numbers of sites, by largest remainder.

    Rounding each fraction on its own would not add up to n_sites; largest remainder
    keeps the total exact and puts the leftover sites where the rounding hurt most.
    The worst |realised - requested| fraction comes back with them, because that error
    is the one thing about this step worth reporting: it is zero when the cell can
    represent the formula and grows quietly when it cannot.
    """
    exact = {el: f * n_sites for el, f in fracs.items()}
    counts = {el: int(np.floor(v)) for el, v in exact.items()}
    left = n_sites - sum(counts.values())
    for el in sorted(exact, key=lambda e: (-(exact[e] - counts[e]), e))[:left]:
        counts[el] += 1
    err = max(abs(counts[el] / n_sites - fracs[el]) for el in fracs)
    return counts, err


# ------------------------------------------------------------------------ the template

def read_template(path):
    """The template POSCAR -> (cartesian coords (natoms,3), cell (3,3), species list)."""
    try:
        import dpdata
    except ImportError:
        sys.exit("[ERROR] dpdata is needed to read the template POSCAR "
                 "(pip install dpdata)")
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] no such POSCAR: {path}")
    s = dpdata.System(path, fmt="vasp/poscar")
    names = s.data["atom_names"]
    species = [names[t] for t in s.data["atom_types"]]
    return (np.asarray(s.data["coords"][0], dtype=float),
            np.asarray(s.data["cells"][0], dtype=float), species)


def decorate(rng, n_sites, counts, n_configs):
    """`n_configs` distinct random assignments of `counts` elements over the sites."""
    pool = [el for el, n in sorted(counts.items()) for _ in range(n)]
    assert len(pool) == n_sites
    seen, out, tries = set(), [], 0
    while len(out) < n_configs and tries < 100 * n_configs:
        tries += 1
        perm = tuple(pool[i] for i in rng.permutation(n_sites))
        if perm in seen:
            continue
        seen.add(perm)
        out.append(list(perm))
    return out


# ------------------------------------------------------------------------------ output

def write_mixed(dest, coords, cell, types, labels, label_name, type_map, set_size):
    """One deepmd/npy/mixed system: <dest>/<natoms>/ with per-frame species."""
    nframes, natoms = types.shape
    arrays = {
        "coord": np.repeat(coords.reshape(1, -1), nframes, axis=0),
        "box": np.repeat(cell.reshape(1, 9), nframes, axis=0),
        "real_atom_types": types.astype(np.int64),
        label_name: np.asarray(labels, dtype=float).reshape(-1, 1),
    }
    sysdir = os.path.join(dest, str(natoms))
    os.makedirs(sysdir, exist_ok=True)
    with open(os.path.join(sysdir, "type.raw"), "w") as fh:
        fh.write("\n".join(["0"] * natoms) + "\n")
    with open(os.path.join(sysdir, "type_map.raw"), "w") as fh:
        fh.write("\n".join(type_map) + "\n")
    raws = {f: os.path.join(sysdir, f) for f in ("type.raw", "type_map.raw")}
    write_system(sysdir, arrays, raws, list(range(nframes)), set_size)
    return sysdir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poscar", required=True,
                    help="template structure; its substitutable sites are relabelled per "
                         "formula, its coordinates and cell are used unchanged")
    ap.add_argument("--xlsx", required=True, help="spreadsheet of formula + label")
    ap.add_argument("--sheet", default=0, help="sheet name or index (default: the first)")
    ap.add_argument("--formula-col", default="formula", help="column holding the formula")
    ap.add_argument("--label-col", default=None,
                    help="column holding the measured value (default: the first numeric "
                         "column that is not the formula)")
    ap.add_argument("--label-name", default="overpotential",
                    help="name of the label array; must equal property_name in the "
                         "deepmd input.json (default: overpotential)")
    ap.add_argument("--keep-cols", default=None,
                    help="comma list of further columns to carry into dataset_index.csv, "
                         "e.g. the iteration a row came from")
    ap.add_argument("--fixed", default=FIXED_DEFAULT,
                    help=f"comma list of species that are NOT substituted, in the formula "
                         f"and in the POSCAR alike (default: {FIXED_DEFAULT})")
    ap.add_argument("--sites", default=None,
                    help="comma list of POSCAR species whose sites get decorated; the "
                         "default is every species not in --fixed")
    ap.add_argument("--n-configs", type=int, default=5,
                    help="random decorations per formula, each one frame (default 5)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the decorations and for dealing formulas into folds")
    ap.add_argument("--round-tol", type=float, default=0.02,
                    help="largest site-fraction error the cell may impose before this is "
                         "an error rather than a note (default 0.02)")
    ap.add_argument("--type-map", default=None,
                    help="comma list fixing the species order; the default is derived "
                         "from the spreadsheet.  It is written to type_map.raw either way")
    ap.add_argument("--kfold", type=int, default=0,
                    help="write K disjoint folds, every formula held out in exactly one; "
                         "overrides --valid-frac")
    ap.add_argument("--valid-frac", type=float, default=0.0,
                    help="fraction of FRAMES held out for validation, by formula "
                         "(default 0: one undivided dataset)")
    ap.add_argument("--out", default="property_dataset", help="output directory")
    ap.add_argument("--set-size", type=int, default=5000, help="frames per set.* dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the parsing, the rounding and the split; write nothing")
    args = ap.parse_args(argv)

    if args.kfold and args.kfold < 2:
        sys.exit("[ERROR] --kfold needs at least 2 folds")
    fixed = {s.strip() for s in args.fixed.split(",") if s.strip()}

    # ---- the spreadsheet -------------------------------------------------------------
    try:
        sheet = int(args.sheet)
    except (TypeError, ValueError):
        sheet = args.sheet
    if not os.path.isfile(args.xlsx):
        sys.exit(f"[ERROR] no such spreadsheet: {args.xlsx}")
    df = pd.read_excel(args.xlsx, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    if args.formula_col not in df.columns:
        sys.exit(f"[ERROR] {args.xlsx} has no column {args.formula_col!r}; it has "
                 f"{list(df.columns)}")
    label_col = args.label_col
    if label_col is None:
        numeric = [c for c in df.columns
                   if c != args.formula_col and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            sys.exit(f"[ERROR] no numeric column to use as the label; name one with "
                     f"--label-col.  Columns are {list(df.columns)}")
        label_col = numeric[0]
        print(f"[*] label column : {label_col!r} (first numeric column; --label-col "
              f"to choose)")
    elif label_col not in df.columns:
        sys.exit(f"[ERROR] {args.xlsx} has no column {label_col!r}; it has "
                 f"{list(df.columns)}")

    n_all = len(df)
    # Blank rows separate the iteration blocks, and a formula with no measurement is a
    # candidate that has not been made yet -- neither is a training row, and both are
    # reported so a shrinking dataset is never a surprise.
    blank = int(df[args.formula_col].isna().sum())
    df = df.dropna(subset=[args.formula_col])
    unlabelled = int(df[label_col].isna().sum())
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    print(f"[*] spreadsheet  : {n_all} row(s) -> {len(df)} usable "
          f"({blank} blank, {unlabelled} with a formula but no {label_col})")
    if df.empty:
        sys.exit("[ERROR] nothing left to build from")
    keep = [c.strip() for c in (args.keep_cols or "").split(",") if c.strip()]
    for c in keep:
        if c not in df.columns:
            sys.exit(f"[ERROR] --keep-cols {c!r} is not a column of {args.xlsx}")

    # ---- the template ----------------------------------------------------------------
    coords, cell, species = read_template(args.poscar)
    if args.sites:
        site_species = {s.strip() for s in args.sites.split(",") if s.strip()}
        unknown = site_species - set(species)
        if unknown:
            sys.exit(f"[ERROR] --sites names {sorted(unknown)}, which the POSCAR does "
                     f"not contain; it has {sorted(set(species))}")
    else:
        site_species = {s for s in species if s not in fixed}
        if not site_species:
            sys.exit(f"[ERROR] every species in the POSCAR ({sorted(set(species))}) is "
                     f"in --fixed, so there is nothing to decorate")
    sites = [i for i, s in enumerate(species) if s in site_species]
    n_sites = len(sites)
    print(f"[*] template     : {args.poscar}  {len(species)} atoms "
          f"({', '.join(f'{n} {e}' for e, n in sorted(Counter(species).items()))})")
    print(f"[*] decorating   : {n_sites} site(s) held by {sorted(site_species)}; "
          f"{sorted(set(species) - site_species)} kept as they are")

    # ---- parse every formula against those sites --------------------------------------
    rows, dropped = [], []
    for i, r in df.iterrows():
        f = r[args.formula_col]
        try:
            fracs = site_fractions(parse_formula(f), fixed)
        except ValueError as e:
            dropped.append((f, str(e)))
            continue
        counts, err = integer_counts(fracs, n_sites)
        rows.append({"formula": str(f).strip(), "label": float(r[label_col]),
                     "counts": counts, "round_err": err, "xlsx_row": int(i) + 2,
                     **{c: r[c] for c in keep}})
    for f, why in dropped:
        print(f"    [SKIP] {f}: {why}")
    if not rows:
        sys.exit("[ERROR] no formula could be parsed")

    worst = max(r["round_err"] for r in rows)
    print(f"[*] formulas     : {len(rows)} parsed, worst site-fraction rounding error "
          f"{worst:.4f}")
    if worst > args.round_tol:
        bad = sorted(rows, key=lambda r: -r["round_err"])[:5]
        for r in bad:
            print(f"    [ROUNDING] {r['formula']}: off by {r['round_err']:.4f}")
        sys.exit(f"\n[ERROR] {n_sites} substitutable site(s) cannot represent these "
                 f"formulas to within --round-tol {args.round_tol}.\n"
                 f"        The fractions in this sheet are multiples of 0.05, so the "
                 f"cell needs a multiple of 20 sites.\n"
                 f"        Use a larger supercell, or raise --round-tol to accept the "
                 f"approximation.")

    # ---- species order ----------------------------------------------------------------
    if args.type_map:
        type_map = [s.strip() for s in args.type_map.split(",") if s.strip()]
    else:
        used = sorted({el for r in rows for el in r["counts"]})
        type_map = used + [e for e in sorted(set(species)) if e not in used]
    missing = {el for r in rows for el in r["counts"]} - set(type_map)
    missing |= set(species) - set(type_map)
    if missing:
        sys.exit(f"[ERROR] --type-map leaves out {sorted(missing)}")
    tid = {el: i for i, el in enumerate(type_map)}
    print(f"[*] type_map     : {len(type_map)} species, {type_map}")

    # ---- decorate ----------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    base_types = np.array([tid[s] for s in species], dtype=np.int64)
    frames, index = [], []
    short = 0
    for r in rows:
        configs = decorate(rng, n_sites, r["counts"], args.n_configs)
        if len(configs) < args.n_configs:
            short += 1
        for c, assignment in enumerate(configs):
            t = base_types.copy()
            for site, el in zip(sites, assignment):
                t[site] = tid[el]
            frames.append(t)
            index.append({"formula": r["formula"], "config": c, args.label_name: r["label"],
                          "round_err": round(r["round_err"], 6), "xlsx_row": r["xlsx_row"],
                          **{k: r[k] for k in keep}})
    types = np.stack(frames)
    idx = pd.DataFrame(index)
    print(f"[*] frames       : {len(idx)} = {len(rows)} formula(s) x "
          f"{args.n_configs} configuration(s), {len(species)} atoms each")
    if short:
        print(f"[!]   {short} formula(s) have fewer distinct decorations than "
              f"--n-configs; they carry what exists")

    # ---- split, by formula --------------------------------------------------------------
    by_formula = idx.groupby("formula").size()
    if args.kfold:
        if len(by_formula) < args.kfold:
            sys.exit(f"[ERROR] --kfold {args.kfold} needs at least {args.kfold} formulas; "
                     f"there are {len(by_formula)}")
        fold_of = assign_folds(by_formula.to_dict(), args.kfold, args.seed)
        idx["fold"] = idx["formula"].map(fold_of)
        idx["split"] = "fold_" + idx["fold"].astype(str)
        print(f"\n[*] split        : {args.kfold}-fold, held out by formula "
              f"({len(by_formula)} formula(s), {len(idx)} frame(s))")
        for k in range(args.kfold):
            g = idx[idx["fold"] == k]
            print(f"[*]   fold {k}: {len(g)} valid / {len(idx) - len(g)} train frame(s) "
                  f"({len(g) / len(idx):.1%}), {g['formula'].nunique()} formula(s)")
        splits = [(f"fold_{k}", idx[idx["fold"] == k]) for k in range(args.kfold)]
    elif args.valid_frac > 0:
        rng2 = np.random.default_rng(args.seed)
        order = list(by_formula.index)
        rng2.shuffle(order)
        target, held, got = args.valid_frac * len(idx), [], 0
        for f in order:
            if got >= target:
                break
            held.append(f)
            got += by_formula[f]
        idx["split"] = np.where(idx["formula"].isin(held), "valid", "train")
        n_valid = int((idx["split"] == "valid").sum())
        print(f"\n[*] split        : {len(idx) - n_valid} train / {n_valid} valid "
              f"frame(s) ({n_valid / len(idx):.1%}), held out by formula")
        splits = [("train", idx[idx["split"] == "train"]),
                  ("valid", idx[idx["split"] == "valid"])]
    else:
        idx["split"] = ""
        splits = [("", idx)]

    idx_path = os.path.join(args.out, "dataset_index.csv")
    if args.dry_run:
        print(f"\n[dry-run] would write "
              + ", ".join(f"{len(g)} frame(s) to {os.path.join(args.out, n)}/"
                          for n, g in splits if len(g))
              + f" and the index to {idx_path}; nothing written.")
        return

    # ---- write ---------------------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    systems_of = {}
    for name, part in splits:
        if part.empty:
            continue
        dest = os.path.join(args.out, name) if name else args.out
        sysdir = write_mixed(dest, coords, cell, types[part.index.to_numpy()],
                             part[args.label_name].to_numpy(float), args.label_name,
                             type_map, args.set_size)
        systems_of[name] = [sysdir]
        print(f"\n[*] wrote {len(part)} frame(s) to {sysdir}/ (deepmd/npy/mixed)")

    idx.to_csv(idx_path, index=False)
    if args.kfold:
        folds = fold_manifest(args.out, args.kfold,
                              {k: systems_of.get(f"fold_{k}", [])
                               for k in range(args.kfold)})
        for k in range(args.kfold):
            folds[f"fold_{k}"]["valid_formulas"] = \
                sorted(idx.loc[idx["fold"] == k, "formula"].unique())
            folds[f"fold_{k}"]["n_valid_frames"] = int((idx["fold"] == k).sum())
            folds[f"fold_{k}"]["n_train_frames"] = int((idx["fold"] != k).sum())
        folds_path = os.path.join(args.out, "folds.json")
        with open(folds_path, "w") as fh:
            json.dump(folds, fh, indent=2)
        print(f"\n[*] folds -> {folds_path}   (the systems lists for each of the "
              f"{args.kfold} runs)")
    print(f"[*] index -> {idx_path}")
    print(f"[*] label file is {args.label_name}.npy; set property_name = "
          f"{args.label_name!r} in the deepmd input.json")
    if args.kfold:
        print(f"[*] run {args.kfold} trainings, taking each fold's training_data.systems "
              f"and\n[*]   validation_data.systems from folds.json, then average the "
              f"scores weighted by fold size")
    elif args.valid_frac > 0:
        print(f"[*]   training_data.systems   = {os.path.join(args.out, 'train')}/*")
        print(f"[*]   validation_data.systems = {os.path.join(args.out, 'valid')}/*")
    else:
        print(f"[*]   training_data.systems   = {args.out}/*")


if __name__ == "__main__":
    main()
