#!/usr/bin/env python3
"""
Split an already-written dpdata dataset into K folds, without rebuilding it.

    clc kfold <dataset> [<more> ...] -k 5 [--out kfold_dataset]
              [--group-by composition|structure|system|frame] [--seed 0] [--dry-run]

`clc delta --kfold` does this at build time, from the spreadsheet and the structure tree.
This does it to a dataset that already exists, which is what you want when the dataset was
expensive to build, when it came from somewhere else entirely, or when you started with one
train/valid split and now want K.

POINT IT AT THE PARENT OF AN EXISTING SPLIT.  `clc kfold delta_dataset` finds the systems
under `train/` and `valid/` alike and deals all of them out fresh -- the old split is
undone, not folded into the new one.  Give the two directories separately and you get the
same thing.  Any number of datasets can be listed; they are pooled first, so this doubles
as the way to fold several campaigns together.

THE SPLIT IS BY GROUP, NEVER BY FRAME.  Default grouping is by composition, which for this
pipeline is the compound: every SQS realisation and every measurement of one composition
has the same atom counts, so grouping on the species multiset reproduces exactly the
by-compound split `clc delta` writes.  --group-by frame gives plain K-fold and says why you
probably do not want it.

Each fold is written once, as fold_0/ ... fold_{K-1}/ -- disjoint, no duplication on disk.
A fold's training set is the other K-1 directories; folds.json lists both sides for each of
the K runs, and fold_index.csv records where every frame came from and where it went.

The arrays are sliced and the raw files copied verbatim, so whatever the dataset carries --
fparam, delta, energy, force, an array added later -- comes through untouched.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from clc_workflow.kfold import (assign_folds, find_systems, fold_manifest, frame_groups,
                                load_system, write_system)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="+",
                    help="dpdata dataset directory, or a parent of several -- systems are "
                         "found recursively, so a train/ + valid/ pair is pooled and "
                         "redealt.  Several datasets are pooled together")
    ap.add_argument("-k", "--kfold", type=int, required=True, help="number of folds")
    ap.add_argument("--out", default="kfold_dataset", help="output directory")
    ap.add_argument("--group-by", default="composition",
                    choices=["composition", "structure", "system", "frame"],
                    help="what moves as a unit.  'composition' (default) = the compound, "
                         "the same unit clc delta splits on; 'structure' = one cell; "
                         "'system' = one system directory; 'frame' = plain K-fold, which "
                         "leaks a compound's other measurements into training")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for dealing groups into folds")
    ap.add_argument("--set-size", type=int, default=5000, help="frames per set.* dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the grouping and the fold sizes; write nothing")
    args = ap.parse_args(argv)

    if args.kfold < 2:
        sys.exit("[ERROR] -k needs at least 2 folds")

    # ---- pool every system of every input -------------------------------------------
    systems = []
    for ds in args.dataset:
        if not os.path.isdir(ds):
            sys.exit(f"[ERROR] no such dataset directory: {ds}")
        found = find_systems(ds)
        if not found:
            sys.exit(f"[ERROR] {ds}: no deepmd/npy systems under it "
                     f"(looking for a directory with type.raw and a set.*)")
        for sysdir in found:
            systems.append((ds, sysdir))
    print(f"[*] inputs       : "
          + ", ".join(f"{ds} ({sum(1 for d, _ in systems if d == ds)} system(s))"
                      for ds in args.dataset))

    # Systems from different inputs merge when they carry the same name and the same raw
    # files -- which is exactly the train/40 + valid/40 case, two halves of one system.
    # A shared name over different raws is a genuine collision and is not silently merged.
    loaded, order, raws_of, sig_of = {}, [], {}, {}
    total = 0
    for ds, sysdir in systems:
        arrays, raws, n = load_system(sysdir)
        name = os.path.basename(sysdir)
        sig = tuple(sorted((f, open(p, "rb").read()) for f, p in raws.items()))
        if name in sig_of and sig_of[name] != sig:
            sys.exit(f"[ERROR] two systems are both named {name!r} but carry different "
                     f"type.raw/type_map.raw:\n        {sysdir}\n        and an earlier "
                     f"one.  Pooling them would mix two different systems under one name; "
                     f"rename one, or fold them separately")
        if name not in loaded:
            order.append(name)
            loaded[name] = defaultdict(list)
            raws_of[name], sig_of[name] = raws, sig
        for key, a in arrays.items():
            loaded[name][key].append(a)
        total += n
        print(f"[*]   {os.path.relpath(sysdir, ds)}  {n} frame(s), "
              f"{', '.join(sorted(arrays))}")

    pooled = {name: {k: np.concatenate(v, axis=0) for k, v in sorted(loaded[name].items())}
              for name in order}
    print(f"[*] pooled       : {len(pooled)} system(s), {total} frame(s)")

    # ---- group, then deal the groups out --------------------------------------------
    rows = []                     # (system name, frame index within the pooled system)
    keys = []
    for name in order:
        g = frame_groups(pooled[name], raws_of[name], args.group_by, name)
        for i, key in enumerate(g):
            rows.append((name, i))
            keys.append(key)

    sizes = Counter(keys)
    print(f"[*] grouping     : {len(sizes)} group(s) by {args.group_by}, "
          f"{min(sizes.values())}-{max(sizes.values())} frame(s) each")
    if args.group_by == "frame":
        print("[!] --group-by frame is plain K-fold: a compound's other temperatures and "
              "its\n"
              "[!] other SQS realisations end up in training, so the validation score "
              "measures\n"
              "[!] interpolation between near-duplicates rather than transfer to an "
              "unseen material.")
    if len(sizes) < args.kfold:
        sys.exit(f"[ERROR] -k {args.kfold} needs at least {args.kfold} groups; the data "
                 f"falls into {len(sizes)} by {args.group_by}.  Every fold has to hold out "
                 f"a whole group, so K cannot exceed the group count")

    fold_of = assign_folds(sizes, args.kfold, args.seed)
    fold_rows = defaultdict(lambda: defaultdict(list))       # fold -> system -> [frames]
    per_fold = Counter()
    for (name, i), key in zip(rows, keys):
        f = fold_of[key]
        fold_rows[f][name].append(i)
        per_fold[f] += 1

    print(f"\n[*] split        : {args.kfold}-fold, held out by {args.group_by}")
    for f in range(args.kfold):
        ng = sum(1 for g, i in fold_of.items() if i == f)
        print(f"[*]   fold {f}: {per_fold[f]} valid / {total - per_fold[f]} train "
              f"frame(s) ({per_fold[f] / total:.1%}), {ng} group(s)")

    if args.dry_run:
        print(f"\n[dry-run] would write "
              + ", ".join(f"{per_fold[f]} frame(s) to "
                          f"{os.path.join(args.out, f'fold_{f}')}/"
                          for f in range(args.kfold))
              + "; nothing written.")
        return

    # ---- write ----------------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    index = []
    for f in range(args.kfold):
        dest_root = os.path.join(args.out, f"fold_{f}")
        for name in order:
            take = fold_rows[f].get(name)
            if not take:
                continue
            write_system(os.path.join(dest_root, name), pooled[name], raws_of[name],
                         take, args.set_size)
            for pos, i in enumerate(take):
                index.append({"fold": f, "system": name, "src_frame": i,
                              "fold_frame": pos})
        print(f"[*] wrote {per_fold[f]} frame(s) to {dest_root}/")

    folds = fold_manifest(args.out, args.kfold)
    for f in range(args.kfold):
        folds[f"fold_{f}"]["n_valid_frames"] = per_fold[f]
        folds[f"fold_{f}"]["n_train_frames"] = total - per_fold[f]
    folds_path = os.path.join(args.out, "folds.json")
    with open(folds_path, "w") as fh:
        json.dump(folds, fh, indent=2)

    idx_path = os.path.join(args.out, "fold_index.csv")
    with open(idx_path, "w") as fh:
        fh.write("fold,system,src_frame,fold_frame\n")
        for r in index:
            fh.write(f"{r['fold']},{r['system']},{r['src_frame']},{r['fold_frame']}\n")

    print(f"\n[*] folds -> {folds_path}   (the systems lists for each of the "
          f"{args.kfold} runs)")
    print(f"[*] index -> {idx_path}   (which pooled frame went to which fold)")
    print(f"[*] each fold is written once; run {args.kfold} trainings, fold k using")
    print(f"[*]   validation_data.systems = [{os.path.join(args.out, 'fold_k')}/*]")
    print(f"[*]   training_data.systems   = the other {args.kfold - 1} fold globs "
          f"(see folds.json)")
    print(f"[*] average the {args.kfold} validation scores; every group is held out "
          f"exactly once")


if __name__ == "__main__":
    main()
