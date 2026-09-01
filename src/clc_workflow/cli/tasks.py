#!/usr/bin/env python3
"""
Lay out one training task per fold: data/task.000y/{train,valid}/iter_x/fold_n.

    clc tasks <root> [--out data] [--input-template input.json] [--copy] [--dry-run]

<root> holds the folds, either directly or one iteration deep:

    iter_1/fold_0 ... iter_1/fold_4        several rounds of data collection,
    iter_2/fold_0 ... iter_2/fold_4        each K-folded the same way
    ...

Fold y of EVERY iteration becomes the validation set of task.000y, and every other fold
becomes its training set:

    data/task.0000/valid/iter_1/fold_0     data/task.0000/train/iter_1/fold_1 ... fold_4
                        iter_2/fold_0                          iter_2/fold_1 ... fold_4

Pairing the iterations up by fold index is what keeps the K tasks comparable.  A task that
validated on all of iter_3 instead would be scoring transfer between collection rounds,
which is a different question from the one the other tasks answer.

The iteration and fold names are kept rather than flattened, so a task directory still says
where each piece came from, and two iterations contributing the same fold index cannot
collide.  Each fold is one link to the original directory; `--copy` writes real copies, for
a filesystem where the training job cannot follow a link.

PASS THE SYSTEMS EXPLICITLY, not the parent directory.  Every task writes systems.json
naming its system directories one by one, and `--input-template` fills a copy of your
input.json per task from that list.  Handing deepmd `train/` to walk instead would find
nothing: a directory walk does not descend into a symlink, so the training set would come
back empty with nothing obviously wrong.
"""
import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

from clc_workflow.kfold import find_systems

_FOLD = re.compile(r"^fold[_-]?(\d+)$")


def discover(root, exclude=()):
    """
    <root> -> {fold index: [(iteration, fold name, fold directory, [systems])]}.

    A fold directory is found wherever it sits under the root, so `root/fold_0` and
    `root/iter_1/fold_0` are both picked up and both count as fold 0.  `iteration` is
    whatever path stands between the root and the fold -- "" when the folds sit directly
    under it -- and it is reproduced inside every task directory.

    Task directories already under the root are pruned, by path and by the systems.json
    every task carries.  Without that, a second run reads its own output back: the fold
    directories inside data/task.0000/train/ look exactly like source folds, and each
    would be counted again under a new iteration name.  Symlinked tasks hide this, since
    a walk does not follow them -- it is `--copy` that makes the trap real, which is
    precisely the case where it is hardest to notice.
    """
    root = os.path.abspath(root)
    exclude = {os.path.abspath(p) for p in exclude}
    folds = {}
    for cur, dirs, files in os.walk(root):
        dirs.sort()
        if os.path.abspath(cur) in exclude or "systems.json" in files:
            dirs[:] = []
            continue
        m = _FOLD.match(os.path.basename(cur))
        if not m:
            continue
        dirs[:] = []                       # folds do not nest inside folds
        rel = os.path.relpath(cur, root)
        systems = find_systems(cur)
        if not systems:
            print(f"[warn] {rel}: no deepmd/npy systems under it; skipped")
            continue
        iteration = os.path.dirname(rel)
        folds.setdefault(int(m.group(1)), []).append(
            (iteration, os.path.basename(cur), cur, systems))
    return folds


def count_frames(sysdir):
    """Frames in one system, or 0 if it carries nothing countable."""
    n = 0
    for s in sorted(os.listdir(sysdir)):
        c = os.path.join(sysdir, s, "coord.npy")
        if s.startswith("set.") and os.path.isfile(c):
            n += np.load(c, mmap_mode="r").shape[0]
    return n


def place(dest, entries, copy):
    """
    Reproduce `entries` under `dest` as <iteration>/<fold name>, and return the system
    directories at their new paths.
    """
    out = []
    for iteration, fold_name, fold_dir, systems in entries:
        target = os.path.join(dest, iteration, fold_name) if iteration \
            else os.path.join(dest, fold_name)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        if os.path.islink(target):
            os.unlink(target)
        elif os.path.isdir(target):
            shutil.rmtree(target)
        if copy:
            shutil.copytree(fold_dir, target)
        else:
            os.symlink(fold_dir, target)
        for sysdir in systems:
            rel = os.path.relpath(sysdir, fold_dir)
            out.append(os.path.abspath(target if rel == "."
                                       else os.path.join(target, rel)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".",
                    help="directory holding the folds, directly or one iteration deep "
                         "(default: the current directory)")
    ap.add_argument("--out", default="data", help="where the task directories go")
    ap.add_argument("--prefix", default="task.",
                    help="task directory prefix (default: task.)")
    ap.add_argument("--width", type=int, default=4,
                    help="digits in the task number, so task.0000 (default 4)")
    ap.add_argument("--input-template", default=None,
                    help="an input.json to copy into every task with its "
                         "training_data.systems and validation_data.systems filled in")
    ap.add_argument("--copy", action="store_true",
                    help="copy the fold directories instead of symlinking them")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the tasks and their sizes; write nothing")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.root):
        sys.exit(f"[ERROR] no such directory: {args.root}")
    folds = discover(args.root, exclude=[args.out])
    if not folds:
        sys.exit(f"[ERROR] {args.root}: found no fold directories.  Expected fold_0, "
                 f"fold_1, ... either directly under it or one level down "
                 f"(iter_1/fold_0, ...)")

    order = sorted(folds)
    iterations = sorted({it for e in folds.values() for it, _, _, _ in e})
    print(f"[*] root         : {os.path.abspath(args.root)}")
    print(f"[*] folds        : {len(order)} ({', '.join('fold_%d' % k for k in order)})")
    print(f"[*] iterations   : {len(iterations)} "
          f"({', '.join(it or '<root>' for it in iterations)})")

    frames = {}
    for k in order:
        n = sum(count_frames(s) for _, _, _, systems in folds[k] for s in systems)
        nsys = sum(len(systems) for _, _, _, systems in folds[k])
        frames[k] = n
        srcs = ", ".join(os.path.join(it, fn) if it else fn
                         for it, fn, _, _ in sorted(folds[k]))
        print(f"[*]   fold_{k}: {nsys} system(s), {n} frame(s)   <- {srcs}")

    # An iteration that is missing a fold is not fatal -- the tasks still make sense --
    # but it does mean the K validation sets are not drawn alike, so say so rather than
    # letting one task quietly validate on less data than its neighbours.
    ragged = {k: sorted(set(iterations) - {it for it, _, _, _ in folds[k]}) for k in order}
    ragged = {k: v for k, v in ragged.items() if v}
    if ragged:
        print("[!] not every iteration contributes every fold:")
        for k, missing in ragged.items():
            print(f"[!]   fold_{k} has nothing from {[m or '<root>' for m in missing]}")
        print("[!] the K validation sets are then not drawn alike; check this is "
              "what you meant")

    total = sum(frames.values())
    print(f"\n[*] tasks        : {len(order)}, {total} frame(s) in all")
    plan = []
    for y, k in enumerate(order):
        task = os.path.join(args.out, f"{args.prefix}{y:0{args.width}d}")
        tr = [e for j in order if j != k for e in folds[j]]
        print(f"[*]   {task}: valid = fold_{k} ({frames[k]} frame(s)), "
              f"train = {', '.join('fold_%d' % j for j in order if j != k)} "
              f"({total - frames[k]} frame(s))")
        plan.append((task, tr, folds[k], k))

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return

    template = None
    if args.input_template:
        if not os.path.isfile(args.input_template):
            sys.exit(f"[ERROR] no such template: {args.input_template}")
        with open(args.input_template) as fh:
            template = json.load(fh)
        for key in ("training_data", "validation_data"):
            if key not in template.get("training", {}):
                print(f"[warn] {args.input_template}: training.{key} is absent; "
                      f"it will be created")

    for task, tr, va, k in plan:
        os.makedirs(task, exist_ok=True)
        tr_paths = place(os.path.join(task, "train"), tr, args.copy)
        va_paths = place(os.path.join(task, "valid"), va, args.copy)
        with open(os.path.join(task, "systems.json"), "w") as fh:
            json.dump({"fold": k, "training_data": tr_paths,
                       "validation_data": va_paths}, fh, indent=2)
        if template is not None:
            cfg = json.loads(json.dumps(template))
            cfg.setdefault("training", {}).setdefault("training_data", {})
            cfg["training"].setdefault("validation_data", {})
            cfg["training"]["training_data"]["systems"] = tr_paths
            cfg["training"]["validation_data"]["systems"] = va_paths
            with open(os.path.join(task, "input.json"), "w") as fh:
                json.dump(cfg, fh, indent=2)
        print(f"[*] wrote {task}/  train: {len(tr_paths)} system(s), "
              f"valid: {len(va_paths)} system(s)"
              f"{', input.json' if template is not None else ''}")

    how = "copies of" if args.copy else "symlinks to"
    print(f"\n[*] each task holds {how} the fold directories under train/<iter>/ and "
          f"valid/<iter>/,")
    print(f"[*]   and systems.json names the system directories one by one -- give "
          f"deepmd that")
    print(f"[*]   list, not the parent, since a tree walk does not descend into a symlink")
    if template is None:
        print(f"[*] --input-template input.json would also write a filled-in input.json "
              f"per task")


if __name__ == "__main__":
    main()
