#!/usr/bin/env python3
"""
Lay out one training task per fold: data/task.000y/{train,valid}/iter_x/fold_n.

    clc tasks <root> [--out data] [--iters ... | --upto iter_N]
                     [--input-template input.json] [--model ckpt.pt]
                     [--task-model 'prev/task.{task}/model.ckpt.pt'] [--copy] [--dry-run]

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

ROUND N TRAINS ON ITERATIONS 1..N.  `--upto iter_2` selects iter_1 and iter_2 and leaves
later rounds out; `--iters` names them outright.  Each round is a fresh K-fold over the
data collected so far, which is why the fold directories are re-read every time rather
than a previous round's tasks being extended.

`--model` copies one pretrained checkpoint into every task.  `--task-model` gives each task
its own, which is what chains the rounds: fold k of round N continues from the model fold k
produced in round N-1, and never from a model that has already trained on fold k's data.
Point it at the previous round with a pattern --

    clc tasks . --upto iter_2 --out data_iter2 \
                --task-model 'data_iter1/task.{task}/model.ckpt.pt'

Models are copied rather than linked on purpose: runs sharing one file would write over
each other's output the moment any of them saves beside it.

PASS THE SYSTEMS EXPLICITLY, not the parent directory.  Every task writes systems.json
naming its system directories one by one, and `--input-template` fills a copy of your
input.json per task from that list.  Handing deepmd `train/` to walk instead would find
nothing: a directory walk does not descend into a symlink, so the training set would come
back empty with nothing obviously wrong.
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import sys

import numpy as np

from clc_workflow.kfold import find_systems

_FOLD = re.compile(r"^fold[_-]?(\d+)$")


def natural(name):
    """Sort key that reads embedded numbers as numbers, so iter_10 follows iter_9."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


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
    ap.add_argument("--iters", nargs="+", default=None,
                    help="iterations to include, by name or glob (iter_1 iter_2, or "
                         "'iter_[12]').  The default is every iteration under the root")
    ap.add_argument("--upto", default=None,
                    help="include every iteration up to and including this one, in "
                         "natural order -- the cumulative form: --upto iter_2 trains on "
                         "iter_1 + iter_2")
    ap.add_argument("--task-model", nargs="+", default=None,
                    help="a DIFFERENT model per task, for chaining rounds: each fold "
                         "continues from the model that same fold produced last round.  "
                         "Either one path per task in order, or a single pattern with "
                         "{task} (the zero-padded number), {y} (the index) or {fold} "
                         "(the fold index), e.g. "
                         "'data_iter1/task.{task}/model.ckpt.pt'")
    ap.add_argument("--model", nargs="+", default=None,
                    help="file(s) copied into EVERY task directory -- the shared "
                         "pretrained checkpoint, and anything else each run needs beside "
                         "it.  Copied, not linked: a run that writes next to its model "
                         "must not write into the others.  Use --task-model when the "
                         "tasks need different models")
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

    if args.iters and args.upto:
        sys.exit("[ERROR] --iters lists the iterations outright and --upto takes every "
                 "one through a given name.  Use one or the other")

    available = sorted({it for e in folds.values() for it, _, _, _ in e}, key=natural)
    if args.iters or args.upto:
        if args.upto:
            if args.upto not in available:
                sys.exit(f"[ERROR] --upto {args.upto!r} is not an iteration under "
                         f"{args.root}; it has "
                         f"{[a or '<root>' for a in available]}")
            cut = available.index(args.upto)
            wanted = set(available[:cut + 1])
            print(f"[*] selection    : --upto {args.upto} -> "
                  f"{', '.join(available[:cut + 1])}")
        else:
            wanted = {a for a in available
                      if any(fnmatch.fnmatch(a, pat) for pat in args.iters)}
            unmatched = [pat for pat in args.iters
                         if not any(fnmatch.fnmatch(a, pat) for a in available)]
            if unmatched:
                sys.exit(f"[ERROR] --iters {unmatched} matched no iteration; the root "
                         f"has {[a or '<root>' for a in available]}")
            print(f"[*] selection    : --iters -> "
                  f"{', '.join(sorted(wanted, key=natural))}")
        folds = {k: [e for e in v if e[0] in wanted] for k, v in folds.items()}
        folds = {k: v for k, v in folds.items() if v}
        if not folds:
            sys.exit("[ERROR] the selected iterations hold no folds")

    order = sorted(folds)
    iterations = sorted({it for e in folds.values() for it, _, _, _ in e}, key=natural)
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

    models = []
    for m in (args.model or []):
        if not os.path.exists(m):
            sys.exit(f"[ERROR] no such model file: {m}")
        models.append(os.path.abspath(m))

    # One model per task, either spelled out in task order or generated from a pattern.
    # Resolved and checked here, before any directory is made, so a wrong pattern fails
    # with the paths it tried rather than leaving a half-built tree behind.
    task_models = {}
    if args.task_model:
        spec = args.task_model
        if len(spec) == 1 and "{" in spec[0]:
            want = {y: spec[0].format(task=f"{y:0{args.width}d}", y=y, fold=k)
                    for y, k in enumerate(order)}
        elif len(spec) == len(order):
            want = dict(enumerate(spec))
        else:
            sys.exit(f"[ERROR] --task-model was given {len(spec)} path(s) for "
                     f"{len(order)} task(s).\n"
                     f"        Give one per task in order, or a single pattern "
                     f"containing {{task}}, {{y}} or {{fold}}.")
        missing = {y: p for y, p in want.items() if not os.path.exists(p)}
        if missing:
            sys.exit("[ERROR] --task-model resolved to paths that do not exist:\n"
                     + "\n".join(f"        task {y:0{args.width}d} -> {p}"
                                 for y, p in sorted(missing.items())))
        task_models = {y: os.path.abspath(p) for y, p in want.items()}
        print(f"[*] task model   : one per task, e.g. task {0:0{args.width}d} -> "
              f"{os.path.relpath(task_models[0])}")
    if models:
        print(f"[*] model        : "
              + ", ".join(f"{os.path.basename(m)} "
                          f"({os.path.getsize(m) / 1e6:.1f} MB)" if os.path.isfile(m)
                          else f"{os.path.basename(m)}/ (directory)" for m in models)
              + " -> copied into every task")

    total = sum(frames.values())
    print(f"\n[*] tasks        : {len(order)}, {total} frame(s) in all")
    plan = []
    for y, k in enumerate(order):
        task = os.path.join(args.out, f"{args.prefix}{y:0{args.width}d}")
        tr = [e for j in order if j != k for e in folds[j]]
        print(f"[*]   {task}: valid = fold_{k} ({frames[k]} frame(s)), "
              f"train = {', '.join('fold_%d' % j for j in order if j != k)} "
              f"({total - frames[k]} frame(s))")
        plan.append((task, tr, folds[k], k, y))

    if args.dry_run:
        if models:
            print(f"[*]   each with a copy of "
                  f"{', '.join(os.path.basename(m) for m in models)}")
        for y in sorted(task_models):
            print(f"[*]   task {y:0{args.width}d} also gets "
                  f"{os.path.relpath(task_models[y])}")
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

    for task, tr, va, k, y in plan:
        os.makedirs(task, exist_ok=True)
        tr_paths = place(os.path.join(task, "train"), tr, args.copy)
        va_paths = place(os.path.join(task, "valid"), va, args.copy)
        with open(os.path.join(task, "systems.json"), "w") as fh:
            json.dump({"fold": k, "training_data": tr_paths,
                       "validation_data": va_paths}, fh, indent=2)
        for m in models + ([task_models[y]] if y in task_models else []):
            target = os.path.join(task, os.path.basename(m))
            if os.path.islink(target):
                os.unlink(target)
            elif os.path.isdir(target):
                shutil.rmtree(target)
            if os.path.isdir(m):
                shutil.copytree(m, target)
            else:
                shutil.copyfile(m, target)
        if template is not None:
            cfg = json.loads(json.dumps(template))
            cfg.setdefault("training", {}).setdefault("training_data", {})
            cfg["training"].setdefault("validation_data", {})
            cfg["training"]["training_data"]["systems"] = tr_paths
            cfg["training"]["validation_data"]["systems"] = va_paths
            with open(os.path.join(task, "input.json"), "w") as fh:
                json.dump(cfg, fh, indent=2)
        n_models = len(models) + (1 if y in task_models else 0)
        extra = "".join([", input.json" if template is not None else "",
                         f", {n_models} model file(s)" if n_models else ""])
        print(f"[*] wrote {task}/  train: {len(tr_paths)} system(s), "
              f"valid: {len(va_paths)} system(s){extra}")

    how = "copies of" if args.copy else "symlinks to"
    print(f"\n[*] each task holds {how} the fold directories under train/<iter>/ and "
          f"valid/<iter>/,")
    print(f"[*]   and systems.json names the system directories one by one -- give "
          f"deepmd that")
    print(f"[*]   list, not the parent, since a tree walk does not descend into a symlink")
    if models:
        print(f"[*] each task carries its own copy of "
              f"{', '.join(os.path.basename(m) for m in models)}, so a run can "
              f"fine-tune\n[*]   from it and write its checkpoints beside it without "
              f"touching the other tasks")
    if template is None:
        print(f"[*] --input-template input.json would also write a filled-in input.json "
              f"per task")
    if task_models:
        print(f"[*] each task also carries its own model, so fold k continues from what "
              f"fold k\n[*]   produced last round rather than from a model that has "
              f"already seen fold k")
    if not models and not task_models:
        print(f"[*] --model <checkpoint> copies one model into every task; --task-model "
              f"gives\n[*]   each task its own, for chaining rounds")


if __name__ == "__main__":
    main()
