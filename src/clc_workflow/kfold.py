"""
Group K-fold over a dpdata dataset: the fold assignment, and the npy-level machinery
that splits an already-written dataset without going back to the source structures.

WHY GROUPS AND NOT FRAMES.  A frame here is one (structure, measurement) pair, and a
compound contributes many of them -- every temperature, every pressure window, every SQS
realisation -- all sharing the same label and differing only in fparam.  Dealing those
frames into K parts would train four folds out of five on a near-duplicate of what they
are scored on.  Whole groups move together instead, so a group is validation in exactly
one fold and never sits on both sides of the same one.

WHY THE SPLIT IS DONE ON THE .npy FILES.  Reading a dataset back through dpdata means
teaching it about every extra array first -- fparam, delta, whatever a later dataset
adds -- and anything it does not know about is dropped on the way out.  A deepmd/npy
system is just per-frame arrays stacked along axis 0 plus a few raw files, so slicing
the arrays and copying the raws preserves the dataset exactly, including arrays this
code has never heard of.
"""
import hashlib
import os
import shutil

import numpy as np


# --------------------------------------------------------------------- fold assignment

def assign_folds(weights, k, seed=0):
    """
    {group: n_frames} -> {group: fold index}, balanced on FRAMES.

    Greedy longest-first: shuffle for a seeded tie-break, then hand each group to
    whichever fold holds the fewest frames so far.  Dealing groups round-robin instead
    would leave folds of visibly different size, because groups carry very different
    frame counts -- a compound has between 4 and 7 measurements times n_sets frames.
    Balancing on frames is what makes the K validation scores comparable to each other.

    Perfect balance is not always reachable: a fold cannot hold half a group.
    """
    rng = np.random.default_rng(seed)
    order = list(weights)
    rng.shuffle(order)
    order.sort(key=lambda g: -int(weights[g]))
    load = [0] * k
    fold_of = {}
    for g in order:
        i = int(np.argmin(load))
        fold_of[g] = i
        load[i] += int(weights[g])
    return fold_of


# ------------------------------------------------------------------ reading a dataset

def find_systems(root):
    """
    Every deepmd/npy system directory under `root`, sorted.

    A system is a directory holding type.raw and at least one set.* -- the walk does not
    descend into one once found.  Pointing this at a dataset that was already split gives
    back the systems of BOTH sides, which is what makes `train/` + `valid/` -> K folds a
    single pass rather than a merge step the caller has to do first.
    """
    root = os.path.abspath(root)
    if os.path.isfile(os.path.join(root, "type.raw")):
        return [root]
    found = []
    for cur, dirs, files in os.walk(root):
        if "type.raw" in files and any(d.startswith("set.") for d in dirs):
            found.append(cur)
            dirs[:] = []                      # a system holds no nested systems
        else:
            dirs.sort()
    return sorted(found)


# Directory names that mean "this is one side of a split", not "this is a system".
SPLIT_DIRS = {"train", "training", "valid", "validation", "test", "eval"}


def system_name(root, sysdir):
    """
    What to call a system once it is pooled: its path under its own dataset root, with a
    leading train/ or valid/ dropped.

    The basename alone is not enough.  A mixed-type dataset names its systems after the
    atom count, so `train/1_scfm/320` and `train/34_scfc_sbfc/320` are both called `320`
    while being different systems over different species -- pooling those under one name
    would mix them.  The path keeps them apart and keeps the tree recognisable in the
    output.

    The split component is dropped so that `train/1_scfm/320` and `valid/1_scfm/320` --
    two halves of one system, which is exactly what an existing split looks like -- come
    back together instead of being carried through as two systems that happen to hold the
    same species.
    """
    rel = os.path.relpath(os.path.abspath(sysdir), os.path.abspath(root))
    if rel in (".", ""):
        return os.path.basename(os.path.abspath(sysdir))
    parts = [p for p in rel.split(os.sep) if p not in (".",)]
    if len(parts) > 1 and parts[0].lower() in SPLIT_DIRS:
        parts = parts[1:]
    return "/".join(parts)


def load_system(sysdir):
    """
    One system -> (per-frame arrays concatenated over its sets, raw files, n_frames).

    Sets are read in sorted order, which is the order they were written in, so a frame's
    position here is the position it had in the dataset.  Every array in a set must agree
    on its first axis; that axis is the frame axis by definition of the format, and an
    array that disagrees means the system is corrupt rather than merely unfamiliar.
    """
    sets = sorted(d for d in os.listdir(sysdir)
                  if d.startswith("set.") and os.path.isdir(os.path.join(sysdir, d)))
    if not sets:
        raise ValueError(f"{sysdir}: no set.* directories")

    names, chunks, nframes = None, {}, 0
    for s in sets:
        sd = os.path.join(sysdir, s)
        here = sorted(f[:-4] for f in os.listdir(sd) if f.endswith(".npy"))
        if names is None:
            names = here
        elif here != names:
            raise ValueError(
                f"{sysdir}: {s} holds {here} but an earlier set holds {names}.  "
                f"The sets of one system have to carry the same arrays")
        n = None
        for name in names:
            a = np.load(os.path.join(sd, name + ".npy"))
            if n is None:
                n = a.shape[0]
            elif a.shape[0] != n:
                raise ValueError(f"{sd}: {name}.npy has {a.shape[0]} frames but "
                                 f"another array in the same set has {n}")
            chunks.setdefault(name, []).append(a)
        nframes += n

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    raws = {f: os.path.join(sysdir, f) for f in sorted(os.listdir(sysdir))
            if os.path.isfile(os.path.join(sysdir, f))}
    return arrays, raws, nframes


def read_type_map(raws):
    """type_map.raw -> the species list, or None if the system does not carry one."""
    p = raws.get("type_map.raw")
    if not p:
        return None
    with open(p) as fh:
        return [line.strip() for line in fh if line.strip()]


# ------------------------------------------------------------------------- group keys

def frame_groups(arrays, raws, how, sysname):
    """
    A grouping key per frame.  Frames sharing a key always land in the same fold.

    composition  the species multiset of the frame -- for this pipeline that IS the
                 compound, since every SQS realisation and every measurement of one
                 composition has the same atom counts.  The default, and the only one
                 that reproduces the by-compound split `clc delta` writes.
    structure    composition plus the coordinates and cell, so identical cells move
                 together but two realisations of one composition may separate.
    system       the whole system directory moves as one.
    frame        no grouping at all: plain K-fold.
    """
    n = len(next(iter(arrays.values())))
    if how == "frame":
        return [(sysname, i) for i in range(n)]
    if how == "system":
        return [sysname] * n

    type_map = read_type_map(raws)

    def comp_key(types):
        types = np.asarray(types)
        types = types[types >= 0]
        u, c = np.unique(types, return_counts=True)
        return tuple(sorted((type_map[t] if type_map and t < len(type_map) else int(t),
                             int(k)) for t, k in zip(u, c)))

    if "real_atom_types" in arrays:                       # deepmd/npy/mixed
        comps = [comp_key(arrays["real_atom_types"][i]) for i in range(n)]
    else:                                                 # plain deepmd/npy
        with open(raws["type.raw"]) as fh:
            types = [int(t) for t in fh.read().split()]
        comps = [comp_key(types)] * n

    if how == "composition":
        return comps
    if how == "structure":
        geom = [arrays[k] for k in ("coord", "box") if k in arrays]
        if not geom:
            raise ValueError(f"{sysname}: --group-by structure needs coord.npy")
        keys = []
        for i in range(n):
            h = hashlib.blake2b(digest_size=16)
            for a in geom:
                h.update(np.ascontiguousarray(np.round(a[i], 6)).tobytes())
            keys.append((comps[i], h.hexdigest()))
        return keys
    raise ValueError(f"unknown grouping {how!r}")


# ------------------------------------------------------------------------- writing out

def write_system(dest, arrays, raws, rows, set_size):
    """Write the frames `rows` of one system to `dest`, keeping every array it carries."""
    dest = os.path.join(*dest.split("/")) if os.sep != "/" else dest
    os.makedirs(dest, exist_ok=True)
    for name, src in raws.items():
        shutil.copyfile(src, os.path.join(dest, name))
    rows = np.asarray(rows, dtype=int)
    for i, start in enumerate(range(0, len(rows), set_size)):
        sd = os.path.join(dest, "set.%03d" % i)
        os.makedirs(sd, exist_ok=True)
        take = rows[start:start + set_size]
        for name, a in arrays.items():
            np.save(os.path.join(sd, name + ".npy"), a[take])
    return len(rows)


def fold_manifest(out, k, systems_of):
    """
    The train/valid systems lists for each of the K runs, as they go into folds.json.

    `systems_of` is {fold: [system directories written there]} and the directories are
    listed one by one rather than collapsed to `fold_k/*`.  A system's name can carry
    subdirectories -- a dataset laid out as <subset>/<atom count> puts its systems two
    levels down -- and there a one-star glob resolves to the subset directories, which
    hold no type.raw and are not systems at all.  Listing them is exact at any depth.
    """
    def dirs(i):
        return sorted(systems_of.get(i, []))

    return {f"fold_{i}": {
        "training_data": [p for j in range(k) if j != i for p in dirs(j)],
        "validation_data": dirs(i),
    } for i in range(k)}
