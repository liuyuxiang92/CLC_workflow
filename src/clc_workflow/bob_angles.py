"""
B-O-B bond angles from a POSCAR -- the octahedral-tilt observable.

Each bridging O is assigned the two nearest B-site cations within `rcut`; the angle
subtended at that O is 180 deg for an untilted cubic perovskite and falls as the
octahedra rotate.  It is the cheapest structural quantity that distinguishes a mean
MD structure from its 0 K relaxation, which is what stage 6.5 (`clc check-bob`) uses
it for: a relaxation that ran away shows up here as a tilt that moves by tens of
degrees, or as O that lost a B neighbour entirely, long before the phonons say so.

Minimum image is taken over the 27 neighbouring lattice translations, so triclinic
cells are handled; the search is a KD-tree over those images rather than an O(nB)
scan per O, because a 2560-atom cell has ~1500 O and ~500 B.

The B site is *not* hard-coded to (Fe, Co) -- config.yaml drives several families and
their B species come from the manifest's b_base / b_dopant columns.
"""
import numpy as np

__all__ = ["read_poscar", "bob_angles", "analyze_poscar", "angle_stats", "RCUT_DEFAULT"]

# B-O first shell.  Ideal B-O is ~1.93 A and the second shell sits beyond 3.5 A, so
# anything in 2.5-3.0 A gives the same neighbour list; 2.8 A also survives the ~0.15 A
# smearing that averaging an NPT trajectory leaves in the mean structure.
RCUT_DEFAULT = 2.8

_SHIFTS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                   dtype=float)


def read_poscar(path):
    """
    Minimal VASP POSCAR reader -> (cell, symbols, frac).

    Deliberately not ase.io.read: this runs over every structure in the tree from a
    driver that otherwise needs nothing but numpy/pandas, and ase's reader is the one
    import that would make stage 6.5 fail on a machine that only ran the collectors.
    Handles selective dynamics, Cartesian coordinates and a negative scale (VASP reads
    that as a target volume).
    """
    with open(path) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip() != ""]
    if len(lines) < 8:
        raise ValueError(f"{path}: truncated POSCAR ({len(lines)} non-blank lines)")

    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[2 + k].split()[:3]] for k in range(3)])
    if scale < 0:                       # negative scale = target volume, not a factor
        cell *= (abs(scale) / abs(np.linalg.det(cell))) ** (1.0 / 3.0)
    else:
        cell *= scale

    names = lines[5].split()
    if all(tok.lstrip("+-").isdigit() for tok in names):
        # VASP-4 POSCAR: counts on line 6 with no species line above them.  Nothing in
        # this pipeline writes one, and guessing the species would silently mislabel the
        # B site, so refuse instead.
        raise ValueError(f"{path}: no species-name line (VASP-4 format), cannot identify "
                         f"the B site")
    counts = [int(x) for x in lines[6].split()]

    mode = lines[7].strip()[0].lower()
    first = 8
    if mode in "sc":                    # selective dynamics; the real mode is one line down
        mode = lines[8].strip()[0].lower()
        first = 9
    n = sum(counts)
    if len(lines) < first + n:
        raise ValueError(f"{path}: header promises {n} atoms, file holds {len(lines) - first}")
    coords = np.array([[float(x) for x in lines[first + i].split()[:3]] for i in range(n)])
    if mode != "d":                     # 'c'/'k' = Cartesian
        coords = coords @ np.linalg.inv(cell)
    sym = np.array([s for s, c in zip(names, counts) for _ in range(c)])
    return cell, sym, coords % 1.0


def bob_angles(cell, sym, frac, bsite, rcut=RCUT_DEFAULT, kmax=12):
    """
    B-O-B angles for one structure.

    `bsite` is the collection of B-site element symbols (e.g. ("Fe", "Mn")).  Returns a
    dict with, per bridging O, the angle (deg), the sorted cation-pair label and the
    crystallographic axis the B...B bridge runs along; plus the per-O B coordination
    number and every B-O bond length inside `rcut`.

    An O with fewer than two B neighbours contributes no angle -- that is the signature
    of a relaxation that pulled an octahedron apart, so it is counted (`n_underco`)
    rather than dropped quietly.  Images of the *same* B atom are collapsed first, so a
    cell small enough for one B to bridge to itself cannot fake a 2-neighbour O.
    """
    from scipy.spatial import cKDTree

    cell = np.asarray(cell, dtype=float)
    isb = np.isin(sym, list(bsite))
    bfrac, bsym = frac[isb], sym[isb]
    ofrac = frac[sym == "O"]
    empty = dict(cell=cell, n_B=int(isb.sum()), n_O=len(ofrac),
                 angles=np.zeros(0), pairs=np.zeros(0, dtype="<U8"),
                 axis=np.zeros(0, dtype="<U1"), coord=np.zeros(0, dtype=int),
                 blen=np.zeros(0), bridge_blen=np.zeros(0),
                 n_underco=0, n_overco=0, n_truncated=0)
    if len(bfrac) == 0 or len(ofrac) == 0:
        return empty

    # 3x3x3 images of the B sublattice; parent[i] maps an image back to its atom
    bcart = bfrac @ cell
    images = (bcart[None, :, :] + (_SHIFTS @ cell)[:, None, :]).reshape(-1, 3)
    parent = np.tile(np.arange(len(bcart)), len(_SHIFTS))
    tree = cKDTree(images)

    ocart = ofrac @ cell
    k = min(kmax, len(images))
    dist, idx = tree.query(ocart, k=k, distance_upper_bound=rcut)
    inv_cell = np.linalg.inv(cell)

    angles, pairs, axes_, coord, blen, bridge_blen = [], [], [], [], [], []
    n_truncated = 0
    for o, dd, ii in zip(ocart, dist, idx):
        hit = np.isfinite(dd)
        if hit.all():
            n_truncated += 1            # kmax images inside rcut: neighbour list clipped
        # nearest image per parent atom, in increasing distance (dd is already sorted)
        seen, keep = set(), []
        for d_, i_ in zip(dd[hit], ii[hit]):
            p = parent[i_]
            if p in seen:
                continue
            seen.add(p)
            keep.append((d_, i_, p))
        coord.append(len(keep))
        blen += [d_ for d_, _, _ in keep]
        if len(keep) < 2:
            continue
        (d1, i1, p1), (d2, i2, p2) = keep[0], keep[1]
        v1, v2 = images[i1] - o, images[i2] - o
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angles.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
        pairs.append("-".join(sorted([bsym[p1], bsym[p2]])))
        bridge_blen += [d1, d2]
        # which cell axis the B...B bridge runs along, for tilt anisotropy
        bb = np.abs((v1 - v2) @ inv_cell)
        axes_.append("xyz"[int(np.argmax(bb))])

    coord = np.array(coord, dtype=int)
    return dict(cell=cell, n_B=int(isb.sum()), n_O=len(ofrac),
                angles=np.array(angles), pairs=np.array(pairs, dtype="<U8"),
                axis=np.array(axes_, dtype="<U1"), coord=coord,
                blen=np.array(blen), bridge_blen=np.array(bridge_blen),
                n_underco=int((coord < 2).sum()), n_overco=int((coord > 2).sum()),
                n_truncated=n_truncated)


def analyze_poscar(path, bsite, rcut=RCUT_DEFAULT):
    """read_poscar + bob_angles, with the file's symbol list kept for reporting."""
    cell, sym, frac = read_poscar(path)
    out = bob_angles(cell, sym, frac, bsite, rcut=rcut)
    out["sym"] = sym
    out["path"] = str(path)
    return out


def angle_stats(a, prefix=""):
    """
    Scalar summary of one angle distribution: N, mean, sigma, min/max, median and the
    mean tilt (180 - mean).  `p_lt_160` is the tail that matters -- a Ca-rich cell that
    has genuinely tilted sits well below 170 deg, while a broken octahedron shows up
    only in the minimum.
    """
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        keys = ["n", "mean", "std", "min", "max", "median", "p05", "p95", "tilt",
                "p_lt_160", "p_lt_150"]
        return {prefix + k: (0 if k == "n" else float("nan")) for k in keys}
    return {prefix + "n": int(a.size),
            prefix + "mean": float(a.mean()),
            prefix + "std": float(a.std()),
            prefix + "min": float(a.min()),
            prefix + "max": float(a.max()),
            prefix + "median": float(np.median(a)),
            prefix + "p05": float(np.percentile(a, 5)),
            prefix + "p95": float(np.percentile(a, 95)),
            prefix + "tilt": float(180.0 - a.mean()),
            prefix + "p_lt_160": float((a < 160).mean()),
            prefix + "p_lt_150": float((a < 150).mean())}
