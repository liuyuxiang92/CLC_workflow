"""
Phonon-side helpers: phonopy conversion, band.yaml stability verdict, and the
safe-interatomic-distance screen.

One home for what used to be duplicated across predict_energy.py,
get_stability_file.py and parse_traj.py.  The distance screen keeps the original
thresholds and verdict but replaces the O(N^2) pure-Python double loop over a 2x2x2
replication (2560 atoms ~ 3.3M pairs per structure) with a KD-tree query.
"""
import re
import numpy as np

__all__ = ["safe_dist_dict", "ase_to_phonopy_atoms",
           "analyze_phonon_frequencies_pure", "min_pair_violation", "is_too_close"]

# Safe interatomic distances (the original table, verbatim).  Used as
# (r_i + r_j) * 0.529 -- kept exactly as the legacy code applied it.
safe_dist_dict = {
    "H": 0.510, "He": 0.509, "Li": 1.224, "Be": 1.061, "B": 0.897, "C": 0.897, "N": 0.897,
    "O": 0.897, "F": 0.856, "Ne": 1.305, "Na": 1.183, "Mg": 1.387, "Al": 1.387, "Si": 1.305,
    "P": 1.224, "S": 1.224, "Cl": 1.224, "Ar": 1.55, "K": 1.877, "Ca": 1.877, "Sc": 1.632,
    "Ti": 1.632, "V": 1.632, "Cr": 1.55, "Mn": 1.591, "Fe": 1.55, "Co": 1.55, "Ni": 1.55,
    "Cu": 1.55, "Zn": 1.55, "Ga": 1.632, "Ge": 1.632, "As": 1.632, "Se": 1.713, "Br": 1.713,
    "Kr": 1.877, "Rb": 2.04, "Sr": 2.04, "Y": 1.713, "Zr": 1.713, "Nb": 1.713, "Mo": 1.713,
    "Tc": 1.713, "Ru": 1.713, "Rh": 1.713, "Pd": 1.713, "Ag": 1.713, "Cd": 1.713, "In": 1.632,
    "Sn": 1.632, "Sb": 1.632, "Te": 1.632, "I": 1.632, "Xe": 1.632, "Cs": 2.04, "Ba": 2.285,
    "La": 2.04, "Ce": 2.081, "Pr": 2.203, "Nd": 2.285, "Pm": 2.285, "Sm": 2.285, "Eu": 2.285,
    "Gd": 2.285, "Tb": 2.285, "Dy": 2.285, "Ho": 2.285, "Er": 2.122, "Tm": 2.285, "Yb": 2.285,
    "Lu": 2.285, "Hf": 1.958, "Ta": 2.04, "W": 1.877, "Re": 1.877, "Os": 1.877, "Ir": 1.877,
    "Pt": 1.877, "Au": 1.877, "Hg": 1.877, "Tl": 1.877, "Pb": 1.877, "Bi": 1.877, "Ac": 2.366,
    "Th": 2.285, "Pa": 2.285, "U": 2.285, "Np": 2.285, "Pu": 2.285,
}
_BOHR = 0.529


def ase_to_phonopy_atoms(atoms):
    """ASE Atoms -> PhonopyAtoms, with masses passed explicitly."""
    from phonopy.structure.atoms import PhonopyAtoms
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        masses=list(atoms.get_masses()),
    )


def analyze_phonon_frequencies_pure(yaml_file="band.yaml"):
    """
    Stability verdict from a phonopy band.yaml, unchanged from the original:
    A = |min frequency|, B = 0.03 * max frequency; imaginary if A > B.

    Returns (output_str, stability_str) where output_str is "A B" to 4 decimals.
    """
    try:
        with open(yaml_file, "r") as f:
            content = f.read()
        frequencies = re.findall(r"frequency:\s*(-?\d+\.\d+)", content)
        if not frequencies:
            return "0.0000 0.0000", "error"
        freq = [float(x) for x in frequencies]
        A = abs(min(freq))
        B = max(freq) * 0.03
        output = f"{A:.4f} {B:.4f}"
        if output == "0.0000 0.0000":
            return output, "error"
        return output, ("imaginary frequency" if A > B else "stable")
    except Exception as e:
        print(f"band.yaml reading error: {e}")
        return "0.0000 0.0000", "error"


def min_pair_violation(cell, cart_coords, symbols):
    """
    Find the worst violation of the safe-distance table under periodic boundaries.

    Returns (violated: bool, detail: str).  Same thresholds and verdict as the legacy
    is_too_close(), but O(N log N) via a KD-tree over a 3x3x3 image expansion instead
    of an O(N^2) Python loop over a 2x2x2 replication.
    """
    from scipy.spatial import cKDTree

    cell = np.asarray(cell, dtype=float)
    pos = np.asarray(cart_coords, dtype=float)
    n = len(pos)
    if n < 2:
        return False, ""

    uniq = sorted(set(symbols))
    unknown = [s for s in uniq if s not in safe_dist_dict]
    if unknown:
        raise KeyError(f"no safe distance for {unknown}")
    r = np.array([safe_dist_dict[s] * _BOHR for s in symbols], dtype=float)
    cutoff = 2.0 * max(safe_dist_dict[s] * _BOHR for s in uniq)

    # 3x3x3 periodic images so any pair closer than `cutoff` is captured
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                      dtype=float) @ cell
    images = (pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    img_index = np.tile(np.arange(n), len(shifts))
    is_home = np.repeat(np.all(shifts == 0.0, axis=1), n)

    tree = cKDTree(images)
    home = np.flatnonzero(is_home)
    worst = None
    for a in home:
        ia = img_index[a]
        for b in tree.query_ball_point(images[a], cutoff):
            if b == a:
                continue
            ib = img_index[b]
            if is_home[b] and ib <= ia:
                continue                      # count each home-home pair once
            d = float(np.linalg.norm(images[a] - images[b]))
            thr = r[ia] + r[ib]
            if d < thr and (worst is None or (thr - d) > worst[0]):
                worst = (thr - d, symbols[ia], symbols[ib], d, thr)
    if worst is None:
        return False, ""
    _, si, sj, d, thr = worst
    return True, f"dangerous close for {si} - {sj}, {d:.5f} less than {thr:.5f}"


def is_too_close(cell, cart_coords, symbols, verbose=True):
    """Boolean wrapper matching the legacy call site."""
    violated, detail = min_pair_violation(cell, cart_coords, symbols)
    if violated and verbose and detail:
        print(detail)
    return violated
