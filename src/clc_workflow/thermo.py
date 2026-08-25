"""
Thermodynamics shared by the analysis stages.

Lifted from the corrected plot_heatmap.py so stage 8
(`clc collect`) and stage 9 (the plot script) share one implementation.
Behaviour must stay identical -- scratchpad/verify_plot_bugs.py plus the synthetic
known-answer tree are the regression suite.

Framework (from the source paper):

    ABO3 -> ABO(3-d) + (d/2) O2
    G  = H - T(S_vib + S_config) = E_DFT + F_phonopy + PV - T*S_config
    dG = dE_DFT + dF_phonopy - T*dS_config + dPV     (dPV neglected for the solid)
    S_config = |a| R [2d ln(2d) + (1-2d) ln(1-2d)]   with the sign that makes S >= 0

Note on scope: E and F come from one representative SQS structure while S_config comes
from this analytical mean-field model.  That is the standard representative-structure
approach; it would double-count only if combined with Boltzmann weighting over
realisations, which is why simple-mean averaging is the default.
"""
import math
import os
import numpy as np

KB = 8.617333262145e-5     # eV/K
KJ_TO_EV = 0.01036427      # eV per (kJ/mol);  R_KJ_PER_MOL_K * KJ_TO_EV == KB exactly
R_KJ_PER_MOL_K = 0.008314  # kJ/mol/K

__all__ = [
    "KB", "KJ_TO_EV", "R_KJ_PER_MOL_K",
    "S_config_of_delta", "mu_at_T", "parse_mu_table",
    "infer_cell_stoichiometry", "boltz_weights", "wmean_wstd",
    "delta_from_o_dirname", "o_tag_for_delta",
]


def S_config_of_delta(delta, n_fu, a_mag=2.0):
    """
    Configurational entropy of the WHOLE cell in eV/K, >= 0.

    Source: Wang et al., Energy Environ. Sci. 2022, 15, 1512 (DOI 10.1039/d1ee02889h)
    section 4.1, which prints  dS_conf = a R [2d ln(2d) + (1-2d) ln(1-2d)]  with a = 2
    ("the factor referring to the interaction of oxygen vacancies, a = 2 describing an
    ideal solid solution with no defect interaction") and dG = dU - T(dS_vib+dS_conf).

    The bracket is <= 0 on 0 < 2d < 1, so a = +2 as printed would make S_conf negative
    -- entropy decreasing with disorder, dG rising with delta, and ordered
    brownmillerite (d=0.5) more disordered than d=0.25.  The printed equation is
    missing its leading minus (the standard form is -aR[...]), so we use a = -|a_mag|
    with |a_mag| = 2 as the paper specifies.

    Extensive over the cell via n_fu: E_DFT, F_phonopy and mu are all bookkept per
    cell, so S_config must be too.  R_KJ_PER_MOL_K * KJ_TO_EV is already k_B, so no
    further scaling is applied.
    """
    if delta is None or n_fu is None:
        return None
    d = float(delta)
    if d <= 0.0:
        return 0.0
    if d >= 0.5:
        d = 0.499999            # paper's domain: x = 2d must stay below 1
    x = 2.0 * d
    term = x * math.log(x) + (1.0 - x) * math.log(1.0 - x)     # <= 0
    return -abs(a_mag) * R_KJ_PER_MOL_K * float(n_fu) * term * KJ_TO_EV


def mu_at_T(table, T, name="mu", strict=True):
    """Linear interpolation of a {T: value} table.  Raises outside its range if strict."""
    if not table:
        raise ValueError(f"{name}: empty table")
    Ts = sorted(table)
    if len(Ts) == 1:
        if strict and abs(Ts[0] - T) > 1e-9:
            raise ValueError(f"{name}: only tabulated at {Ts[0]} K, cannot supply T={T} K")
        return float(table[Ts[0]])
    if T < Ts[0] - 1e-9 or T > Ts[-1] + 1e-9:
        if strict:
            raise ValueError(
                f"{name}: T={T} K is outside the tabulated range {Ts[0]}-{Ts[-1]} K. "
                f"Extend the table or restrict the temperature list."
            )
    return float(np.interp(T, Ts, [table[t] for t in Ts]))


def parse_mu_table(spec, scale=1.0):
    """Parse 'T:value,T:value' (or a dict) into {T(float): value*scale}."""
    if isinstance(spec, dict):
        return {float(k): float(v) * scale for k, v in spec.items()}
    table = {}
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k, v = chunk.split(":")
        table[float(k.strip())] = float(v.strip()) * scale
    if not table:
        raise ValueError(f"could not parse mu table from {spec!r}")
    return table


# ---------------------------------------------------------------- cell stoichiometry
# POSCAR_md_final is what stage 4 wrote before it averaged; kept for older trees.
_STRUCTURE_CANDIDATES = ("optimized_POSCAR", "CONTCAR", "POSCAR_md_avg",
                         "POSCAR_md_final", "POSCAR", "unconverged_POSCAR")


def _read_poscar_species_counts(path):
    try:
        with open(path, "r") as f:
            lines = [next(f) for _ in range(8)]
        species = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        if len(species) != len(counts) or not species:
            return None, None
        return species, counts
    except Exception:
        return None, None


def infer_cell_stoichiometry(o_dir, anion="O", candidates=None):
    """
    Infer (n_fu, n_O, nv) for an ABO(3-d) cell from whichever structure file is present.

        n_fu = formula units = n_A cations = n_B cations = (total cations) / 2
        nv   = 3*n_fu - n_O

    Returns (None, None, None) if nothing readable was found.
    """
    for cand in (candidates or _STRUCTURE_CANDIDATES):
        p = os.path.join(str(o_dir), cand)
        if not os.path.isfile(p):
            continue
        species, counts = _read_poscar_species_counts(p)
        if species is None:
            continue
        n_O = sum(c for s, c in zip(species, counts) if s == anion)
        n_cat = sum(c for s, c in zip(species, counts) if s != anion)
        if n_cat <= 0 or (n_cat % 2) != 0:
            continue
        n_fu = n_cat // 2
        return n_fu, n_O, 3 * n_fu - n_O
    return None, None, None


# ---------------------------------------------------------------- weighted stats
def boltz_weights(G_list, T):
    """Boltzmann weights from G_state at T.  Returns (w, mask, n_eff)."""
    x = np.asarray(G_list, dtype=float)
    mask = np.isfinite(x)
    if not mask.any():
        return np.zeros_like(x), mask, np.nan
    xv = x[mask]
    beta = 1.0 / (KB * float(T))
    w = np.exp(-beta * (xv - np.min(xv)))
    sw = w.sum()
    w = w / sw if sw > 0 else np.ones_like(xv) / len(xv)
    return w, mask, 1.0 / np.sum(w**2)


def wmean_wstd(x, w):
    """Weighted mean and finite-sample-corrected weighted std."""
    x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
    if x.size == 0 or w.size == 0:
        return np.nan, np.nan
    m = np.sum(w * x)
    sumw2 = np.sum(w**2)
    num = np.sum(w * (x - m)**2)
    var = num / (1.0 - sumw2) if (1.0 - sumw2) > 1e-12 else num
    return m, math.sqrt(max(var, 0.0))


# ---------------------------------------------------------------- o-tag helpers
def o_tag_for_delta(delta):
    """delta -> 'o####' label, i.e. round(1000*(1-delta))."""
    return f"o{int(round(1000.0 * (1.0 - float(delta)))):03d}"


def delta_from_o_dirname(name):
    """
    'o####' -> delta.  NOTE this is lossy: nv=1 in a 64 f.u. cell is delta=0.015625
    but tags as o984 -> 0.016 (2.4% off).  Prefer delta = nv/n_fu whenever nv is known;
    use this only as a fallback or for labelling.
    """
    import re
    m = re.fullmatch(r"o(\d{3,4})", str(name))
    if not m:
        return None
    v = int(m.group(1))
    if v < 0 or v > 1000:
        return None
    return (1000 - v) / 1000.0
