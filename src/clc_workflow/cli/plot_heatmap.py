#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 9 -- ΔG(T, pO2) for oxygen vacancy formation, from the per-set trees to one
table of "how many oxygens does this composition give up".

    clc plot --parent . --series-glob 'set_*'

WHAT COMES OUT

  dG0_vacancy_summary.csv   <- the answer.  One row per (composition, temperature):

      family, composition, formula, Temperature_K, n_formula_units, A_frac, B_frac,
      x_1atm, x_0p8atm, x_3e-05atm, x_swing, n_sets_used, n_sets_total, R2

    x_<p> is the ΔG = 0 crossing as the OXYGEN VACANCY FRACTION x = n_v/n_fu at that
    oxygen partial pressure -- the same axis the ΔG line plots use, so a number here
    reads straight off the corresponding figure.  **A negative crossing is written as
    0**: it means ΔG > 0 already at the first vacancy, i.e. no oxygen is released at
    all, and there is no such thing as a negative vacancy fraction.  x_swing is the
    extra fraction unlocked by going from the highest listed pressure to the lowest --
    the usable oxygen capacity.  n_formula_units is carried for anyone who wants a
    count instead: multiply by it.

  state_boltz_bavg.csv      the per-set average that feeds it, carrying n_series_used
  state_series_raw.csv      the raw per-set state energies

  Everything else (per-pair ΔG, regression predictions, the per-pressure fit dump, the
  S_config sign study) is diagnostic and needs --full-csv.

THREE THINGS IT WORKS OUT FOR ITSELF

  1. The elements.  The four symbols are read off each composition directory name
     ('Sr875_Ca125_Fe875_Mn125' -> Sr/Ca on A, Fe/Mn on B), so no --elements is needed
     and several families can share one tree -- each gets its own plots with its own
     axis labels.  --elements survives only as a fallback for names that will not parse.
  2. Where the data is.  o* directories are found by their own name at any depth, so
     both <set>/<group>/<comp>/o#### and flatter layouts are read as-is.
  3. How many sets there really are.  See below.

FAILED CALCULATIONS

  The average at each (composition, o, T) is taken over the sets that produced a usable
  G_state, NOT over the number launched: 50 sets with one dead directory average over
  49.  Every averaged row records n_series_used / n_series_total and names the sets that
  dropped out, and the run prints the points that came up short, so a thin average is
  visible instead of silently widening the error bar.  E, H and G share one mask, so the
  three always describe the same subset of realisations.  --complete-sets-only instead
  restricts each composition to the sets that succeeded across its whole o-series, which
  makes every point of one ΔG fit come from the same realisations.

ΔE and ΔH are off by default (--with-EH), since ΔG = 0 is what this pipeline reports.

Full chain:
- Read multi-series set_* results → state-level E/H/G for each (comp_dir, o, T)
- Average across series at fixed (comp_dir, o, T): simple mean (default) or Boltzmann-weighted
- Build adjacent-pair deltas ΔG (+ μ corrections, ΔE/ΔH with --with-EH) with std propagation
- Line plots (G/H) against the NUMERIC vacancy fraction x = nv_lo/n_fu, i.e. the pair
  o984-o1000 plots at 1/64, o968-o984 at 2/64, ... o516-o531 at 31/64 -- not the
  'o_lo - o_hi' string it used to be
- Heatmaps are OPT-IN (--heatmaps); crossing heatmaps stay opt-in (--crossing-heatmaps)
- Continuous-threshold of ΔG=0 via OLS on ΔG ~ x.  Default x-mode is 'nv', so x0 is
  directly the vacancy fraction (and x0*n_fu the vacancy count) at which ΔG changes sign
- Oxygen partial-pressure shift, paper eq. (1):
      ΔG_vac(T, pO2) = ΔG_vac^DFT(T, P0) + (1/2) kB T ln(pO2/P0),   P0 = 1 atm
  Reproduces the paper's table exactly: -0.00644 / -0.30064 eV per vacancy at 670 K for
  0.8 / 0.00003 atm, and -0.01029 / -0.48013 eV at 1070 K.  x0 is reported at 1 atm and
  at every --pO2-list value, together with the x0 difference between the two pressures
  named by --pO2-diff (default 0.00003 vs 0.8 atm) -- the usable vacancy swing.
- The S_config SIGN comparison (a<0 corrected / a>0 as printed / a=0 omitted) is run side
  by side through the whole chain, so its influence on x0 is quantified rather than argued
- Optional fit-cut: keep only pairs with hi >= --fit-hi-min in regressions/thresholds

Thermodynamic framework (per the source paper):

    ABO3 -> ABO(3-δ) + (δ/2) O2
    G  = H - T(S_vib + S_config) = E_DFT + F_phonopy + PV - T*S_config
    ΔG = ΔE_DFT + ΔF_phonopy - T*ΔS_config + ΔPV        (ΔPV neglected for the solid)
    S_config = |a| R [2δ ln(2δ) + (1-2δ) ln(1-2δ)]  with the sign that makes S >= 0

CORRECTIONS APPLIED (see S_config_of_delta and build_pair_deltas):
  1. S_config sign.  The source paper (Wang et al., Energy Environ. Sci. 2022, 15,
     1512, DOI 10.1039/d1ee02889h, section 4.1) prints
         dS_conf = a R [2d ln(2d) + (1-2d) ln(1-2d)],   a = 2
     and defines dG = dU - T(dS_vib + dS_conf).  The bracket is <= 0 on 0<2d<1, so
     with a = +2 as printed S_conf would be NEGATIVE: zero at d=0, -11.53 J/mol/K/f.u.
     at d=0.25, and zero again at d=0.5.  That has vacancy disorder *lowering* the
     entropy and *raising* dG, and makes ordered brownmillerite (d=0.5) more
     disordered than d=0.25.  The printed equation is missing its leading minus; the
     standard form is -aR[...].  The default is therefore a = -2, which gives
     +11.53 J/mol/K/f.u. at d=0.25, peaking at d=0.25 and vanishing at both
     ordered end members.
     --s-config-a is SIGNED and used verbatim, so `-2` and `+2` are two different
     runs rather than two spellings of the same one: pass +2 to reproduce the
     paper's printed branch and see what the missing minus costs.  (It used to
     take a magnitude and apply -|a|, which silently turned `+2` and `-2` into
     identical output.)
  2. S_config unit scale.  R[kJ/mol/K] * KJ_TO_EV is already k_B in eV/K, so the
     old trailing "* 0.001" was a spurious factor of 1000 that reduced the whole
     term to sub-meV (i.e. effectively removed it).  Removed.
  3. S_config extensivity.  E_DFT and F_phonopy are whole-cell quantities and μ is
     bookkept per cell, so S_config must be extensive over the same cell.  The old
     hardcoded "0.008314 * 8" is R times 8 formula units, which is correct for the
     PAPER's cell (its DFT used 2x2x2 perovskite supercells = 8 f.u. = 40 atoms) but
     wrong for these datasets: the 320-atom SQS cells are 64 f.u. and the older
     2560-atom cells are 512 f.u.  n_fu is now read from the structure file in each
     o* directory (n_fu = n_cations/2), with --n-fu as an override.
  4. μ(T).  Previously muG_base.get(T, muG_base[670]) silently substituted the
     670 K value at any other temperature.  Now linearly interpolated in T, with
     an explicit error outside the tabulated range.
  5. μ per pair.  One adjacent step releases ½O2, but if an o* directory is missing
     the pair spans Δnv > 1 while still receiving a single μ.  μ is now multiplied
     by the pair's actual Δnv, and Δnv is reported in the CSVs.
  6. n_eff.  The Boltzmann effective sample size was computed and discarded; it is
     now written to state_boltz_bavg.csv so a collapsed average (n_eff → 1, i.e.
     "the minimum" rather than "the average") is visible.  --avg-mode now defaults
     to 'simple'.
  7. Element names are configurable via --elements, so axis labels and composition
     parsing are correct for families other than Sr/Ca-Fe/Mn (e.g. Sr/Ca-Fe/Ti).
  8. add_regressed_pair_values vectorised (was a full-frame scan per pair).

Saved CSVs (default):
  - dG0_vacancy_summary.csv            # THE RESULT: vacancy fraction at ΔG=0 per pO2
  - state_series_raw.csv               # per-series raw state-level E/F/H/G (before averaging)
  - state_boltz_bavg.csv               # averaged (E/H/G)_bavg + stds + n_series_used

Saved CSVs (--full-csv):
  - pair_deltas_from_state_bavg.csv    # per-pair ΔG (+std propagation, +dnv, +x)
  - pair_deltas_regression_pred.csv    # per-pair regression predictions
  - thresholds_DG_pO2.csv              # x0 per (S_config variant, T, composition, pO2)
  - thresholds_DG_pO2_summary.csv      # x0 at each pO2 + the 0.00003-vs-0.8 atm difference
  - filtered_pairs_*_dGlt0_dHgt0.csv   # ΔG<0 & ΔH>0 filter dump  (needs --with-EH)
  - s_config_sign_comparison.csv       # x0 for a<0 / a>0 / a=0    (needs --sign-compare)
  - thresholds_DG_continuous.csv       # legacy x-mode threshold dump (only with --thresholds-csv)
"""

import os, re, math, argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', message='Format strings passed to MaskedConstant')

KB = 8.617333262145e-5     # eV/K
KJ_TO_EV = 0.01036427      # eV per (kJ/mol);  0.008314 * KJ_TO_EV == KB
R_KJ_PER_MOL_K = 0.008314  # kJ/mol/K

# ---------- YAML loader ----------
try:
    from yaml import CLoader as _YamlLoader
except Exception:
    from yaml import SafeLoader as _YamlLoader

# ---------- Basic I/O ----------

def read_energy_txt(p):
    try:
        with open(p, "r") as f:
            s = f.read().strip()
            return float(s.split()[0]) if s else None
    except Exception:
        return None

def extract_dft_energy_from_outcar(p):
    try:
        with open(p, 'r') as f:
            for line in f:
                if "energy(sigma->0)" in line:
                    return float(line.split()[-1])
    except Exception:
        return None
    return None

def extract_thermal_properties(p, temperatures, tol=1e-8):
    out = {}
    try:
        with open(p, 'r') as f:
            data = yaml.load(f, Loader=_YamlLoader)
        tps = data.get("thermal_properties", []) if isinstance(data, dict) else []
        byT = {round(tp.get("temperature", None), 8): tp for tp in tps if "temperature" in tp}
        for T in temperatures:
            m = None
            for Tk, entry in byT.items():
                if Tk is None: continue
                if abs(Tk - T) <= tol:
                    m = entry; break
            if m:
                F = m['free_energy'] * KJ_TO_EV
                H = m['energy']      * KJ_TO_EV
                S = m['entropy'] * 1e-3 * KJ_TO_EV   # J/K/mol -> kJ/K/mol -> eV/K
                out[T] = {"F (eV)": F, "H (eV)": H, "S_vib (eV/K)": S}
    except Exception:
        pass
    return out

# ---------- Cell stoichiometry (n_fu) ----------
_STRUCTURE_CANDIDATES = ("optimized_POSCAR", "CONTCAR", "POSCAR", "unconverged_POSCAR")

def _read_poscar_species_counts(path):
    """Return (species, counts) from a VASP POSCAR-format file, or (None, None)."""
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

def infer_cell_stoichiometry(o_dir, anion="O"):
    """
    Infer (n_fu, n_O, nv) for an ABO(3-δ) cell from whichever structure file is present.

    n_fu = number of formula units = number of A cations = number of B cations
         = (total cations) / 2
    nv   = 3*n_fu - n_O   (oxygen vacancies in the cell)

    Returns (None, None, None) if no structure file could be read.
    """
    for cand in _STRUCTURE_CANDIDATES:
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

# ---------- Thermodynamics ----------

def S_config_signed(delta, n_fu, a_signed):
    """
    Configurational entropy of the WHOLE cell in eV/K, using `a_signed` VERBATIM.

    Paper form (per formula unit):
        ΔS_config = a R [2δ ln(2δ) + (1-2δ) ln(1-2δ)]

    The bracket is <= 0 for 0 < 2δ < 1, so the sign of `a` decides the sign of S_config.
    Only a < 0 is physical (see S_config_of_delta); a > 0 and a = 0 are provided so the
    sign's influence can be measured instead of asserted.

    Extensive over the cell via n_fu, because E_DFT, F_phonopy and μ are all bookkept
    per cell.  R_KJ_PER_MOL_K * KJ_TO_EV == k_B, so no extra scaling.
    """
    if delta is None or n_fu is None:
        return None
    if not np.isfinite(float(delta)) or not np.isfinite(float(n_fu)):
        return None
    if a_signed == 0.0:
        return 0.0
    d = float(delta)
    if d <= 0.0:
        return 0.0
    if d >= 0.5:
        d = 0.499999            # paper's domain: x = 2δ must stay below 1
    x = 2.0 * d
    term = x * math.log(x) + (1.0 - x) * math.log(1.0 - x)     # <= 0
    return float(a_signed) * R_KJ_PER_MOL_K * float(n_fu) * term * KJ_TO_EV

def S_config_of_delta(delta, n_fu, a_mag=2.0):
    """
    Configurational entropy of the WHOLE cell in eV/K, >= 0 (the physical branch).

    Takes a MAGNITUDE and forces the physical sign: a = -|a_mag|.  Stage 8
    (`clc collect`) reads its |a| from config.yaml and calls this, so the
    magnitude semantics are kept here.  Stage 9 goes through S_config_signed
    instead, because its --s-config-a is signed -- see s_config_variants.
    """
    return S_config_signed(delta, n_fu, -abs(a_mag))


def s_config_label(a_signed):
    """
    The variant name that matches a signed `a`, so a label can never contradict
    the number it describes ('corrected_a_neg' on a positive a, say).
    """
    a = float(a_signed)
    if a == 0.0:
        return "omitted_a_zero"
    return "corrected_a_neg" if a < 0 else "as_printed_a_pos"


# S_config variants used by the sign comparison: (label, signed a factor)

def s_config_variants(a_signed, enabled=True):
    """
    The branches to run, primary first.

    `a_signed` is taken VERBATIM, so -2 and +2 are genuinely different runs:
    -2 is the physical branch (S_config >= 0, disorder lowers G) and +2 is the
    sign as printed in the paper.  With the sign study on, the other branches at
    the same magnitude follow, de-duplicated against the primary so a=+2 does
    not get compared against itself.
    """
    a = float(a_signed)
    primary = (s_config_label(a), a)
    if not enabled:
        return [primary]
    mag = abs(a)
    others = [(s_config_label(v), v) for v in (-mag, +mag, 0.0)]
    return [primary] + [o for o in others if o[0] != primary[0]]

def dG_pO2_shift(T, p_atm, p_ref_atm=1.0):
    """
    Oxygen partial-pressure correction PER VACANCY, in eV.  Paper eq. (1):

        ΔG_vac(T, pO2) = ΔG_vac^DFT(T, P0) + (1/2) kB T ln(pO2/P0)

    Reproduces the paper's tabulated corrections exactly:
        670 K:  0.8 atm -> -0.00644 eV,  0.00003 atm -> -0.30064 eV
       1070 K:  0.8 atm -> -0.01029 eV,  0.00003 atm -> -0.48013 eV
    """
    return 0.5 * KB * float(T) * math.log(float(p_atm) / float(p_ref_atm))

def fmt_pressure(p):
    """Compact, filename-safe pressure tag, e.g. 1atm / 0p8atm / 3e-05atm."""
    s = f"{float(p):g}"
    return s.replace(".", "p").replace("+", "") + "atm"

def delta_from_o_dirname(s: str):
    m = re.fullmatch(r"o(\d{3,4})", s)
    if not m: return None
    v = int(m.group(1))
    if v < 0 or v > 1000: return None
    return (1000 - v) / 1000.0

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
                f"Extend --mu-g/--mu-h or restrict --temps (the previous code silently "
                f"substituted the {Ts[0]} K value here)."
            )
    return float(np.interp(T, Ts, [table[t] for t in Ts]))

def parse_mu_table(spec, scale=1.0):
    """Parse 'T:value,T:value' into {T(float): value*scale}."""
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

# ---------- Composition parsing (any A_base/A_dop - B_base/B_dop family) ----------
# 'Sr875' -> ('Sr', 875).  Letters are greedy so the legacy 'None000' placeholder that
# old trees use for an absent dopant parses too.
_ELEM_TAG = re.compile(r"^([A-Za-z]+)(\d{1,4})$")
_VAC_TAG = re.compile(r"^v([AB])(\d{1,4})$")


def parse_comp_dir(comp_dir):
    """
    Read a composition directory name with NO prior knowledge of the family.

    'Sr875_Ca125_Fe875_Mn125'      -> Sr/Ca on A, Fe/Mn on B
    'Sr750_Ba250_vA050_Fe_Co125'   -> the same, with a 5% A-site cation vacancy
    'Sr1000_Ca000_Fe875_Ti125'     -> a different family in the same tree

    stage 1 (sqs_generator.build_directory_name) writes
    <A_base###>_<A_dop###>[_vA###]_<B_base###>_<B_dop###>[_vB###], so the four element
    tags in order ARE (A_base, A_dopant, B_base, B_dopant).  Reading them off the name
    is what lets one run cover several families at once -- the old --elements was a
    single global guess, and it silently mis-parsed every directory it did not match.

    Returns a dict, or None if the name is not a composition directory.
    """
    name = os.path.basename(str(comp_dir))
    tags, vac = [], {"A": 0.0, "B": 0.0}
    for tok in name.split("_"):
        mv = _VAC_TAG.match(tok)
        if mv:
            vac[mv.group(1)] = int(mv.group(2)) / 1000.0
            continue
        me = _ELEM_TAG.match(tok)
        if me:
            tags.append((me.group(1), int(me.group(2)) / 1000.0))
    if len(tags) != 4:
        return None                      # group dirs ('Sr875_Ca125_Fe_Mn') land here

    (a_base, a_f), (a_dop, ad_f), (b_base, b_f), (b_dop, bd_f) = tags
    return {
        "elements": (a_base, a_dop, b_base, b_dop),
        "a_base": a_base, "a_dopant": a_dop, "b_base": b_base, "b_dopant": b_dop,
        "A_frac": a_f, "B_frac": b_f,
        "a_dop_frac": ad_f, "b_dop_frac": bd_f,
        "a_vac": vac["A"], "b_vac": vac["B"],
        # family key: what varies inside a family is x and y, so the symbols alone
        # identify the (A_base/A_dop, B_base/B_dop) system the plots group by
        "family": f"{a_base}{a_dop}_{b_base}{b_dop}",
        "formula": (f"{a_base}{a_f:g}{a_dop}{ad_f:g}"
                    f"{b_base}{b_f:g}{b_dop}{bd_f:g}O3-d"),
    }


def detect_elements(comp_dirs, fallback=None):
    """
    Family -> (A_base, A_dopant, B_base, B_dopant), auto-detected from directory names.

    `fallback` (the --elements override) is used only for names that cannot be parsed,
    and those are reported by the caller rather than absorbed.
    """
    fams = {}
    for c in comp_dirs:
        info = parse_comp_dir(c)
        if info:
            fams.setdefault(info["family"], info["elements"])
    if not fams and fallback:
        a, b, c_, d = fallback
        fams[f"{a}{b}_{c_}{d}"] = tuple(fallback)
    return fams


def parse_composition(comp_dir, elements=("Sr", "Ca", "Fe", "Mn")):
    """
    Return (A_base_fraction, B_base_fraction) from a directory name such as
    'Sr875_Ca125_Fe875_Mn125', for the family given by
    elements = (a_base, a_dopant, b_base, b_dopant).

    Kept for the explicit --elements override; parse_comp_dir is what the pipeline
    normally uses, because it needs no family to be named up front.

    Fractions are the ###/1000 tags in the name.  A missing tag on one site is
    filled in as 1 - (the other tag on that site).
    """
    a_base, a_dop, b_base, b_dop = elements
    name = os.path.basename(str(comp_dir))
    vals = {}
    for el in (a_base, a_dop, b_base, b_dop):
        if not el:
            vals[el] = None
            continue
        # element symbol not preceded by a letter, not followed by a lowercase
        # letter (so 'C' does not match inside 'Ca'), then its 1-4 digit tag
        m = re.search(rf'(?<![A-Za-z]){re.escape(el)}(?![a-z])(\d{{1,4}})(?![0-9])', name)
        vals[el] = int(m.group(1)) / 1000.0 if m else None

    if vals.get(a_base) is None and vals.get(a_dop) is not None:
        vals[a_base] = 1.0 - vals[a_dop]
    if vals.get(a_dop) is None and vals.get(a_base) is not None:
        vals[a_dop] = 1.0 - vals[a_base]
    if vals.get(b_base) is None and vals.get(b_dop) is not None:
        vals[b_base] = 1.0 - vals[b_dop]
    if vals.get(b_dop) is None and vals.get(b_base) is not None:
        vals[b_dop] = 1.0 - vals[b_base]

    if vals.get(a_base) is None or vals.get(b_base) is None:
        return None, None
    return vals[a_base], vals[b_base]

# ---------- Weighted stats ----------

def boltz_weights(G_list, T):
    """Boltzmann weights from G_state at temperature T.  Returns (w, mask, n_eff)."""
    x = np.asarray(G_list, dtype=float)
    mask = np.isfinite(x)
    if not mask.any():
        return np.zeros_like(x), mask, np.nan
    xv = x[mask]
    beta = 1.0 / (KB * float(T))
    m = np.min(xv)
    w = np.exp(-beta*(xv - m))
    sw = w.sum()
    w = w/sw if sw > 0 else np.ones_like(xv)/len(xv)
    n_eff = 1.0 / np.sum(w**2)
    return w, mask, n_eff

def wmean_wstd(x, w):
    """Weighted mean & finite-sample-corrected weighted std."""
    x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
    if x.size == 0 or w.size == 0: return np.nan, np.nan
    m = np.sum(w*x)
    sumw2 = np.sum(w**2)
    num = np.sum(w*(x-m)**2)
    var = num/(1.0 - sumw2) if (1.0 - sumw2) > 1e-12 else num
    return m, math.sqrt(max(var, 0.0))

# ---------- Pair helpers ----------
_PAIR_RE = re.compile(r"o(\d{3,4})\s*-\s*o(\d{3,4})")

def pair_to_lo_hi(pair_label: str):
    m = _PAIR_RE.fullmatch(str(pair_label).strip())
    if not m:
        return np.nan, np.nan
    a, b = int(m.group(1)), int(m.group(2))
    lo, hi = (a, b) if a < b else (b, a)
    return lo, hi

def pair_sort_key(pair_label: str):
    """Ascending sort by this key == (hi desc, lo desc), i.e. fewest vacancies first."""
    lo, hi = pair_to_lo_hi(pair_label)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return (1, 0.0, 0.0)
    return (0, -hi, -lo)

def order_pairs_unique(pairs):
    """Canonical order: sort by hi desc, then lo desc; keep original label."""
    keyed = [p for p in pairs if np.isfinite(pair_to_lo_hi(p)[0])]
    return sorted(keyed, key=pair_sort_key)

def pair_to_x(pair_label: str, x_mode: str = "hi"):
    """
    Tag-derived x for the legacy x-modes.  NOTE: x_mode 'nv' (the default everywhere
    else) cannot be computed from the label -- it needs nv and n_fu -- and is taken
    from the exact 'x_pair' column built in build_pair_deltas instead.
    """
    lo, hi = pair_to_lo_hi(pair_label)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    if x_mode == "lo":
        return lo/1000.0
    elif x_mode == "mid":
        return (0.5*(lo+hi))/1000.0
    elif x_mode == "delta":
        return 1.0 - hi/1000.0  # oxygen-vacancy fraction
    return hi/1000.0

def parse_pair_hi(pair_label: str):
    _, hi = pair_to_lo_hi(pair_label)
    return hi

# ---------- t critical ----------
_T975 = { 1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,
          10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,
          18:2.101,19:2.093,20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,
          26:2.056,27:2.052,28:2.048,29:2.045,30:2.042 }

def _z_from_alpha(alpha):
    if abs(alpha-0.10)<1e-9: return 1.645
    if abs(alpha-0.05)<1e-9:  return 1.960
    if abs(alpha-0.01)<1e-9:  return 2.576
    return 1.960

def t_crit(alpha: float, df: int):
    """Two-sided t critical value.  Uses scipy when available for arbitrary alpha."""
    if df <= 0:
        return _z_from_alpha(alpha)
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(1.0 - alpha/2.0, df))
    except Exception:
        pass
    if abs(alpha - 0.05) < 1e-9 and 1 <= df <= 30:
        return _T975[df]
    return _z_from_alpha(alpha)

# ---------- Regression for line plots (index-x) ----------

def fit_linear_regression_with_uncertainty(y_values, x_values=None,
                                           ci_level=0.95, ci_mode="mean"):
    """
    OLS y vs x (x defaults to the index 0..N-1) with SEs and CI/PI bands.
    Returns: slope, intercept, R2, y_pred_full, n_used, se_slope, se_intercept, ci_lo, ci_hi
    """
    y = np.asarray(y_values, dtype=float)
    x_full = (np.arange(len(y), dtype=float) if x_values is None
              else np.asarray(x_values, dtype=float))
    mask = np.isfinite(y) & np.isfinite(x_full)
    x = x_full[mask]; z = y[mask]
    n = z.size
    if n < 2:
        nanarr = np.full_like(x_full, np.nan, dtype=float)
        return (np.nan, np.nan, np.nan, nanarr, int(n), np.nan, np.nan, nanarr, nanarr)

    slope, intercept = np.polyfit(x, z, 1)
    y_pred_full = slope * x_full + intercept

    y_hat = slope * x + intercept
    resid = z - y_hat
    RSS = float(np.sum(resid**2))
    df = n - 2
    if df <= 0:
        s2 = np.nan; se_m = se_b = np.nan
    else:
        s2 = RSS / df
        xbar = float(np.mean(x))
        Sxx = float(np.sum((x - xbar)**2))
        if Sxx <= 0:
            se_m = se_b = np.nan
        else:
            se_m = math.sqrt(s2 / Sxx)
            se_b = math.sqrt(s2 * (1.0/n + (xbar**2)/Sxx))

    ss_tot = float(np.sum((z - np.mean(z))**2))
    r2 = 1.0 - RSS/ss_tot if ss_tot > 0 else np.nan

    if df > 0 and not np.isnan(s2):
        xbar = float(np.mean(x))
        Sxx  = float(np.sum((x - xbar)**2))
        alpha = 1.0 - float(ci_level)
        tcrit = t_crit(alpha, df)
        s = math.sqrt(s2)
        x0 = x_full
        base = (1.0/n) + ((x0 - xbar)**2)/Sxx if Sxx > 0 else np.inf
        if ci_mode == "pred":
            base = 1.0 + base
        half = tcrit * s * np.sqrt(base)
        ci_lo = y_pred_full - half
        ci_hi = y_pred_full + half
    else:
        ci_lo = np.full_like(x_full, np.nan); ci_hi = np.full_like(x_full, np.nan)

    return slope, intercept, r2, y_pred_full, int(n), se_m, se_b, ci_lo, ci_hi

def _fit_line_return_full_pred(y_list, x_list=None):
    y = np.asarray(y_list, dtype=float)
    x_full = (np.arange(len(y), dtype=float) if x_list is None
              else np.asarray(x_list, dtype=float))
    mask = np.isfinite(y) & np.isfinite(x_full)
    if mask.sum() < 2:
        return np.full_like(x_full, np.nan, dtype=float)
    slope, intercept = np.polyfit(x_full[mask], y[mask], 1)
    return slope * x_full + intercept

def add_regressed_pair_values(df_pairs, metrics=("ΔE_bavg","ΔH_bavg","ΔG_bavg"),
                              x_col="x_pair"):
    """
    For each (comp_dir, T), fit each metric against x and write the predictions into
    '<metric>_reg'.

    x is the exact vacancy fraction nv_lo/n_fu when available, falling back to the pair
    index.  The two agree whenever the o-series is complete (x is then affine in the
    index); they differ only across a missing o* directory, where x is the correct one.

    Vectorised: one index-aligned assignment per (group, metric) instead of a
    full-frame boolean scan per (group, metric, pair) as before.
    """
    df = df_pairs.copy()
    for m in metrics:
        df[f"{m}_reg"] = np.nan
    if df.empty:
        return df

    use_x = (x_col in df.columns) and df[x_col].notna().any()
    df["_ord"] = df["Pair"].map(pair_sort_key)
    for _, block in df.groupby(["comp_dir", "Temperature"], sort=False):
        b = block.sort_values("_ord")
        xv = b[x_col].to_numpy(dtype=float) if use_x else None
        for m in metrics:
            if m not in df.columns:
                continue
            pred = _fit_line_return_full_pred(b[m].to_numpy(dtype=float), xv)
            df.loc[b.index, f"{m}_reg"] = pred
    return df.drop(columns=["_ord"])

# ---------- OLS on continuous x for ΔG zero-threshold ----------

def linear_zero_with_uncertainty(x, y):
    """
    Returns m, b, R2, n_used, se_m, se_b, cov_mb, x0, x0_std
    with x0 = -b/m and std via the delta method including cov(m,b).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = x.size
    if n < 2:
        return (np.nan, np.nan, np.nan, int(n), np.nan, np.nan, np.nan, np.nan, np.nan)

    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    Sxx  = float(np.sum((x - xbar)**2))
    Sxy  = float(np.sum((x - xbar) * (y - ybar)))
    if Sxx <= 0:
        return (np.nan, np.nan, np.nan, int(n), np.nan, np.nan, np.nan, np.nan, np.nan)

    m = Sxy / Sxx
    b = ybar - m*xbar
    resid = y - (m*x + b)
    RSS = float(np.sum(resid**2))
    TSS = float(np.sum((y - ybar)**2))
    R2  = 1.0 - (RSS/TSS) if TSS > 0 else np.nan

    df = n - 2
    if df <= 0:
        return (m, b, R2, int(n), np.nan, np.nan, np.nan, np.nan, np.nan)

    sigma2 = RSS / df
    se_m = math.sqrt(sigma2 / Sxx)
    se_b = math.sqrt(sigma2 * (1.0/n + (xbar**2)/Sxx))
    cov_mb = - xbar * sigma2 / Sxx

    if abs(m) < 1e-15:
        x0 = np.nan; x0_std = np.nan
    else:
        x0 = -b / m
        dx_dm =  b / (m*m)
        dx_db = -1.0 / m
        var_x0 = (dx_dm**2) * (se_m**2) + (dx_db**2) * (se_b**2) + 2.0 * dx_dm * dx_db * cov_mb
        x0_std = math.sqrt(max(var_x0, 0.0))

    return (m, b, R2, int(n), se_m, se_b, cov_mb, x0, x0_std)

def compute_continuous_thresholds_for_DG(
    df_pairs, temps, out_csv_path, x_mode="nv", group_mode="by_A", fit_hi_min=None
):
    """Build x from Pair via x_mode and regress ΔG ~ x per line."""
    rows = []
    gcol = "ΔG_bavg_reg" if ("ΔG_bavg_reg" in df_pairs.columns) else "ΔG_bavg"

    for T in temps:
        dT = df_pairs[df_pairs["Temperature"] == T].copy()
        if dT.empty: continue
        if x_mode == "nv":
            dT["x_val"] = pd.to_numeric(dT["x_pair"], errors="coerce")
        else:
            dT["x_val"] = dT["Pair"].apply(lambda p: pair_to_x(p, x_mode=x_mode))
        if fit_hi_min is not None:
            dT = dT[dT["Pair"].apply(lambda p: np.isfinite(pair_to_lo_hi(p)[1])
                                     and pair_to_lo_hi(p)[1] >= fit_hi_min)]
        if dT.empty: continue

        outer, inner = ("A_frac", "B_frac") if group_mode == "by_A" else ("B_frac", "A_frac")
        for o_val, d_o in dT.groupby(outer):
            for i_val, d_line in d_o.groupby(inner):
                m,b,R2,n,sem,seb,covmb,x0,x0std = linear_zero_with_uncertainty(
                    d_line["x_val"].to_numpy(), d_line[gcol].to_numpy())
                a_frac = o_val if group_mode == "by_A" else i_val
                b_frac = i_val if group_mode == "by_A" else o_val
                rows.append({
                    "Temperature": T,
                    "comp_dir": d_line["comp_dir"].iloc[0] if not d_line.empty else "",
                    "A_frac": float(a_frac), "B_frac": float(b_frac),
                    "metric": "ΔG", "x_mode": x_mode,
                    "x0": x0, "x0_std": x0std,
                    "m": m, "se_m": sem, "b": b, "se_b": seb, "R2": R2, "n_points": n
                })
    pd.DataFrame(rows).to_csv(out_csv_path, index=False)

# ---------- ΔG=0 threshold vs oxygen partial pressure ----------

def compute_pO2_thresholds(df_pairs, temps, pressures_atm, variant_label="",
                           a_signed=np.nan, gcol="ΔG_bavg", x_col="x_pair",
                           fit_hi_min=None):
    """
    For every (composition, T, pO2): fit ΔG ~ x and solve ΔG = 0.

    x is the exact vacancy fraction nv_lo/n_fu, so x0 is the vacancy fraction at which
    oxygen release stops being favourable, and x0 * n_fu is the vacancy count.

    The pressure correction (paper eq. 1) is +dnv * (1/2) kB T ln(pO2/P0) per pair.  When
    every pair has dnv = 1 -- a complete o-series -- that is a constant, so this is
    exactly "shift the 1 atm fitting line vertically and re-read the crossing".  Pairs
    that straddle a missing o* directory carry dnv > 1 and get the correction scaled, so
    the fit is done on corrected y rather than by shifting the intercept.
    """
    rows = []
    for T in temps:
        dT = df_pairs[df_pairs["Temperature"] == T].copy()
        if dT.empty:
            continue
        if fit_hi_min is not None:
            dT = dT[dT["Pair"].apply(lambda p: np.isfinite(pair_to_lo_hi(p)[1])
                                     and pair_to_lo_hi(p)[1] >= fit_hi_min)]
        if dT.empty:
            continue

        for comp, d in dT.groupby("comp_dir"):
            x = pd.to_numeric(d[x_col], errors="coerce").to_numpy(dtype=float)
            y0 = pd.to_numeric(d[gcol], errors="coerce").to_numpy(dtype=float)
            dnv = pd.to_numeric(d["dnv"], errors="coerce").to_numpy(dtype=float)
            dnv = np.where(np.isfinite(dnv), dnv, 1.0)
            nfu = pd.to_numeric(d["n_fu"], errors="coerce").dropna()
            n_fu = float(nfu.iloc[0]) if not nfu.empty else np.nan
            A_frac = d["A_frac"].iloc[0] if "A_frac" in d.columns else np.nan
            B_frac = d["B_frac"].iloc[0] if "B_frac" in d.columns else np.nan
            n_used = pd.to_numeric(d.get("n_series_used"), errors="coerce").dropna() \
                if "n_series_used" in d.columns else pd.Series(dtype=float)
            n_tot = pd.to_numeric(d.get("n_series_total"), errors="coerce").dropna() \
                if "n_series_total" in d.columns else pd.Series(dtype=float)

            for p in pressures_atm:
                shift = dG_pO2_shift(T, p)
                y = y0 + dnv * shift
                m, b, R2, n, sem, seb, covmb, x0, x0std = \
                    linear_zero_with_uncertainty(x, y)
                rows.append({
                    "s_config_variant": variant_label, "a_signed": a_signed,
                    "Temperature": T, "comp_dir": comp,
                    "A_frac": A_frac, "B_frac": B_frac,
                    "n_fu": n_fu, "pO2_atm": float(p),
                    "dG_pO2_per_vacancy_eV": shift,
                    "x0": x0, "x0_std": x0std,
                    "nv0": x0 * n_fu if np.isfinite(x0) else np.nan,
                    "nv0_std": x0std * n_fu if np.isfinite(x0std) else np.nan,
                    "m": m, "se_m": sem, "b": b, "se_b": seb,
                    "R2": R2, "n_points": n,
                    "n_series_min": int(n_used.min()) if not n_used.empty else np.nan,
                    "n_series_total": int(n_tot.max()) if not n_tot.empty else np.nan,
                    "all_dnv_1": bool(np.all(dnv == 1.0)),
                })
    return pd.DataFrame(rows)


def build_simple_summary(df_thr, pressures, variant_label, comp_info=None):
    """
    The headline table: one row per (composition, temperature), the ΔG = 0 crossing as
    the OXYGEN VACANCY FRACTION x = n_v/n_fu at each oxygen partial pressure.

        x_1atm, x_0p8atm, x_3e-05atm, ...      one column per --pO2-list value

    x is the same axis the ΔG line plots use, so a number here can be read straight off
    the corresponding figure.  **A negative crossing is written as 0**: it means ΔG > 0
    already at the first vacancy, i.e. the material releases no oxygen at that pressure,
    and a negative vacancy fraction is not a thing.  n_formula_units is carried along
    for anyone who wants x * n_fu as a count.

    Only the physical S_config branch (a < 0) is reported here; the sign study, the
    per-pair deltas and the fit diagnostics stay in the --full-csv files.
    """
    if df_thr.empty:
        return pd.DataFrame()
    comp_info = comp_info or {}
    d = df_thr[df_thr["s_config_variant"] == variant_label] \
        if "s_config_variant" in df_thr.columns else df_thr
    if d.empty:
        return pd.DataFrame()

    def _x0(v):
        """Vacancy fraction at the crossing, floored at zero."""
        return round(max(0.0, float(v)), 4) if np.isfinite(v) else np.nan

    keys = ["comp_dir", "Temperature"]
    rows = []
    for (comp, T), g in d.groupby(keys, sort=True):
        info = comp_info.get(comp) or parse_comp_dir(comp) or {}
        by_p = {float(r["pO2_atm"]): r for _, r in g.iterrows()}
        n_fu = pd.to_numeric(g["n_fu"], errors="coerce").dropna()
        n_fu = float(n_fu.iloc[0]) if not n_fu.empty else np.nan

        row = {
            "family": info.get("family", ""),
            "composition": comp,
            "formula": info.get("formula", ""),
            "Temperature_K": int(T) if float(T).is_integer() else T,
            "n_formula_units": int(n_fu) if np.isfinite(n_fu) else np.nan,
            "A_frac": info.get("A_frac", g["A_frac"].iloc[0] if "A_frac" in g else np.nan),
            "B_frac": info.get("B_frac", g["B_frac"].iloc[0] if "B_frac" in g else np.nan),
        }

        for p in pressures:
            r = by_p.get(float(p))
            row[f"x_{fmt_pressure(p)}"] = _x0(r["x0"]) if r is not None else np.nan

        n_min = pd.to_numeric(g.get("n_series_min"), errors="coerce").dropna() \
            if "n_series_min" in g.columns else pd.Series(dtype=float)
        n_tot = pd.to_numeric(g.get("n_series_total"), errors="coerce").dropna() \
            if "n_series_total" in g.columns else pd.Series(dtype=float)
        row["n_sets_used"] = int(n_min.min()) if not n_min.empty else np.nan
        row["n_sets_total"] = int(n_tot.max()) if not n_tot.empty else np.nan
        row["R2"] = round(float(g["R2"].median()), 4) if g["R2"].notna().any() else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    # the usable oxygen swing: the extra vacancy fraction unlocked by dropping pO2 from
    # the highest listed pressure to the lowest.  Both terms are already floored at 0.
    p_sorted = sorted(float(p) for p in pressures)
    if len(p_sorted) >= 2:
        lo, hi = f"x_{fmt_pressure(p_sorted[0])}", f"x_{fmt_pressure(p_sorted[-1])}"
        if lo in out.columns and hi in out.columns:
            out["x_swing"] = (out[lo] - out[hi]).round(4)
    return out.sort_values(["family", "Temperature_K", "A_frac", "B_frac"]) \
              .reset_index(drop=True)

def summarize_pO2_thresholds(df_thr, p_lo, p_hi):
    """
    One row per (variant, T, composition): x0 at every pressure, plus the difference
    between p_lo and p_hi.

    diff = x0(p_lo) - x0(p_hi).  With p_lo < p_hi and ΔG rising with x this is positive:
    it is the extra vacancy fraction that becomes accessible on dropping the oxygen
    partial pressure from p_hi to p_lo, i.e. the usable oxygen swing.
    """
    if df_thr.empty:
        return pd.DataFrame()
    keys = ["s_config_variant", "a_signed", "Temperature", "comp_dir",
            "A_frac", "B_frac", "n_fu"]
    out = []
    for kv, g in df_thr.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, kv))
        by_p = {float(r["pO2_atm"]): r for _, r in g.iterrows()}
        for p, r in sorted(by_p.items(), reverse=True):
            tag = fmt_pressure(p)
            row[f"x0_{tag}"] = r["x0"]
            row[f"x0_std_{tag}"] = r["x0_std"]
            row[f"nv0_{tag}"] = r["nv0"]
            row[f"dG_pO2_{tag}_eV"] = r["dG_pO2_per_vacancy_eV"]
        r_lo, r_hi = by_p.get(float(p_lo)), by_p.get(float(p_hi))
        if r_lo is not None and r_hi is not None:
            row["pO2_lo_atm"] = float(p_lo)
            row["pO2_hi_atm"] = float(p_hi)
            row["dx0"] = r_lo["x0"] - r_hi["x0"]
            row["dnv0"] = r_lo["nv0"] - r_hi["nv0"]
            m, se_m = r_hi["m"], r_hi["se_m"]
            dshift = (r_lo["dG_pO2_per_vacancy_eV"] - r_hi["dG_pO2_per_vacancy_eV"])
            all1 = bool(r_hi["all_dnv_1"]) and bool(r_lo["all_dnv_1"])
            if all1 and np.isfinite(m) and abs(m) > 1e-15 and np.isfinite(se_m):
                # every pair has dnv = 1, so the two fits differ by a CONSTANT: the
                # slope is identical and dx0 = -dshift/m exactly (verified to 1e-16).
                # Delta method on that single ratio.
                row["dx0_std"] = abs(dshift / (m * m)) * se_m
                row["dx0_std_exact"] = True
            else:
                # a missing o* directory makes some dnv > 1, so the correction is not
                # constant, both slope and intercept move, and -dshift/m no longer
                # holds (measured 0.021 in x on the synthetic gapped series).  Fall
                # back to adding the two x0 uncertainties in quadrature -- an upper
                # bound, since the two fits are positively correlated.
                a_, b_ = r_lo["x0_std"], r_hi["x0_std"]
                row["dx0_std"] = (math.sqrt(a_**2 + b_**2)
                                  if np.isfinite(a_) and np.isfinite(b_) else np.nan)
                row["dx0_std_exact"] = False
            row["dnv0_std"] = (row["dx0_std"] * row["n_fu"]
                               if np.isfinite(row["dx0_std"]) else np.nan)
            row["R2"] = r_hi["R2"]
            row["m"] = m
            row["se_m"] = se_m
            row["m_pO2_lo"] = r_lo["m"]
            row["n_points"] = r_hi["n_points"]
            row["all_dnv_1"] = all1
        out.append(row)
    return pd.DataFrame(out)

def compare_s_config_signs(df_summary, p_lo, p_hi, reference="corrected_a_neg"):
    """
    Pivot the summary so each (T, composition) row carries x0/dx0 for every S_config
    variant side by side, plus the shift relative to the corrected (a<0) result.
    """
    if df_summary.empty:
        return pd.DataFrame()
    lo, hi = fmt_pressure(p_lo), fmt_pressure(p_hi)
    keep = ["Temperature", "comp_dir", "A_frac", "B_frac", "n_fu"]
    cols = [f"x0_{hi}", f"x0_{lo}", f"nv0_{hi}", f"nv0_{lo}", "dx0", "dnv0", "m", "R2"]
    cols = [c for c in cols if c in df_summary.columns]

    frames = {}
    for var, g in df_summary.groupby("s_config_variant", sort=False):
        frames[var] = g.set_index(keep)[cols].add_suffix(f"__{var}")
    merged = None
    for var, f in frames.items():
        merged = f if merged is None else merged.join(f, how="outer")
    merged = merged.reset_index()

    for var in frames:
        if var == reference:
            continue
        for c in cols:
            a, b = f"{c}__{var}", f"{c}__{reference}"
            if a in merged.columns and b in merged.columns:
                merged[f"{c}__{var}_minus_{reference}"] = merged[a] - merged[b]
    return merged

# ---------- Heatmaps ----------

def _corner_labels(ax, elements):
    """Axis corner labels, derived from the family instead of hardcoded Sr/Ca/Fe/Mn."""
    a_base, a_dop, b_base, b_dop = elements
    ax.text(-0.06, 1.02, b_base, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=14, fontweight='bold')
    ax.text(-0.06,-0.06, b_dop,  transform=ax.transAxes, ha='right', va='top',
            fontsize=14, fontweight='bold')
    ax.text(-0.02,-0.12, a_dop,  transform=ax.transAxes, ha='right', va='top',
            fontsize=14, fontweight='bold')
    ax.text( 1.02,-0.12, a_base, transform=ax.transAxes, ha='left',  va='top',
            fontsize=14, fontweight='bold')

def _draw_heatmap(pivot, title, out_path, elements, vmin, vmax, annotate, dpi):
    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        pivot, annot=annotate, fmt=".2f", cmap="RdYlGn",
        linewidths=1, linecolor="black", cbar=True,
        annot_kws={"fontsize": 9} if annotate else None,
        vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.xticks(rotation=0); plt.yticks(rotation=0)
    _corner_labels(ax, elements)
    plt.title(title)
    plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

def plot_heatmap(df, value_col, title, filename, target_path,
                 elements=("Sr","Ca","Fe","Mn"), vmin=-2, vmax=2,
                 annotate=True, dpi=200):
    """df must already carry the A_frac / B_frac columns."""
    d = df.dropna(subset=["A_frac", "B_frac"])
    if d.empty:
        return
    d = d.groupby(["B_frac", "A_frac"], as_index=False)[[value_col]].mean()
    pivot = d.pivot(index="B_frac", columns="A_frac", values=value_col)
    if pivot.size == 0:
        return
    Path(target_path).mkdir(parents=True, exist_ok=True)
    _draw_heatmap(pivot, title, os.path.join(target_path, filename),
                  elements, vmin, vmax, annotate, dpi)

def find_crossing_hi_ratio_from_regression(y, pair_order, on_all_positive_return_one=False):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return np.nan
    x = np.arange(n, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan

    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    if not np.isfinite(slope):
        return np.nan

    y_pred = slope * x + intercept
    spans_zero = (np.nanmin(y_pred) <= 0.0 <= np.nanmax(y_pred))
    if not spans_zero:
        if on_all_positive_return_one and np.nanmin(y_pred) > 0.0:
            return 1.0
        return np.nan

    if abs(slope) < 1e-12:
        if on_all_positive_return_one and np.nanmin(y_pred) > 0.0:
            return 1.0
        return np.nan

    x0 = -intercept / slope
    idx = int(np.ceil(x0)) if slope > 0 else int(np.floor(x0))
    idx = max(0, min(n-1, idx))
    hi = parse_pair_hi(pair_order[idx])
    return float(hi)/1000.0 if np.isfinite(hi) else np.nan

def plot_crossing_heatmap_metric(df_pairs, temps, outdir, metric_col="ΔG_bavg",
                                 title_prefix="ΔG", elements=("Sr","Ca","Fe","Mn"),
                                 vmin=0.5, vmax=1.0, annotate=True, dpi=200,
                                 h_all_pos_to_one=False):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for T in temps:
        dT = df_pairs[df_pairs["Temperature"] == T].dropna(subset=["A_frac", "B_frac"])
        if dT.empty: continue

        pair_order = order_pairs_unique(dT["Pair"].unique())
        rows = []
        for comp, block in dT.groupby("comp_dir"):
            val_map = block.set_index("Pair")[metric_col].to_dict()
            y = [val_map.get(p, np.nan) for p in pair_order]
            rows.append({
                "comp_dir": comp,
                "A_frac": block["A_frac"].iloc[0], "B_frac": block["B_frac"].iloc[0],
                "Temperature": T,
                "hi_ratio": find_crossing_hi_ratio_from_regression(
                    y, pair_order, on_all_positive_return_one=h_all_pos_to_one),
            })

        df_val = pd.DataFrame(rows).dropna(subset=["A_frac", "B_frac"])
        if df_val.empty: continue
        df_val = df_val.groupby(["B_frac", "A_frac"], as_index=False)["hi_ratio"].mean()
        pivot = df_val.pivot(index="B_frac", columns="A_frac", values="hi_ratio")
        if pivot.size == 0: continue

        _draw_heatmap(
            pivot, f"{title_prefix}=0 crossing → hi/1000 @ {T}K",
            os.path.join(outdir, f"heatmap_crossing_hi_ratio_{title_prefix}_{T}K.png"),
            elements, vmin, vmax, annotate, dpi)

# ---------- Line plots for ΔG/ΔH ----------

def _apply_x_ticks(ax, x_vals, n_fu, mode="auto", max_ticks=12):
    """
    x is a vacancy fraction.  'auto' leaves matplotlib's numeric ticks alone; 'frac'
    labels the data points as n/n_fu; 'nv' labels them with the integer vacancy count.
    """
    xs = np.asarray([v for v in x_vals if np.isfinite(v)], dtype=float)
    if mode == "auto" or xs.size == 0 or not (n_fu and np.isfinite(n_fu)):
        return
    xs = np.unique(xs)
    step = max(1, int(math.ceil(xs.size / float(max_ticks))))
    sel = xs[::step]
    nv = np.rint(sel * float(n_fu)).astype(int)
    labels = ([f"{v}/{int(n_fu)}" for v in nv] if mode == "frac"
              else [str(v) for v in nv])
    ax.set_xticks(sel)
    ax.set_xticklabels(labels, rotation=45 if mode == "frac" else 0,
                       ha="right" if mode == "frac" else "center")

def _line_panel(dT, pair_order, outer_col, inner_col, outer_val, d_outer,
                label_col, std_col, y_label, title, out_path,
                outer_names, inner_names, ymin, ymax, band_type, ci_level, ci_mode,
                reg_csv_rows, figure_tag, T, dpi,
                pair_x=None, pair_dnv=None, n_fu=None, x_tick_mode="auto",
                x_label=None, pO2_list=(), show_x0=True):
    """
    One figure: fixed `outer_col`, one line per `inner_col` value.

    x is the numeric vacancy fraction nv_lo/n_fu (1/64, 2/64, ...), not the pair label.
    When pO2_list is non-empty the 1 atm fit is additionally shifted by
    dnv*(1/2)kB T ln(pO2) (paper eq. 1) and each ΔG=0 crossing is marked.
    """
    pair_x = pair_x or {}
    pair_dnv = pair_dnv or {}
    x_vals = np.asarray([pair_x.get(p, np.nan) for p in pair_order], dtype=float)
    dnv_vals = np.asarray([pair_dnv.get(p, 1.0) for p in pair_order], dtype=float)
    dnv_vals = np.where(np.isfinite(dnv_vals), dnv_vals, 1.0)
    if not np.any(np.isfinite(x_vals)):          # no exact x available: fall back
        x_vals = np.arange(len(pair_order), dtype=float)

    # keep the x range tied to the data: a ΔG=0 crossing extrapolated far beyond the
    # last computed vacancy count must not stretch the axis.  Such an x0 is still
    # reported in the legend and in the CSVs, just not drawn.
    xf = x_vals[np.isfinite(x_vals)]
    x_lo, x_hi = float(np.min(xf)), float(np.max(xf))
    pad = 0.04 * (x_hi - x_lo) if x_hi > x_lo else 0.05
    draw_lo, draw_hi = x_lo - pad, x_hi + pad
    in_view = lambda v: np.isfinite(v) and draw_lo <= v <= draw_hi

    fig, ax = plt.subplots(figsize=(9, 5))
    for inner_val, d_line in d_outer.groupby(inner_col):
        val_map = d_line.set_index("Pair")[label_col].to_dict()
        std_map = (d_line.set_index("Pair")[std_col].to_dict()
                   if std_col in d_line.columns else {})
        y = [val_map.get(p, np.nan) for p in pair_order]
        ystd = [std_map.get(p, np.nan) for p in pair_order] if std_map else None

        base_label = (f"{inner_names[0]}:{inner_val:.3f} "
                      f"{inner_names[1]}:{1-inner_val:.3f}")
        [ln] = ax.plot(x_vals, y, marker='o', markersize=3, linewidth=1, label=base_label)
        color = ln.get_color()

        slope, intercept, r2, y_fit, n_used, se_m, se_b, ci_lo, ci_hi = \
            fit_linear_regression_with_uncertainty(y, x_vals, ci_level=ci_level,
                                                   ci_mode=ci_mode)
        x0 = (-intercept / slope) if (np.isfinite(slope) and abs(slope) > 1e-15) else np.nan
        if np.isfinite(slope):
            ax.plot(x_vals, y_fit, linestyle=':', linewidth=1.5, color=color, alpha=0.9)
            lab = (f"{base_label}  (m={slope:.3f}±{se_m if np.isfinite(se_m) else float('nan'):.3f}, "
                   f"b={intercept:.3f}±{se_b if np.isfinite(se_b) else float('nan'):.3f}, "
                   f"R²={r2:.3f}")
            if show_x0 and np.isfinite(x0):
                # same quantity and same units as the x_<p> columns of the summary CSV
                lab += f", x₀={x0:.4f}"
            ln.set_label(lab + ")")
        if in_view(x0):
            ax.axvline(x0, linestyle='--', linewidth=1, color=color, alpha=0.85)

        # pressure-shifted fits: ΔG(pO2) = ΔG(1 atm) + dnv*(1/2)kB T ln(pO2/P0)
        for p in pO2_list:
            if abs(float(p) - 1.0) < 1e-12:
                continue
            shift = dG_pO2_shift(T, p)
            ys = np.asarray(y, dtype=float) + dnv_vals * shift
            sl, ic, _, yf, _, _, _, _, _ = fit_linear_regression_with_uncertainty(
                ys, x_vals, ci_level=ci_level, ci_mode=ci_mode)
            if not np.isfinite(sl):
                continue
            ax.plot(x_vals, yf, linestyle='-.', linewidth=1.0, color=color, alpha=0.55)
            x0p = (-ic / sl) if abs(sl) > 1e-15 else np.nan
            if in_view(x0p):
                ax.plot([x0p], [0.0], marker='v', markersize=5, color=color, alpha=0.9)

        if band_type == "std" and ystd is not None:
            ylo = [a-b if (np.isfinite(a) and np.isfinite(b)) else np.nan for a, b in zip(y, ystd)]
            yhi = [a+b if (np.isfinite(a) and np.isfinite(b)) else np.nan for a, b in zip(y, ystd)]
            ax.fill_between(x_vals, ylo, yhi, alpha=0.15, color=color, linewidth=0)
        elif band_type == "ci":
            if np.any(np.isfinite(ci_lo)) and np.any(np.isfinite(ci_hi)):
                ax.fill_between(x_vals, ci_lo, ci_hi, alpha=0.12, color=color, linewidth=0)

        if reg_csv_rows is not None:
            reg_csv_rows.append({
                "figure": figure_tag, "Temperature": T,
                "A_frac": float(outer_val) if outer_col == "A_frac" else float(inner_val),
                "B_frac": float(inner_val) if outer_col == "A_frac" else float(outer_val),
                "slope": slope, "SE_slope": se_m,
                "intercept": intercept, "SE_intercept": se_b,
                "R2": r2, "n_points": n_used, "x0_1atm": x0,
            })

    _apply_x_ticks(ax, x_vals, n_fu, mode=x_tick_mode)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label or (f"Oxygen vacancy fraction  x = n_v/{int(n_fu)}"
                              if n_fu and np.isfinite(n_fu)
                              else "Oxygen vacancy fraction  x = n_v/n_fu"))
    ax.set_title(title)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(draw_lo, draw_hi)
    ax.axhline(0, linestyle='--', linewidth=1, color='black', alpha=0.9)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def line_plots_for_metric(df_pairs, temps, outdir_base, metric="G",
                          elements=("Sr","Ca","Fe","Mn"),
                          ymin=-2, ymax=2, band_type="std", ci_level=0.95,
                          ci_mode="mean", reg_csv_rows=None, dpi=200, fit_hi_min=None,
                          x_tick_mode="auto", pO2_list=()):
    a_base, a_dop, b_base, b_dop = elements
    up = metric.upper()
    label_col = f"Δ{up}_bavg"
    std_col   = f"Δ{up}_bavg_wstd"
    y_label   = f"Δ{up} (eV)"
    # pressure shift applies to the oxygen chemical potential, i.e. to ΔG only
    p_list = tuple(pO2_list) if up == "G" else ()

    for T in temps:
        dT = df_pairs[df_pairs["Temperature"] == T].copy()
        if dT.empty: continue
        if fit_hi_min is not None:
            dT = dT[dT["Pair"].apply(lambda p: np.isfinite(pair_to_lo_hi(p)[1])
                                     and pair_to_lo_hi(p)[1] >= fit_hi_min)]
        if dT.empty: continue

        pair_order = order_pairs_unique(dT["Pair"].unique())
        pair_x = (dT.drop_duplicates("Pair").set_index("Pair")["x_pair"].to_dict()
                  if "x_pair" in dT.columns else {})
        pair_dnv = (dT.drop_duplicates("Pair").set_index("Pair")["dnv"].to_dict()
                    if "dnv" in dT.columns else {})
        nfu_vals = pd.to_numeric(dT.get("n_fu"), errors="coerce").dropna() \
            if "n_fu" in dT.columns else pd.Series(dtype=float)
        n_fu = float(nfu_vals.iloc[0]) if not nfu_vals.empty else None
        extras = dict(pair_x=pair_x, pair_dnv=pair_dnv, n_fu=n_fu,
                      x_tick_mode=x_tick_mode, pO2_list=p_list)

        # fixed A site, one line per B composition
        outdir = Path(outdir_base) / f"lines_by_{a_base}{a_dop}_T{T}K_{up}"
        outdir.mkdir(parents=True, exist_ok=True)
        for a_val, d_a in dT.groupby("A_frac"):
            _line_panel(
                dT, pair_order, "A_frac", "B_frac", a_val, d_a,
                label_col, std_col, y_label,
                f"Δ{up} vs vacancy fraction — fixed {a_base}:{a_dop}={a_val:.3f}:{1-a_val:.3f}",
                outdir / (f"lines_fixed_{a_base}{a_dop}_{up}"
                          f"_{a_base.lower()}{int(round(a_val*1000)):04d}"
                          f"_{a_dop.lower()}{int(round((1-a_val)*1000)):04d}.png"),
                (a_base, a_dop), (b_base, b_dop),
                ymin, ymax, band_type, ci_level, ci_mode,
                reg_csv_rows, f"fixed_{a_base}{a_dop}_{up}", T, dpi, **extras)

        # fixed B site, one line per A composition
        outdir = Path(outdir_base) / f"lines_by_{b_base}{b_dop}_T{T}K_{up}"
        outdir.mkdir(parents=True, exist_ok=True)
        for b_val, d_b in dT.groupby("B_frac"):
            _line_panel(
                dT, pair_order, "B_frac", "A_frac", b_val, d_b,
                label_col, std_col, y_label,
                f"Δ{up} vs vacancy fraction — fixed {b_base}:{b_dop}={b_val:.3f}:{1-b_val:.3f}",
                outdir / (f"lines_fixed_{b_base}{b_dop}_{up}"
                          f"_{b_base.lower()}{int(round(b_val*1000)):04d}"
                          f"_{b_dop.lower()}{int(round((1-b_val)*1000)):04d}.png"),
                (b_base, b_dop), (a_base, a_dop),
                ymin, ymax, band_type, ci_level, ci_mode,
                reg_csv_rows, f"fixed_{b_base}{b_dop}_{up}", T, dpi, **extras)

# ---------- Collection ----------

def _collect_one_odir(o_path, temps=(670, 1070)):
    """Worker: read E, thermal properties and cell stoichiometry for one o* directory."""
    od = Path(o_path)
    E = read_energy_txt(str(od / "energy.txt"))
    if E is None:
        E = extract_dft_energy_from_outcar(str(od / "OUTCAR"))
    therm = extract_thermal_properties(str(od / "thermal_properties.yaml"), list(temps))
    n_fu, n_O, nv = infer_cell_stoichiometry(od)
    return {"o_path": str(od), "E": E, "therm": therm,
            "n_fu": n_fu, "n_O": n_O, "nv": nv}

_O_DIR = re.compile(r"^o\d{3,4}$")


def collect_state_rows(parent, series_glob, inner_subdir, temps, elements,
                       a_signed, n_fu_override, nprocs, comp_prefix=None, nv_tag_tol=1):
    """
    Walk the tree and build the state-level table.

    The o* directories are found by their own name rather than by a hardcoded depth or
    an assumed 'Sr' prefix, so any family naming works and both the
    <series>/<group>/<comp>/o#### layout and a flatter <series>/<comp>/o#### one are
    read without being told which is which.  --comp-prefix still filters if given.

    A structure whose own vacancy count contradicts the o#### it sits in by more than
    `nv_tag_tol` is REJECTED: it is not the structure that was asked for, so its energy
    does not describe this composition.  The o-tag is a rounded label, hence the
    tolerance -- 1000ths of delta against n_fu sites round-trips exactly for a 64 f.u.
    cell and can slip by one for a much larger one -- but a cell that lost 142 of its
    186 oxygens is a corrupt file, not rounding.  Rejected structures are counted as
    failed sets, so the point simply averages over the sets that are real.
    """
    tasks = []                       # (series, comp_dir_name, o_dir_path)
    series_dirs = sorted(d for d in Path(parent).glob(series_glob) if d.is_dir())
    if not series_dirs:
        print(f"[warn] no series directories matched {series_glob!r} under {parent}")
    for sd in series_dirs:
        base = sd / inner_subdir if inner_subdir not in ("", ".", None) else sd
        if not base.exists():
            print(f"[warn] missing inner subdir in {sd}: {base}")
            continue
        try:
            found = sorted(d for d in base.rglob("o*") if _O_DIR.match(d.name) and d.is_dir())
        except (PermissionError, OSError):
            continue
        for od in found:
            comp = od.parent.name
            if comp_prefix and not comp.startswith(comp_prefix):
                continue
            tasks.append((sd.name, comp, str(od)))

    if not tasks:
        return pd.DataFrame(), {}

    worker = partial(_collect_one_odir, temps=tuple(temps))
    paths = [t[2] for t in tasks]
    if nprocs and nprocs > 1 and len(paths) > 1:
        with Pool(processes=nprocs) as pool:
            results = pool.map(worker, paths, chunksize=max(1, len(paths)//(nprocs*4) or 1))
    else:
        results = [worker(p) for p in paths]
    by_path = {r["o_path"]: r for r in results}

    # n_fu per composition directory, with a consistency check
    n_fu_by_comp = {}
    for series, comp, opath in tasks:
        r = by_path[opath]
        if r["n_fu"] is None:
            continue
        n_fu_by_comp.setdefault(comp, set()).add(r["n_fu"])
    for comp, s in n_fu_by_comp.items():
        if len(s) > 1:
            print(f"[warn] {comp}: inconsistent formula-unit counts across o* dirs: {sorted(s)}")

    warned_nfu = set()
    rejected = []
    state_rows = []
    for series, comp, opath in tasks:
        r = by_path[opath]
        o_name = Path(opath).name
        delta_tag = delta_from_o_dirname(o_name)

        # --- formula units per cell ---
        n_fu = r["n_fu"]
        if n_fu is None:
            n_fu = n_fu_override if n_fu_override else (
                sorted(n_fu_by_comp.get(comp, {0}))[0] or None)
        if n_fu is None and comp not in warned_nfu:
            print(f"[warn] {comp}: cannot determine n_fu (no readable structure file "
                  f"and no --n-fu); S_config and G_state will be NaN")
            warned_nfu.add(comp)

        # --- vacancy count: prefer the structure, fall back to the o-tag ---
        nv = r["nv"]
        bad_cell = None
        if nv is None and (delta_tag is not None) and (n_fu is not None):
            nv = int(round(delta_tag * n_fu))
        elif nv is not None and delta_tag is not None and n_fu is not None:
            nv_tag = int(round(delta_tag * n_fu))
            if abs(nv - nv_tag) > nv_tag_tol:
                # not a rounding artefact: this cell is not the composition the
                # directory claims, so its energy belongs to nothing here
                bad_cell = (f"structure has nv={nv}, o-tag implies nv={nv_tag} "
                            f"(n_fu={n_fu})")
                rejected.append((series, comp, o_name, bad_cell))
            elif nv != nv_tag and (comp, o_name) not in warned_nfu:
                print(f"[warn] {comp}/{o_name}: structure has nv={nv} but the o-tag implies "
                      f"nv={nv_tag} (n_fu={n_fu}); within tolerance, using the structure")
                warned_nfu.add((comp, o_name))

        # --- delta: exact from nv/n_fu.  The o#### tag is a rounded label (e.g. nv=1 in a
        #     64 f.u. cell is delta=0.015625 but tags as o984 -> 0.016, a 2.4% error), so it
        #     must not be used for the physics when nv and n_fu are known. ---
        delta = (nv / n_fu) if (nv is not None and n_fu) else delta_tag

        # signed verbatim: --s-config-a decides the branch, so -2 and +2 differ
        S_conf = S_config_signed(delta, n_fu, a_signed) if n_fu is not None else None

        for T in temps:
            th = r["therm"].get(T, {})
            # a rejected cell keeps its row (so it is visible in state_series_raw.csv)
            # but contributes no energy, which is what drops it from every average
            E = None if bad_cell else r["E"]
            F = th.get("F (eV)"); H_vib = th.get("H (eV)")
            G_state = (E + F - T*S_conf) if (E is not None and F is not None
                                             and S_conf is not None) else None
            H_state = (E + H_vib) if (E is not None and H_vib is not None) else None
            state_rows.append({
                "series": series, "comp_dir": comp, "o": o_name, "Temperature": T,
                "n_fu": n_fu, "nv": nv, "delta": delta, "delta_from_o_tag": delta_tag,
                "S_config_eV_per_K": S_conf,
                # F is kept so G_state can be recomputed for another S_config sign
                # without re-reading thousands of thermal_properties.yaml files
                "F_state": F, "H_vib_eV": H_vib,
                "E_state": E, "H_state": H_state, "G_state": G_state,
                "rejected": bad_cell or "",
            })

    if rejected:
        print(f"[warn] {len(rejected)} structure(s) rejected: the cell on disk is not the "
              f"composition its directory claims, so the energy is discarded and the "
              f"point averages over the remaining sets:")
        for series, comp, o_name, why in rejected[:10]:
            print(f"      {series}/{comp}/{o_name}: {why}")
        if len(rejected) > 10:
            print(f"      ... and {len(rejected) - 10} more (see the `rejected` column "
                  f"in state_series_raw.csv)")
    return pd.DataFrame(state_rows), n_fu_by_comp

def recompute_state_G(df_state, a_signed):
    """
    Re-derive S_config and G_state = E + F - T*S_config for a different signed `a`.

    Everything else (E, F, H) is independent of S_config, so the whole downstream chain
    can be replayed from here -- which is what makes the sign comparison exact rather
    than an estimate.
    """
    df = df_state.copy()
    delta = pd.to_numeric(df["delta"], errors="coerce")
    n_fu = pd.to_numeric(df["n_fu"], errors="coerce")
    E = pd.to_numeric(df["E_state"], errors="coerce")
    F = pd.to_numeric(df["F_state"], errors="coerce")
    T = pd.to_numeric(df["Temperature"], errors="coerce")

    cache = {}
    S = []
    for d, n in zip(delta, n_fu):
        key = (d, n)
        if key not in cache:
            cache[key] = (np.nan if (pd.isna(d) or pd.isna(n))
                          else S_config_signed(d, n, a_signed))
        S.append(cache[key])
    df["S_config_eV_per_K"] = pd.to_numeric(pd.Series(S, index=df.index),
                                            errors="coerce")
    df["G_state"] = E + F - T * df["S_config_eV_per_K"]
    return df

# ---------- Averaging & pair deltas ----------

def complete_series_only(df_state):
    """
    Keep only the series that succeeded for EVERY o* directory of their composition.

    The default average takes each (comp, o, T) point over whatever succeeded there, so
    a pair delta can subtract a 49-set mean from a 50-set one.  That is the right answer
    when the failures are random, and it is what --complete-sets-only exists to test:
    here a composition is reduced to the series that are complete across its whole
    o-series, so every point in one ΔG fit comes from the same set of realisations.
    """
    ok = df_state["G_state"].apply(lambda v: np.isfinite(v) if v is not None else False)
    d = df_state.assign(_ok=ok)
    keep = []
    for comp, g in d.groupby("comp_dir"):
        need = set(zip(g["o"], g["Temperature"]))
        for series, gs in g.groupby("series"):
            have = set(zip(gs.loc[gs["_ok"], "o"], gs.loc[gs["_ok"], "Temperature"]))
            if have >= need:
                keep.append((comp, series))
    keep = set(keep)
    m = [(c, s) in keep for c, s in zip(d["comp_dir"], d["series"])]
    return df_state[pd.Series(m, index=df_state.index)]


def average_across_series(df_state, avg_mode, all_series=None):
    """
    Average the independent SQS series at fixed (comp_dir, o, T).

    **The divisor is the number of series that produced a usable G_state, not the number
    that were launched.**  With 50 sets of which one has no energy or no phonons, the
    mean is over 49 and the row records n_series_total=50, n_series_used=49 and the
    names of the series that dropped out, so a thin average is visible in the CSV
    instead of silently widening the error bar.

    E, H and G share one mask -- the series where G is finite -- so the three averages
    always describe the same subset of realisations and ΔH - TΔS stays consistent with
    ΔG.  A series that has an energy but no phonons contributes to none of them.
    """
    # the universe of series is the whole tree, passed in by the caller: a set removed
    # by --complete-sets-only is still a set that did not contribute, and reporting
    # "49 of 49" for it would hide exactly what the flag was used to find out
    rows = []
    all_series = sorted(all_series if all_series is not None
                        else df_state["series"].unique())
    n_total_default = len(all_series)
    for (comp, o, T), g in df_state.groupby(["comp_dir", "o", "Temperature"]):
        Gs = g["G_state"].to_numpy(dtype=float)
        if avg_mode == "boltz":
            ws, mask, n_eff = boltz_weights(Gs, T)
        else:
            mask = np.isfinite(Gs)
            ws = (np.ones(mask.sum(), dtype=float) / mask.sum()) if mask.any() \
                 else np.zeros(0, dtype=float)
            n_eff = float(mask.sum()) if mask.any() else np.nan

        if not mask.any():
            E_b = H_b = G_b = E_s = H_s = G_s = np.nan
            g_spread = np.nan
        else:
            E_vals = g["E_state"].to_numpy(dtype=float)[mask]
            H_vals = g["H_state"].to_numpy(dtype=float)[mask]
            G_vals = g["G_state"].to_numpy(dtype=float)[mask]
            E_b, E_s = wmean_wstd(E_vals, ws) if np.isfinite(E_vals).any() else (np.nan, np.nan)
            H_b, H_s = wmean_wstd(H_vals, ws) if np.isfinite(H_vals).any() else (np.nan, np.nan)
            G_b, G_s = wmean_wstd(G_vals, ws) if np.isfinite(G_vals).any() else (np.nan, np.nan)
            g_spread = float(np.nanmax(G_vals) - np.nanmin(G_vals)) if G_vals.size else np.nan

        first = g.iloc[0]
        # every series that exists in the tree but did not yield a G here: both the
        # ones whose directory is present and failed, and the ones with no directory
        used_series = set(g.loc[mask, "series"]) if mask.any() else set()
        missing = [s for s in all_series if s not in used_series]
        rows.append({
            "comp_dir": comp, "o": o, "Temperature": T,
            "n_fu": first["n_fu"], "nv": first["nv"], "delta": first["delta"],
            "n_series": int(len(g)), "n_series_used": int(mask.sum()),
            "n_series_total": n_total_default,
            "missing_series": ";".join(missing),
            "n_eff": n_eff, "G_spread_eV": g_spread,
            "E_bavg": E_b, "H_bavg": H_b, "G_bavg": G_b,
            "E_bavg_wstd": E_s, "H_bavg_wstd": H_s, "G_bavg_wstd": G_s,
        })
    return pd.DataFrame(rows)

def build_pair_deltas(df_bavg, muG_table, muH_table, strict_mu=True, want_EH=False):
    """
    Adjacent-pair deltas with μ scaled by the pair's actual Δnv.

    'hi' is the o-tag with MORE oxygen (fewer vacancies); 'lo' has more vacancies.
    Removing Δnv oxygens from the cell releases Δnv/2 O2, and the μ tables are per
    ½O2, so the correction is Δnv * μ.

    ΔG is the target; ΔE and ΔH are computed only when want_EH is set.  Without it the
    μ_H table is never consulted, so a temperature it does not cover cannot abort a run
    that was never going to report an enthalpy.
    """
    pair_rows = []
    for comp, d_comp in df_bavg.groupby("comp_dir"):
        for T, d_T in d_comp.groupby("Temperature"):
            muG = mu_at_T(muG_table, T, name="mu_G", strict=strict_mu)
            muH = mu_at_T(muH_table, T, name="mu_H", strict=strict_mu) if want_EH else np.nan
            ordered = sorted(((int(x[1:]), x) for x in d_T["o"].unique()),
                             key=lambda t: t[0], reverse=True)
            for i in range(len(ordered) - 1):
                hi, lo = ordered[i][1], ordered[i+1][1]
                ra = d_T[d_T["o"] == hi]
                rb = d_T[d_T["o"] == lo]
                if ra.empty or rb.empty:
                    continue
                a = ra.iloc[0]; b = rb.iloc[0]

                n_fu_val = float(a["n_fu"]) if pd.notna(a["n_fu"]) else np.nan

                def _nv_of(row, tag):
                    """Exact vacancy count, falling back to the rounded o#### tag."""
                    if pd.notna(row["nv"]):
                        return int(round(float(row["nv"])))
                    if np.isfinite(n_fu_val):
                        return int(round((1000 - int(tag[1:])) / 1000.0 * n_fu_val))
                    return None

                nv_hi, nv_lo = _nv_of(a, hi), _nv_of(b, lo)

                # Δnv for this pair: from vacancy counts if known, else from the o-tags
                if nv_hi is not None and nv_lo is not None:
                    dnv = int(round(nv_lo - nv_hi))
                elif np.isfinite(n_fu_val):
                    dnv = int(round((int(hi[1:]) - int(lo[1:])) / 1000.0 * n_fu_val))
                else:
                    dnv = 1
                if dnv <= 0:
                    dnv = 1

                # x for the line plots and the ΔG=0 threshold: the vacancy fraction of
                # the more-deficient member, so o984-o1000 -> 1/64 and o516-o531 -> 31/64
                x_pair = (nv_lo / n_fu_val) if (nv_lo is not None
                                                and np.isfinite(n_fu_val)
                                                and n_fu_val) else np.nan

                def _d(col):
                    return (b[col] - a[col]) if (np.isfinite(b[col]) and np.isfinite(a[col])) \
                           else np.nan
                dG = _d("G_bavg")
                dE = _d("E_bavg") if want_EH else np.nan
                dH = _d("H_bavg") if want_EH else np.nan
                dG_tot = dG + dnv * muG if np.isfinite(dG) else np.nan
                dH_tot = dH + dnv * muH if np.isfinite(dH) else np.nan

                def _prop(col):
                    if np.isfinite(a[col]) and np.isfinite(b[col]):
                        return math.sqrt(a[col]**2 + b[col]**2)
                    return np.nan

                # a pair is only as well averaged as its thinner member
                n_used = [a.get("n_series_used"), b.get("n_series_used")]
                n_used = [int(v) for v in n_used if pd.notna(v)]

                pair_rows.append({
                    "comp_dir": comp, "Temperature": T, "Pair": f"{lo} - {hi}",
                    "n_series_used": min(n_used) if n_used else np.nan,
                    "n_series_total": a.get("n_series_total", np.nan),
                    "dnv": dnv, "n_fu": a["n_fu"],
                    "nv_hi": nv_hi, "nv_lo": nv_lo, "x_pair": x_pair,
                    "mu_G_used": dnv * muG, "mu_H_used": dnv * muH,
                    "ΔE_bavg": dE, "ΔH_bavg": dH_tot, "ΔG_bavg": dG_tot,
                    "ΔG_bavg_wstd": _prop("G_bavg_wstd"),
                    "ΔH_bavg_wstd": _prop("H_bavg_wstd"),
                })
    df = pd.DataFrame(pair_rows)
    if not df.empty:
        df["_hi"] = df["Pair"].map(lambda p: pair_to_lo_hi(p)[1])
        df = (df.sort_values(["comp_dir", "Temperature", "_hi"],
                             ascending=[True, True, False])
                .drop(columns=["_hi"])
                .reset_index(drop=True))
    return df

# ---------- Main ----------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Averaging → pairwise deltas → heatmaps/lines + continuous ΔG=0 threshold.")
    ap.add_argument("--parent", required=True, help="Directory containing the series dirs")
    ap.add_argument("--series-glob", default="sd_*",
                    help="Glob for series directories under --parent (default sd_*; use set_* for the new layout)")
    ap.add_argument("--inner-subdir", default="",
                    help="Optional subdir inside each series dir that contains <group>/<comp>/o* "
                         "(default: none)")
    ap.add_argument("--comp-prefix", default=None,
                    help="Optional filter on composition directory names. Not needed: "
                         "o* directories are found by their own name at any depth.")
    ap.add_argument("--temps", default=None, help="Comma-separated temperatures in K (e.g., 670,1070)")
    ap.add_argument("--temp", dest="temps", help=argparse.SUPPRESS)
    ap.add_argument("--outdir", default=None, help="Output directory (default: <parent>/_state_bavg_regci/)")
    ap.add_argument("--nprocs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="Processes used to read energy.txt / thermal_properties.yaml")

    # Family -- auto-detected; the flag is a fallback for unparseable directory names
    ap.add_argument("--elements", default=None,
                    help="A_base,A_dopant,B_base,B_dopant (e.g. Sr,Ca,Fe,Ti). Normally "
                         "unnecessary: the four element symbols are read off each "
                         "composition directory name, so several families can share one "
                         "tree. Give this only to override names that will not parse.")

    # Thermodynamics
    ap.add_argument("--s-config-a", type=float, default=-2.0,
                    help="SIGNED a in dS_config = a*R*[2d ln2d + (1-2d)ln(1-2d)]. "
                         "The sign is used verbatim, so -2 and +2 are different "
                         "runs: -2 (the default) is the physical branch with "
                         "S_config >= 0, +2 is the sign as printed in the paper, "
                         "and 0 drops the term. A positive value is accepted "
                         "deliberately -- it is the unphysical branch, reported "
                         "as-is rather than silently corrected.")
    ap.add_argument("--sign-compare", dest="sign_compare", action="store_true",
                    help="Also run the S_config sign study (a<0 / a>0 / a=0) and write "
                         "s_config_sign_comparison.csv. Off by default: only the "
                         "physical a<0 branch is reported, which is what keeps the "
                         "headline CSV one row per composition and temperature.")
    ap.add_argument("--no-sign-compare", dest="sign_compare", action="store_false",
                    help=argparse.SUPPRESS)    # back-compat; this is now the default
    ap.set_defaults(sign_compare=False)
    ap.add_argument("--n-fu", type=int, default=None,
                    help="Override the formula-unit count per cell. Normally inferred from "
                         "the structure file in each o* dir (n_cations/2).")
    ap.add_argument("--mu-g", default="670:-5.1768,1070:-5.6699",
                    help="Gibbs mu per 1/2 O2, 'T:value,...' (default = the previous "
                         "hardcoded values, i.e. muO2/2)")
    ap.add_argument("--mu-h", default="670:-4.37519,1070:-4.304685",
                    help="Enthalpy mu per 1/2 O2, 'T:value,...'")
    ap.add_argument("--allow-mu-extrapolation", action="store_true",
                    help="Permit temperatures outside the mu tables (extrapolates instead of failing)")

    # Averaging across series
    ap.add_argument("--nv-tag-tol", type=int, default=1,
                    help="Reject a structure whose own vacancy count differs from its "
                         "o#### directory by more than this (default 1, which absorbs "
                         "the tag's rounding). A rejected structure counts as a failed "
                         "set. Use a huge value to disable the check.")
    ap.add_argument("--with-EH", dest="with_EH", action="store_true",
                    help="Also compute and plot ΔE and ΔH. Off by default: ΔG=0 is the "
                         "result this pipeline reports, so ΔE/ΔH, their line plots and "
                         "the ΔG<0 & ΔH>0 filter are all skipped and the mu_H table is "
                         "not consulted.")
    ap.add_argument("--n-sets", type=int, default=None,
                    help="Use only the FIRST n series in name order (set_01..set_n) and "
                         "ignore the rest, for checking how the answer moves with the "
                         "number of realisations. The n selected become the universe, so "
                         "n_sets_total reports n, not the tree's count. Point --outdir "
                         "somewhere separate or this overwrites the full-tree results.")
    ap.add_argument("--complete-sets-only", action="store_true",
                    help="Restrict each composition to the series that succeeded at "
                         "every one of its o* directories, so one ΔG fit uses one set "
                         "of realisations throughout. Default is to average each point "
                         "over whatever succeeded there (50 sets, 1 failure -> 49).")
    ap.add_argument("--full-csv", action="store_true",
                    help="Also write the diagnostic CSVs (per-pair deltas, regression "
                         "predictions, per-pressure threshold dump, filtered pairs). "
                         "By default only the simple summary and the two state tables "
                         "are written.")
    ap.add_argument("--avg-mode", choices=["boltz", "simple"], default="simple",
                    help="'simple' = unweighted mean (default); 'boltz' = Boltzmann weights "
                         "from G_state, which collapses onto the lowest-G series (watch n_eff).")

    ap.add_argument("--fit-hi-min", type=int, default=None,
                    help="Keep only pairs with hi >= this value for regressions/thresholds.")

    # Ranges
    ap.add_argument("--vmin", type=float, default=-2.0, help="Default lower bound (heatmap + lines)")
    ap.add_argument("--vmax", type=float, default= 2.0, help="Default upper bound (heatmap + lines)")
    ap.add_argument("--heatmap-vmin", type=float, default=None, help="Override heatmap colorbar min")
    ap.add_argument("--heatmap-vmax", type=float, default=None, help="Override heatmap colorbar max")
    ap.add_argument("--line-ymin", type=float, default=None, help="Override line-plot y min")
    ap.add_argument("--line-ymax", type=float, default=None, help="Override line-plot y max")

    # Heatmaps -- OPT-IN
    ap.add_argument("--heatmaps", action="store_true",
                    help="Draw the per-pair heatmaps. OFF by default (they are one PNG "
                         "per pair per metric per temperature).")
    ap.add_argument("--heatmap-which", default=None,
                    help="Metrics for heatmaps from E,H,G (default G, or G,H with --with-EH)")
    ap.add_argument("--heatmap-source", choices=["raw","reg"], default="raw")
    ap.add_argument("--heatmap-no-annot", action="store_true")

    # Crossing heatmap
    ap.add_argument("--crossing-heatmaps", action="store_true")
    ap.add_argument("--crossing-which", default=None)
    ap.add_argument("--cross-annot", action="store_true")

    # Line plots
    ap.add_argument("--line-which", default=None,
                    help="Metrics for the line plots (default G, or G,H with --with-EH)")
    ap.add_argument("--skip-lines", action="store_true")
    ap.add_argument("--skip-heatmaps", action="store_true",
                    help=argparse.SUPPRESS)   # back-compat; heatmaps are off by default now
    ap.add_argument("--x-tick-mode", choices=["auto","frac","nv"], default="auto",
                    help="x tick labels on the line plots: 'auto' plain numbers, "
                         "'frac' n/64, 'nv' the integer vacancy count")
    ap.add_argument("--plot-pO2-lines", action="store_true",
                    help="Overlay the pressure-shifted ΔG fits (paper eq. 1) on the ΔG "
                         "line plots and mark each ΔG=0 crossing")

    # Bands & regression
    ap.add_argument("--band-type", choices=["std","ci","none"], default="std")
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument("--ci-mode", choices=["mean","pred"], default="mean")
    ap.add_argument("--dpi", type=int, default=200)

    # Continuous threshold of ΔG=0
    ap.add_argument("--x-mode", choices=["nv","hi","lo","mid","delta"], default="nv",
                    help="x used for the ΔG=0 regression. 'nv' (default) = nv_lo/n_fu, "
                         "the same numeric axis the line plots now use, so x0 is a "
                         "vacancy fraction. The others are the legacy tag-derived modes.")
    ap.add_argument("--thresholds-csv", default=None,
                    help="Optional legacy per-line threshold dump (x-mode driven)")
    ap.add_argument("--thresholds-group", choices=["by_A","by_B","both","by_SrCa","by_FeMn"],
                    default="by_A")
    ap.add_argument("--reg-csv", default=None)

    # ΔG=0 threshold vs oxygen partial pressure  (paper eq. 1)
    ap.add_argument("--pO2-list", default="1,0.8,0.00003",
                    help="Oxygen partial pressures in atm at which to report the ΔG=0 "
                         "crossing (default 1,0.8,0.00003 -- the paper's values)")
    ap.add_argument("--pO2-diff", default="0.00003,0.8",
                    help="'p_lo,p_hi': the two pressures whose x0 difference is reported "
                         "as dx0 = x0(p_lo) - x0(p_hi)")

    args = ap.parse_args(argv)

    elements_override = None
    if args.elements:
        elements_override = tuple(s.strip() for s in args.elements.split(","))
        if len(elements_override) != 4 or not all(elements_override):
            raise SystemExit("--elements needs exactly four symbols: "
                             "A_base,A_dopant,B_base,B_dopant")

    # ΔG is the deliverable; ΔE/ΔH ride along only when asked for
    default_which = "G,H" if args.with_EH else "G"
    args.line_which = args.line_which or default_which
    args.heatmap_which = args.heatmap_which or default_which
    args.crossing_which = args.crossing_which or default_which

    temps = (670, 1070) if args.temps is None else tuple(
        int(t.strip()) for t in args.temps.split(",") if t.strip())

    parent = Path(args.parent).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (parent/"_state_bavg_regci")
    outdir.mkdir(parents=True, exist_ok=True)

    muG_table = parse_mu_table(args.mu_g)
    muH_table = parse_mu_table(args.mu_h)
    strict_mu = not args.allow_mu_extrapolation

    hv_min = args.heatmap_vmin if args.heatmap_vmin is not None else args.vmin
    hv_max = args.heatmap_vmax if args.heatmap_vmax is not None else args.vmax
    ly_min = args.line_ymin if args.line_ymin is not None else args.vmin
    ly_max = args.line_ymax if args.line_ymax is not None else args.vmax

    pressures = [float(s) for s in args.pO2_list.split(",") if s.strip()]
    p_diff = [float(s) for s in args.pO2_diff.split(",") if s.strip()]
    if len(p_diff) != 2:
        raise SystemExit("--pO2-diff needs exactly two pressures, 'p_lo,p_hi'")
    p_lo, p_hi = p_diff
    for p in p_diff:                       # the difference needs both fits
        if not any(abs(p - q) < 1e-15 for q in pressures):
            pressures.append(p)
    pressures = sorted(set(pressures), reverse=True)

    print(f"[*] temps: {temps}   avg-mode: {args.avg_mode}   "
          f"a = {args.s_config_a:+g} ({s_config_label(args.s_config_a)})")
    if args.s_config_a > 0:
        print("[warn] a > 0 is the sign as printed in the paper, which makes "
              "S_config <= 0: vacancy disorder then RAISES dG. Reported as asked, "
              "but the physical branch is a < 0.")
    print(f"[*] pO2 (atm): {pressures}   dx0 = x0({p_lo:g}) - x0({p_hi:g})")

    # 1) Collect state-level E/H/G per series
    df_state, n_fu_by_comp = collect_state_rows(
        parent, args.series_glob, args.inner_subdir, temps, elements_override,
        args.s_config_a, args.n_fu, args.nprocs, comp_prefix=args.comp_prefix,
        nv_tag_tol=args.nv_tag_tol)
    if df_state.empty:
        print("[Error] No state-level data collected."); return

    # 1b) Families and their element symbols, read off the directory names
    all_comps = sorted(df_state["comp_dir"].unique())
    comp_info = {c: parse_comp_dir(c) for c in all_comps}
    unparsed = sorted(c for c, v in comp_info.items() if v is None)
    if unparsed and elements_override:
        for c in unparsed:                       # fall back to the explicit --elements
            a, b = parse_composition(c, elements_override)
            if a is not None:
                eb = elements_override
                comp_info[c] = {"elements": eb, "a_base": eb[0], "a_dopant": eb[1],
                                "b_base": eb[2], "b_dopant": eb[3],
                                "A_frac": a, "B_frac": b, "a_vac": 0.0, "b_vac": 0.0,
                                "family": f"{eb[0]}{eb[1]}_{eb[2]}{eb[3]}",
                                "formula": f"{eb[0]}{a:g}{eb[1]}{1-a:g}"
                                           f"{eb[2]}{b:g}{eb[3]}{1-b:g}O3-d"}
        unparsed = sorted(c for c, v in comp_info.items() if v is None)
    if unparsed:
        print(f"[warn] {len(unparsed)} composition dir(s) could not be parsed and are "
              f"dropped; e.g. {unparsed[:3]}.  Pass --elements A,Adop,B,Bdop if the "
              f"names do not follow <A###>_<Adop###>_<B###>_<Bdop###>.")
        df_state = df_state[~df_state["comp_dir"].isin(unparsed)]
        if df_state.empty:
            print("[Error] No parseable compositions left."); return
        comp_info = {c: v for c, v in comp_info.items() if v is not None}

    families = {}
    for c, v in comp_info.items():
        families.setdefault(v["family"], v["elements"])
    for fam, els in sorted(families.items()):
        n = sum(1 for v in comp_info.values() if v["family"] == fam)
        print(f"[*] family {fam}: A = {els[0]}/{els[1]}, B = {els[2]}/{els[3]}  "
              f"({n} composition(s), auto-detected)")

    all_series_names = sorted(df_state["series"].unique())   # before any filtering
    n_series_total = len(all_series_names)
    print(f"[*] {n_series_total} series (sets) in the tree")

    # --n-sets: keep the first n in name order and make them the universe.  Reporting
    # "n_sets_total = 50" while only 10 were allowed to contribute would describe a run
    # that did not happen, so the selected sets become the denominator too.
    if args.n_sets is not None:
        if args.n_sets < 1:
            raise SystemExit("--n-sets must be >= 1")
        if args.n_sets < n_series_total:
            dropped = n_series_total - args.n_sets
            all_series_names = all_series_names[:args.n_sets]
            df_state = df_state[df_state["series"].isin(all_series_names)]
            n_series_total = len(all_series_names)
            print(f"[*] --n-sets {args.n_sets}: using {all_series_names[0]}.."
                  f"{all_series_names[-1]}; the other {dropped} set(s) are ignored "
                  f"and n_sets_total reports {n_series_total}")
        else:
            print(f"[*] --n-sets {args.n_sets}: the tree has only {n_series_total} "
                  f"set(s), so all of them are used")

    if args.complete_sets_only:
        before = len(df_state)
        df_state = complete_series_only(df_state)
        if df_state.empty:
            print("[Error] --complete-sets-only left no data: no series is complete "
                  "across a whole o-series."); return
        # "complete" is per composition, so quote the range: a set that failed on one
        # composition is still usable for every other one
        per_comp = df_state.groupby("comp_dir")["series"].nunique()
        print(f"[*] --complete-sets-only: {per_comp.min()}-{per_comp.max()} complete "
              f"series per composition (of {n_series_total}); "
              f"{len(df_state)}/{before} state rows kept")

    seen_nfu = sorted({int(v) for s in n_fu_by_comp.values() for v in s})
    print(f"[*] formula units per cell detected: {seen_nfu if seen_nfu else 'none (using --n-fu)'}")
    df_state.to_csv(outdir / "state_series_raw.csv", index=False)

    # 2-3) Per S_config variant: average across series -> pair deltas -> ΔG=0 vs pO2.
    #      The physical result is variants[0] (a<0); the others exist only to quantify
    #      how much the printed sign changes the answer.
    variants = s_config_variants(args.s_config_a, enabled=args.sign_compare)
    primary_label = variants[0][0]
    thr_frames = []
    df_bavg = df_pairs = None

    for label, a_signed in variants:
        dfs = recompute_state_G(df_state, a_signed)   # no-op for the primary variant
        bavg = average_across_series(dfs, args.avg_mode, all_series=all_series_names)
        pairs = build_pair_deltas(bavg, muG_table, muH_table, strict_mu=strict_mu,
                                  want_EH=args.with_EH)
        if pairs.empty:
            print(f"[warn] {label}: no pairs built; skipped")
            continue

        pairs["family"] = pairs["comp_dir"].map(lambda c: comp_info[c]["family"])
        pairs["A_frac"] = pairs["comp_dir"].map(lambda c: comp_info[c]["A_frac"])
        pairs["B_frac"] = pairs["comp_dir"].map(lambda c: comp_info[c]["B_frac"])

        thr_frames.append(compute_pO2_thresholds(
            pairs, temps, pressures, variant_label=label, a_signed=a_signed,
            gcol="ΔG_bavg", x_col="x_pair", fit_hi_min=args.fit_hi_min))

        if label == primary_label:
            df_bavg, df_pairs = bavg, pairs
            if args.avg_mode == "boltz":
                finite = bavg["n_eff"].dropna()
                if not finite.empty:
                    print(f"[*] Boltzmann n_eff: min={finite.min():.3f} "
                          f"median={finite.median():.3f} max={finite.max():.3f}  "
                          f"(n_eff -> 1 means the average IS the minimum)")
            bad = pairs[pairs["dnv"] != 1]
            if not bad.empty:
                print(f"[warn] {len(bad)} pair(s) span dnv != 1 (a missing o* directory); "
                      f"mu and the pO2 correction are scaled accordingly. See the dnv column.")

            # how many sets each point was actually averaged over
            nu = pd.to_numeric(bavg["n_series_used"], errors="coerce")
            thin = bavg[nu < n_series_total]
            print(f"[*] set averaging: {len(bavg) - len(thin)} of {len(bavg)} "
                  f"(comp, o, T) points used all {n_series_total} set(s); "
                  f"{len(thin)} averaged over fewer (divisor = the sets that succeeded)")
            if not thin.empty:
                for _, r in thin.sort_values("n_series_used").head(8).iterrows():
                    print(f"      {r['comp_dir']}/{r['o']} T={r['Temperature']:g}K  "
                          f"n={int(r['n_series_used'])}/{n_series_total}  "
                          f"missing {r['missing_series']}")
                if len(thin) > 8:
                    print(f"      ... and {len(thin) - 8} more "
                          f"(see n_series_used in state_boltz_bavg.csv)")
                dead = bavg[nu == 0]
                if not dead.empty:
                    print(f"[warn] {len(dead)} point(s) have NO usable set at all; "
                          f"the pairs touching them are dropped from every fit")

    if df_pairs is None:
        print("[Error] No pairs built."); return
    df_bavg.to_csv(outdir / "state_boltz_bavg.csv", index=False)
    if args.full_csv:
        drop = [] if args.with_EH else [c for c in df_pairs.columns
                                        if c.startswith(("ΔE", "ΔH")) or c == "mu_H_used"]
        df_pairs.drop(columns=drop).to_csv(
            outdir / "pair_deltas_from_state_bavg.csv", index=False)

    # 3c) ΔG=0 thresholds vs pO2, their pressure-to-pressure difference, and the
    #      S_config sign comparison
    df_thr = pd.concat(thr_frames, ignore_index=True) if thr_frames else pd.DataFrame()
    simple_path = outdir / "dG0_vacancy_summary.csv"
    if not df_thr.empty:
        # the headline table: vacancy count at ΔG = 0, one column per pressure
        df_simple = build_simple_summary(df_thr, pressures, primary_label, comp_info)
        df_simple.to_csv(simple_path, index=False)

        x_cols = [f"x_{fmt_pressure(p)}" for p in pressures if f"x_{fmt_pressure(p)}"
                  in df_simple.columns]
        print(f"[*] ΔG=0 crossing as the oxygen vacancy fraction x = n_v/n_fu "
              f"(median over compositions; negative crossings floored at 0):")
        for T, g in df_simple.groupby("Temperature_K", sort=True):
            cells = "  ".join(f"{c.replace('x_', '')}={g[c].median():.4f}" for c in x_cols)
            print(f"      {T:>6} K   {cells}")

        if args.full_csv:
            df_thr.to_csv(outdir / "thresholds_DG_pO2.csv", index=False)
            df_sum = summarize_pO2_thresholds(df_thr, p_lo, p_hi)
            df_sum.to_csv(outdir / "thresholds_DG_pO2_summary.csv", index=False)
            if args.sign_compare and df_sum["s_config_variant"].nunique() > 1:
                df_cmp = compare_s_config_signs(df_sum, p_lo, p_hi, reference=primary_label)
                df_cmp.to_csv(outdir / "s_config_sign_comparison.csv", index=False)
                print(f"[*] S_config sign comparison -> s_config_sign_comparison.csv "
                      f"({len(df_cmp)} rows, reference = {primary_label})")

    # 3b) Regression predictions + (ΔG<0 & ΔH>0) filter -- diagnostics only
    reg_metrics = ("ΔE_bavg", "ΔH_bavg", "ΔG_bavg") if args.with_EH else ("ΔG_bavg",)
    df_pairs_reg = add_regressed_pair_values(df_pairs, metrics=reg_metrics)
    use_reg = (args.heatmap_source == "reg")
    if args.full_csv:
        df_pairs_reg.to_csv(outdir / "pair_deltas_regression_pred.csv", index=False)
        # the ΔG<0 & ΔH>0 screen needs an enthalpy, so it only exists with --with-EH
        if args.with_EH:
            colG = "ΔG_bavg_reg" if use_reg else "ΔG_bavg"
            colH = "ΔH_bavg_reg" if use_reg else "ΔH_bavg"
            df_aug = df_pairs_reg if use_reg else df_pairs
            mask = (pd.to_numeric(df_aug[colG], errors="coerce") < 0) & \
                   (pd.to_numeric(df_aug[colH], errors="coerce") > 0)
            cols = ["Temperature","Pair","comp_dir","A_frac","B_frac","dnv", colG, colH]
            df_aug.loc[mask, cols].sort_values(["Temperature","A_frac","B_frac"]).to_csv(
                outdir / f"filtered_pairs_{'reg' if use_reg else 'raw'}_dGlt0_dHgt0.csv",
                index=False)

    fit_hi_min = args.fit_hi_min
    label_map = {"E": "ΔE_bavg", "H": "ΔH_bavg", "G": "ΔG_bavg"}

    # Every plot below is per family: the axis labels are that family's own element
    # symbols, and one figure must never mix Fe/Mn with Fe/Ti compositions.
    multi_fam = len(families) > 1
    def _fam_slices(df):
        for fam, els in sorted(families.items()):
            sub = df[df["family"] == fam]
            if not sub.empty:
                yield fam, els, sub

    # 4) Heatmaps -- opt-in via --heatmaps
    if args.heatmaps and not args.skip_heatmaps:
        metrics = [s.strip().upper() for s in args.heatmap_which.split(",") if s.strip()]
        df_used = df_pairs_reg if use_reg else df_pairs
        def c_label(T): return f"{int(T-273)}°C" if T > 273 else f"{T}K"
        for fam, els, df_fam in _fam_slices(df_used):
            target = str(outdir / fam) if multi_fam else str(outdir)
            for T in temps:
                dT = df_fam[df_fam["Temperature"] == T]
                for pair in sorted(dT["Pair"].unique(), key=pair_sort_key):
                    sub = dT[dT["Pair"] == pair]
                    if sub.empty: continue
                    for m in metrics:
                        src_col = label_map.get(m)
                        if src_col is None: continue
                        if use_reg: src_col += "_reg"
                        if src_col not in sub.columns: continue
                        plot_heatmap(
                            sub, src_col,
                            title=f"{m} ({'regression' if use_reg else 'state avg'}) "
                                  f"{c_label(T)} {pair}",
                            filename=f"heatmap_{m}_{args.heatmap_source}_"
                                     f"{pair.replace(' ','_')}_{T}K.png",
                            target_path=target, elements=els,
                            vmin=hv_min, vmax=hv_max,
                            annotate=(not args.heatmap_no_annot), dpi=args.dpi)

    # 5) Crossing heatmaps
    if args.crossing_heatmaps:
        df_src = df_pairs_reg if use_reg else df_pairs
        for cm in [s.strip().upper() for s in args.crossing_which.split(",") if s.strip()]:
            if cm not in ("G", "H"): continue
            col = label_map[cm] + ("_reg" if use_reg else "")
            if col not in df_src.columns: continue
            for fam, els, df_fam in _fam_slices(df_src):
                sub_out = outdir / "crossing_heatmaps" / (fam if multi_fam else "")
                plot_crossing_heatmap_metric(
                    df_fam, temps, outdir=str(sub_out),
                    metric_col=col, title_prefix=f"Δ{cm}", elements=els,
                    vmin=0.50, vmax=1.00, annotate=args.cross_annot, dpi=args.dpi,
                    h_all_pos_to_one=(cm == "H"))

    # 6) Line plots
    if not args.skip_lines:
        reg_rows = []
        for m in [s.strip().upper() for s in args.line_which.split(",") if s.strip()]:
            if m not in ("G", "H"): continue
            for fam, els, df_fam in _fam_slices(df_pairs):
                base = outdir / "line_plots" / (fam if multi_fam else "")
                line_plots_for_metric(
                    df_fam, temps, outdir_base=str(base), metric=m,
                    elements=els, ymin=ly_min, ymax=ly_max,
                    band_type=args.band_type, ci_level=args.ci_level, ci_mode=args.ci_mode,
                    reg_csv_rows=reg_rows, dpi=args.dpi, fit_hi_min=fit_hi_min,
                    x_tick_mode=args.x_tick_mode,
                    pO2_list=(pressures if args.plot_pO2_lines else ()))
        if args.reg_csv:
            Path(args.reg_csv).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(reg_rows).to_csv(args.reg_csv, index=False)

    # 7) Continuous ΔG=0 thresholds
    if args.thresholds_csv:
        group = {"by_SrCa": "by_A", "by_FeMn": "by_B"}.get(
            args.thresholds_group, args.thresholds_group)
        df_thresh = df_pairs_reg if use_reg else df_pairs
        if group in ("by_A", "both"):
            compute_continuous_thresholds_for_DG(
                df_thresh, temps, out_csv_path=args.thresholds_csv,
                x_mode=args.x_mode, group_mode="by_A", fit_hi_min=fit_hi_min)
        if group in ("by_B", "both"):
            p = Path(args.thresholds_csv)
            compute_continuous_thresholds_for_DG(
                df_thresh, temps, out_csv_path=str(p.with_name(p.stem + "_byB" + p.suffix)),
                x_mode=args.x_mode, group_mode="by_B", fit_hi_min=fit_hi_min)

    print(f"[*] Done. Outputs in: {outdir}")
    print(f"    key file:  dG0_vacancy_summary.csv  (vacancy count at ΔG=0 per pO2)")
    print( "               state_boltz_bavg.csv     (the set average + n_series_used)")
    if not args.skip_lines:
        print("               line_plots/              (ΔG vs vacancy fraction)")
    if not args.full_csv:
        print("    diagnostic CSVs skipped; pass --full-csv for the per-pair dumps")
    if not args.with_EH:
        print("    ΔE/ΔH skipped; pass --with-EH to compute and plot them")
    if not args.heatmaps:
        print("    heatmaps skipped; pass --heatmaps to draw them")

if __name__ == "__main__":
    main()
