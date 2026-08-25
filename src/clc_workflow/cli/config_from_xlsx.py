#!/usr/bin/env python3
"""
Turn the compounds named in a measurement spreadsheet into a `families:` block.

    clc config-from-xlsx measured.xlsx [more.xlsx ...] [--config config.yaml]
    clc config-from-xlsx measured.xlsx --n-a-sites 80

Prints YAML on stdout, with a comment above each line naming the compounds it covers.
Paste it into config.yaml -- nothing is written for you, because the grouping is a
judgement you should read before you generate thousands of structures from it.

WHY THIS IS NOT ONE BLOCK PER COMPOUND.  A family block is an x_values x y_values CROSS
PRODUCT.  Two compounds at (x=0.25, y=0.125) and (x=0.50, y=0.25) cannot share a block:
x:[0.25,0.50] x y:[0.125,0.25] is four compounds, and two of them are ones nobody
measured.  So compounds are grouped by everything EXCEPT y -- same dopant pair, same x,
same vacancies -- and each group becomes one block listing its y values.  That is exactly
the shape config_feco.yaml arrived at by hand, 41 compounds in 21 blocks.

WHY x = 0 IS SPECIAL.  At x = 0 the A dopant is absent, so SrFeO3 written under a Ba
family and under a Ca family is the SAME structure generated twice with two directory
names.  All x = 0 compounds are therefore assigned to a single A dopant, and which one is
reported rather than silently chosen.

The A/B ratio decides vacancies, and it is not symmetric: A/B above 1 means the A
sublattice is full and B is short, so ABO3 with A/B = 1.05 is written A B0.952 O3 and gets
b_vac -- never an A excess, because no lattice holds 1.05 A per B on a full sublattice.
"""
import argparse
import sys
from collections import Counter, defaultdict

from clc_workflow.cli.build_delta_dataset import (A_SITE, B_SITE, _INVISIBLE,
                                                  clean_formula,
                                                  parse_formula, read_all_measurements,
                                                  site_fractions, _sheet_arg)

# Fractions are compared and printed at this many decimals.  The sheet writes 0.875 and
# 1/8 exactly, so 6 is far more than needed and still safe against float noise.
ND = 6

# What an undoped sheet falls back to, matching clc_config's own family defaults so a
# generated block and a hand-written one agree on what "no dopant named" means.
DEFAULT_A_DOPANT = "Ca"
DEFAULT_B_DOPANT = "Mn"


def compound_params(formula, a_base, b_base):
    """
    One formula -> (a_dopant, x, a_vac, b_dopant, y, b_vac), or a reason it cannot be one.

    x and y are fractions of the A and B SITES, matching config.yaml's meaning, and the
    base species takes 1 - x - vac.
    """
    counts = parse_formula(formula)
    a_frac, b_frac = site_fractions(counts)

    def one_site(frac, base, label):
        dopants = {el: v for el, v in frac.items() if el != base and v > 0}
        if len(dopants) > 1:
            raise ValueError(
                f"{len(dopants)} {label}-site dopants ({', '.join(sorted(dopants))}); "
                f"a family block holds one")
        if base not in frac and not dopants:
            raise ValueError(f"no {label}-site species")
        dop, val = next(iter(dopants.items()), (None, 0.0))
        vac = 1.0 - sum(frac.values())
        # Negative vacancy means that sublattice is the FULL one and the other is short;
        # site_fractions already scaled by the fuller sublattice, so this is float noise.
        return dop, round(val, ND), round(max(vac, 0.0), ND)

    a_dop, x, a_vac = one_site(a_frac, a_base, "A")
    b_dop, y, b_vac = one_site(b_frac, b_base, "B")
    return a_dop, x, a_vac, b_dop, y, b_vac


def pick_bases(formulas, a_base, b_base):
    """The most common A and B species become the bases, unless the user named them."""
    a_count, b_count = Counter(), Counter()
    for f in formulas:
        try:
            counts = parse_formula(f)
        except ValueError:
            continue
        for el, v in counts.items():
            if el in A_SITE and v > 0:
                a_count[el] += 1
            elif el in B_SITE and v > 0:
                b_count[el] += 1
    a = a_base or (a_count.most_common(1)[0][0] if a_count else None)
    b = b_base or (b_count.most_common(1)[0][0] if b_count else None)
    if not a or not b:
        sys.exit("[ERROR] could not identify an A-site and a B-site base species; "
                 "pass --a-base and --b-base")
    return a, b


def commensurate_problems(rows, n_sites):
    """Compounds whose site fractions do not land on whole atoms of an n_sites sublattice."""
    bad = []
    for r in rows:
        for frac, label in ((r["x"], "x"), (r["a_vac"], "a_vac"),
                            (r["y"], "y"), (r["b_vac"], "b_vac")):
            n = frac * n_sites
            if abs(n - round(n)) > 1e-6:
                bad.append((r["formula"], label, frac, n))
                break
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx", nargs="+", help="the measurement spreadsheet(s)")
    ap.add_argument("--sheet", default=[0], nargs="+",
                    help="sheet name or index; one for all, or one per file in order")
    ap.add_argument("--a-base", default=None,
                    help="A-site base species (default: the most common one in the sheet)")
    ap.add_argument("--b-base", default=None,
                    help="B-site base species (default: the most common one in the sheet)")
    ap.add_argument("--config", default=None,
                    help="a config.yaml to take sqs.supercell and base_poscar from, so "
                         "the compounds can be checked against the real site count")
    ap.add_argument("--n-a-sites", type=int, default=None,
                    help="A sites in the supercell, if you would rather say it directly "
                         "than point at a config (`clc sqs --dry-run` prints it)")
    ap.add_argument("--a-dopant", default=None,
                    help="force the A-site dopant name; only needed when the sheet has "
                         "no A dopant at all and a block still has to name one")
    ap.add_argument("--b-dopant", default=None,
                    help="force the B-site dopant name; see --a-dopant")
    ap.add_argument("--x0-dopant", default=None,
                    help="which A dopant carries the x=0 compounds (default: the one with "
                         "the most compounds, so the block is where you would look)")
    args = ap.parse_args(argv)

    meas = read_all_measurements(args.xlsx, [_sheet_arg(v) for v in args.sheet])
    # clean_formula drops the "O3-d" tail because delta is the label; keep the tidy
    # original alongside it so the generated comments name compounds the way you wrote them.
    display = {}
    for raw in meas["formula_raw"].unique():
        display.setdefault(clean_formula(raw), str(raw).translate(_INVISIBLE).strip())
    formulas = sorted(display)
    a_base, b_base = pick_bases(formulas, args.a_base, args.b_base)

    rows, skipped = [], []
    for f in formulas:
        try:
            a_dop, x, a_vac, b_dop, y, b_vac = compound_params(f, a_base, b_base)
        except ValueError as e:
            skipped.append((f, str(e)))
            continue
        rows.append({"formula": f, "a_dopant": a_dop, "x": x, "a_vac": a_vac,
                     "b_dopant": b_dop, "y": y, "b_vac": b_vac})

    if not rows:
        sys.exit("[ERROR] no compound in the spreadsheet could be expressed as a family")

    a_dopants = sorted({r["a_dopant"] for r in rows if r["a_dopant"]})
    b_dopants = sorted({r["b_dopant"] for r in rows if r["b_dopant"]})
    print(f"[*] {len(rows)} compound(s); base {a_base}/{b_base}, "
          f"A dopant(s) {', '.join(a_dopants) or 'none'}, "
          f"B dopant(s) {', '.join(b_dopants) or 'none'}", file=sys.stderr)

    # x = 0 has no A dopant, so it would otherwise be generated once per A dopant.
    x0 = [r for r in rows if r["x"] == 0 and r["a_vac"] == 0]
    if x0 and a_dopants:
        carrier = args.x0_dopant or Counter(
            r["a_dopant"] for r in rows if r["a_dopant"]).most_common(1)[0][0]
        if carrier not in a_dopants:
            sys.exit(f"[ERROR] --x0-dopant {carrier} is not one of: {', '.join(a_dopants)}")
        for r in x0:
            r["a_dopant"] = carrier
        print(f"[*] {len(x0)} compound(s) at x=0 assigned to the {carrier} family only, "
              f"so they are not built once per A dopant", file=sys.stderr)

    # A block must NAME a dopant even when no compound uses one -- an undoped sheet still
    # has to produce valid YAML.  The choice is inert while that site's fraction is 0, but
    # it stops being inert the moment someone adds a value, so say which name was picked
    # instead of letting `b_dopant: None` reach the file.
    for site, found, flag, fallback, key in (
            ("A", a_dopants, args.a_dopant, DEFAULT_A_DOPANT, "a_dopant"),
            ("B", b_dopants, args.b_dopant, DEFAULT_B_DOPANT, "b_dopant")):
        if found and not flag:
            chosen = None                      # per-compound dopants already set below
        else:
            chosen = flag or (found[0] if found else fallback)
            if not found:
                print(f"[warn] no {site}-site dopant anywhere in the sheet; blocks are "
                      f"written with {site.lower()}_dopant: {chosen}, which is inert while "
                      f"every {'x' if site == 'A' else 'y'} is 0.  --{site.lower()}-dopant "
                      f"names a different one.", file=sys.stderr)
        for r in rows:
            if chosen or not r[key]:
                r[key] = chosen or flag or (found[0] if found else fallback)

    # commensurability, if we were told the site count
    n_sites = args.n_a_sites
    if n_sites is None and args.config:
        from clc_workflow.clc_config import load_config
        from clc_workflow.sqs_generator import supercell_site_counts
        cfg = load_config(args.config)
        n_sites = supercell_site_counts(cfg["sqs"]["supercell"], cfg["base_poscar"])[0]
        print(f"[*] {n_sites} A sites from {args.config}", file=sys.stderr)
    if n_sites:
        bad = commensurate_problems(rows, n_sites)
        if bad:
            print(f"[warn] {len(bad)} compound(s) do not land on whole atoms of "
                  f"{n_sites} sites -- `clc sqs` will refuse these unless you enlarge the "
                  f"supercell or set sqs.require_commensurate: false", file=sys.stderr)
            for formula, label, frac, n in bad[:8]:
                print(f"       {display.get(formula, formula)}: {label}={frac:g} x {n_sites} = {n:.4f}",
                      file=sys.stderr)
            if len(bad) > 8:
                print(f"       ... and {len(bad) - 8} more", file=sys.stderr)

    for formula, why in skipped:
        print(f"[warn] skipped {display.get(formula, formula)}: {why}", file=sys.stderr)

    # group by everything except y, so no block generates a compound nobody measured
    groups = defaultdict(list)
    for r in rows:
        key = (r["a_dopant"], r["b_dopant"], r["x"], r["a_vac"], r["b_vac"])
        groups[key].append(r)

    print(f"[*] {len(groups)} family block(s) for {len(rows)} compound(s)\n",
          file=sys.stderr)

    def fmt(v):
        """Full stored precision, trailing zeros stripped.

        Not a fixed number of decimals: (Sr0.75Ba0.25)0.95 puts Ba at 0.2375, and
        rounding that to 0.237 asks gen_sqs for a compound nobody measured.
        """
        if not v:
            return "0.0"
        return f"{round(v, ND):.{ND}f}".rstrip("0").rstrip(".")

    print("families:")
    for (a_dop, b_dop, x, a_vac, b_vac), members in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        ys = sorted({m["y"] for m in members})
        names = ", ".join(display.get(m["formula"], m["formula"])
                          for m in sorted(members, key=lambda m: m["y"]))
        extra = ""
        if a_vac:
            extra += f", a_vac: {fmt(a_vac)}"
        if b_vac:
            extra += f", b_vac: {fmt(b_vac)}"
        print(f"  # {names}")
        print(f"  - {{a_dopant: {a_dop}, b_dopant: {b_dop}, "
              f"x_values: [{fmt(x)}], y_values: [{', '.join(fmt(v) for v in ys)}]{extra}}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
