#!/usr/bin/env python3
"""
Turn the compounds named in a measurement spreadsheet into a `families:` block.

    clc config-from-xlsx measured.xlsx --config config.yaml      # edits config.yaml
    clc config-from-xlsx measured.xlsx --config config.yaml --stdout   # just print it

With --config, the file's `families:` block and its a_base / b_base are REWRITTEN from
the compounds the spreadsheet names, and a config.yaml.bak is left beside it.  A comment
above each entry names the compounds it covers.

The edit is surgical -- the families block and two scalars, nothing else.  Loading the
YAML and dumping it back would be far shorter and would silently destroy every comment in
the file, which for this config is most of it.  The result is parsed before anything is
overwritten, so a bad edit fails instead of landing.

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
import re
import sys
from pathlib import Path

import yaml
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



# ------------------------------------------------------------------- editing config.yaml

def _find_block(lines, key):
    """
    (start, end) line indices of a top-level `key:` block, end exclusive.

    The block runs to the next top-level key -- a line starting at column 0 that is
    neither blank nor a comment.  Column-0 COMMENTS stay inside the block on purpose:
    config.yaml keeps commented-out family entries there, and those are old definitions
    being replaced, not a header for whatever comes next.
    """
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^{re.escape(key)}\s*:", ln):
            start = i
            break
    if start is None:
        return None, None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln[:1] not in ("", " ", "\t", "#") and re.match(r"^[A-Za-z_][\w-]*\s*:", ln):
            end = j
            break
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1                      # give the blank separator back to the next section
    return start, end


def _replace_scalar(lines, key, value):
    """Set a top-level `key: value`, keeping any trailing comment on that line."""
    for i, ln in enumerate(lines):
        m = re.match(rf"^({re.escape(key)}\s*:\s*)(.*?)(\s*#.*)?$", ln)
        if m:
            if m.group(2).strip() == str(value):
                return False
            lines[i] = f"{m.group(1)}{value}{m.group(3) or ''}"
            return True
    return None                        # key absent


def write_config(path, block_lines, a_base, b_base, backup=True):
    """
    Replace the `families:` block and the a_base/b_base lines of config.yaml in place.

    A surgical text edit, NOT a YAML round-trip: safe_dump would silently discard every
    comment in the file and reorder the keys, and this config is mostly comments.  Only
    the lines that have to change are touched; every other byte survives.
    """
    path = Path(path)
    original = path.read_text()
    lines = original.splitlines()

    changed = []
    for key, val in (("a_base", a_base), ("b_base", b_base)):
        r = _replace_scalar(lines, key, val)
        if r is None:
            lines.insert(0, f"{key}: {val}")
            changed.append(f"{key} added")
        elif r:
            changed.append(f"{key} -> {val}")

    start, end = _find_block(lines, "families")
    if start is None:
        lines += ["", *block_lines]
        changed.append(f"families added ({len(block_lines) - 1} entries)")
    else:
        old = sum(1 for ln in lines[start:end] if re.match(r"^\s*-\s", ln))
        lines[start:end] = block_lines
        new = sum(1 for ln in block_lines if re.match(r"^\s*-\s", ln))
        changed.append(f"families {old} -> {new} blocks")

    updated = "\n".join(lines) + "\n"

    # Parse before overwriting: a config that does not load is worse than no change, and
    # this is the one failure the user could not easily undo by hand.
    try:
        parsed = yaml.safe_load(updated)
    except yaml.YAMLError as e:
        raise SystemExit(f"[ERROR] the edit would produce invalid YAML, so nothing was "
                         f"written:\n{e}")
    if not isinstance(parsed, dict) or not parsed.get("families"):
        raise SystemExit("[ERROR] the edit lost the families block, so nothing was written")

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text(original)
        print(f"[*] backup       : {bak}", file=sys.stderr)
    path.write_text(updated)
    print(f"[*] wrote        : {path}  ({'; '.join(changed)})", file=sys.stderr)


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
                    help="the config.yaml to UPDATE: its families block and a_base/b_base "
                         "are rewritten from the spreadsheet, and its sqs.supercell is "
                         "used to check the compounds are realisable.  --stdout to print "
                         "instead of writing")
    ap.add_argument("--stdout", action="store_true",
                    help="print the families block instead of editing --config")
    ap.add_argument("--no-backup", action="store_true",
                    help="do not leave a config.yaml.bak beside the edited file")
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
        # The site count needs the base POSCAR, which a config may legitimately point at
        # before you have put it there.  That is a reason to skip the CHECK, not to refuse
        # to write the families -- so warn and carry on rather than dying with a traceback
        # and leaving the config untouched.
        try:
            from clc_workflow.clc_config import load_config
            from clc_workflow.sqs_generator import supercell_site_counts
            cfg = load_config(args.config)
            n_sites = supercell_site_counts(cfg["sqs"]["supercell"], cfg["base_poscar"])[0]
            print(f"[*] {n_sites} A sites from {args.config}", file=sys.stderr)
        except FileNotFoundError as e:
            print(f"[warn] cannot check the compounds are realisable: {e.filename} is "
                  f"missing.\n"
                  f"       Put the base POSCAR there, or pass --n-a-sites, and rerun to "
                  f"check.", file=sys.stderr)
        except Exception as e:
            print(f"[warn] cannot check the compounds are realisable ({type(e).__name__}: "
                  f"{e})", file=sys.stderr)
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

    block = ["families:"]
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
        block.append(f"  # {names}")
        block.append(f"  - {{a_dopant: {a_dop}, b_dopant: {b_dop}, "
                     f"x_values: [{fmt(x)}], y_values: [{', '.join(fmt(v) for v in ys)}]"
                     f"{extra}}}")

    if args.config and not args.stdout:
        write_config(args.config, block, a_base, b_base, backup=not args.no_backup)
        # The top-level x_values / y_values are the defaults for a family that omits
        # them.  Every generated block names both, so they no longer reach anything --
        # say so, because a stale [0.750] sitting at the top of the file reads like it is
        # still in force.
        top = yaml.safe_load(Path(args.config).read_text())
        if "x_values" in top or "y_values" in top:
            print("[*] note         : the top-level x_values / y_values are now unused -- "
                  "every generated block sets its own", file=sys.stderr)
        print(f"[*] next         : clc sqs {args.config} --dry-run", file=sys.stderr)
    else:
        print("\n".join(block))
    return 0


if __name__ == "__main__":
    sys.exit(main())
