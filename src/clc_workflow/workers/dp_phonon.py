#!/usr/bin/env python3
"""
Stage 7 worker -- DP energy + phonons for ONE optimised structure, in its own directory.

    python dp_phonon.py [--structure optimized_POSCAR] --model model.ckpt.pt
                        [--head DOWNSTREAM_DATA] [--supercell 1 1 1]
                        [--displacement 0.01] [--mesh 20 20 20]
                        [--temps 670:1070:5] [--delete-band-yaml]

Self-contained on purpose: dpdispatcher forwards only this file, so it must not depend on
anything in CLC_workflow/src.  submit.py bakes the parameters into the job command.

Physics is unchanged from the legacy predict_energy.py, including the stability verdict
(A = |min frequency|, B = 0.03*max frequency; imaginary if A > B) and the skip-if-exists
behaviour for each artifact.  What changed: the supercell matrix, displacement, mesh,
temperatures, model path and head are arguments instead of edited-in constants -- the old
code had predict_energy.py at 1x1x1 and get_stability_file.py at 2x2x2 in otherwise
identical files.  band.yaml is deleted after stability.txt by default (it holds ~960
bands per q-point for a 320-atom cell), which is what the commented-out cleanup at
predict_energy.py:170 intended.

Cost warning: a 320-atom SQS cell has no symmetry, so generate_displacements yields of
order 3N-6N displaced supercells, each needing a DP force call.  Time ONE structure
before submitting in bulk.
"""
import argparse
import os
import re
import sys
import time


def analyze_band_yaml(path="band.yaml"):
    """(("A B"), verdict) from a phonopy band.yaml -- verbatim legacy logic."""
    try:
        with open(path, "r") as f:
            content = f.read()
        freqs = re.findall(r"frequency:\s*(-?\d+\.\d+)", content)
        if not freqs:
            return "0.0000 0.0000", "error"
        vals = [float(x) for x in freqs]
        A, B = abs(min(vals)), max(vals) * 0.03
        out = f"{A:.4f} {B:.4f}"
        if out == "0.0000 0.0000":
            return out, "error"
        return out, ("imaginary frequency" if A > B else "stable")
    except Exception as e:
        print(f"band.yaml reading error: {e}")
        return "0.0000 0.0000", "error"


def parse_temps(spec):
    """'670:1070:5' -> linspace; '670,870' -> explicit list."""
    import numpy as np
    s = str(spec)
    if ":" in s:
        start, stop, num = s.split(":")
        return list(np.linspace(float(start), float(stop), int(num)))
    return [float(v) for v in s.split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", default="optimized_POSCAR")
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default=None)
    ap.add_argument("--supercell", type=int, nargs=3, default=[1, 1, 1])
    ap.add_argument("--displacement", type=float, default=0.01)
    ap.add_argument("--mesh", type=int, nargs=3, default=[20, 20, 20])
    ap.add_argument("--temps", default="670:1070:5")
    ap.add_argument("--delete-band-yaml", action="store_true")
    ap.add_argument("--keep-band-yaml", dest="delete_band_yaml", action="store_false")
    ap.add_argument("--skip-phonon", action="store_true", help="energy.txt only")
    args = ap.parse_args()

    if not os.path.isfile(args.structure):
        sys.exit(f"[ERROR] {args.structure} not found in {os.getcwd()}")

    import numpy as np
    from ase import Atoms
    from ase.io import read
    from deepmd.calculator import DP
    import phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    calc = DP(model=args.model, **({"head": args.head} if args.head else {}))

    # ---------- 1) energy ----------
    if not os.path.exists("energy.txt"):
        atoms = read(args.structure)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        with open("energy.txt", "w") as f:
            f.write(f"{energy}\n")
        print(f"[energy] {energy}")
    else:
        print("[skip energy] energy.txt exists")

    if args.skip_phonon:
        return

    need_thermal = not os.path.exists("thermal_properties.yaml")
    need_band = not os.path.exists("band.yaml")
    need_stability = not os.path.exists("stability.txt")
    if not (need_thermal or need_band or need_stability):
        print("[skip phonon] all artifacts exist")
        return

    # ---------- 2) force constants ----------
    atoms = read(args.structure)
    ph_atoms = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                            positions=atoms.get_positions(),
                            cell=atoms.get_cell(),
                            masses=list(atoms.get_masses()))
    sc = [[args.supercell[0], 0, 0], [0, args.supercell[1], 0], [0, 0, args.supercell[2]]]
    phonon = phonopy.Phonopy(ph_atoms, supercell_matrix=sc)
    phonon.generate_displacements(distance=float(args.displacement), is_diagonal=False)

    n_disp = len(phonon.supercells_with_displacements)
    print(f"[phonon] {len(atoms)} atoms, supercell {args.supercell}, {n_disp} displacements")
    t0 = time.time()
    forcesets = []
    for i, config in enumerate(phonon.supercells_with_displacements, 1):
        ca = Atoms(cell=config.cell, symbols=config.symbols,
                   scaled_positions=config.scaled_positions, pbc=True)
        ca.calc = calc
        forcesets.append(ca.get_forces())
        if i % 100 == 0 or i == n_disp:
            rate = i / max(time.time() - t0, 1e-9)
            print(f"    forces {i}/{n_disp}  ({rate:.1f}/s, "
                  f"eta {int((n_disp - i) / max(rate, 1e-9))} s)")
    phonon.forces = forcesets
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    print(f"[phonon] force constants in {time.time() - t0:.1f} s")

    # ---------- 3) thermal properties ----------
    if need_thermal:
        phonon.run_mesh(list(args.mesh))
        phonon.run_thermal_properties(temperatures=parse_temps(args.temps))
        phonon.write_yaml_thermal_properties(filename="thermal_properties.yaml")
        print("[thermal] wrote thermal_properties.yaml")

    # ---------- 4) band structure ----------
    if need_band:
        phonon.auto_band_structure(plot=False)
        phonon.write_yaml_band_structure(filename="band.yaml")
        print("[band] wrote band.yaml")

    # ---------- 5) stability verdict ----------
    if need_stability and os.path.exists("band.yaml"):
        out, verdict = analyze_band_yaml("band.yaml")
        with open("stability.txt", "w") as f:
            f.write(f"{out}\n{verdict}\n")
        print(f"[stability] {out}  {verdict}")

    # ---------- 6) reclaim space ----------
    if args.delete_band_yaml and os.path.exists("stability.txt") \
            and os.path.exists("band.yaml"):
        os.unlink("band.yaml")
        print("[cleanup] deleted band.yaml")


if __name__ == "__main__":
    main()
