import subprocess
import os
import numpy as np
from pymatgen.io.vasp import Poscar
from ionic_data import MAGMOM_VALUES, LDAUL_VALUES, LDAUU_VALUES
import re


def generate_potcar_with_vaspkit(poscar_dir):
    """
    Runs VASPkit to generate POTCAR from POSCAR in the specified directory.
    Requires VASPkit to be installed and in $PATH.
    """
    cwd = os.getcwd()
    os.chdir(poscar_dir)

    try:
        process = subprocess.run(
            ["vaspkit"],
            input="01\n103\n",
            text=True,
            capture_output=True
        )
        if process.returncode == 0:
            print(f"[INFO] POTCAR generated successfully in {poscar_dir}")
        else:
            print(f"[ERROR] VASPkit failed in {poscar_dir}: {process.stderr}")
    except Exception as e:
        print(f"[EXCEPTION] Failed to run VASPkit in {poscar_dir}: {e}")
    finally:
        os.chdir(cwd)

def generate_kpoints(
    poscar_path,
    output_path="KPOINTS",
    kspacing=None,
    kmesh=None,
    scheme="Gamma",
    debug=False
):
    """
    Generate VASP KPOINTS from either kspacing or explicit kmesh.
    
    Parameters:
    - poscar_path (str): Path to POSCAR file
    - output_path (str): Output KPOINTS file path
    - kspacing (float): Target spacing in Å⁻¹ (ignored if kmesh is provided)
    - kmesh (list or tuple): Explicit k-point grid, e.g., [6, 6, 6]
    - scheme (str): "Gamma" or "Monkhorst"
    - debug (bool): If True, print debug info
    """
    if kmesh is None and kspacing is None:
        raise ValueError("Must provide either kspacing or kmesh")

    if kmesh is not None:
        if debug:
            print(f"[DEBUG] Using user-defined kmesh: {kmesh}")
        mesh = kmesh
    else:
        poscar = Poscar.from_file(poscar_path)
        rec_lattice = poscar.structure.lattice.reciprocal_lattice
        lengths = [np.linalg.norm(vec) for vec in rec_lattice.matrix]

        if debug:
            print(f"[DEBUG] Reciprocal lattice lengths: {lengths}")
        
        mesh = [max(1, int(round(L / kspacing))) for L in lengths]

        if debug:
            print(f"[DEBUG] Computed mesh from kspacing={kspacing}: {mesh}")

    # Write KPOINTS
    with open(output_path, "w") as f:
        f.write("Automatic mesh\n")
        f.write("0\n")
        f.write(f"{scheme}\n")
        f.write(f"{mesh[0]} {mesh[1]} {mesh[2]}\n")

    print(f"[INFO] KPOINTS written to {output_path} with mesh: {mesh}")

def generate_incar_from_template(
    poscar_path,
    incar_template_path,
    output_path="INCAR",
    spin=True,
    dftu=True
):
    poscar = Poscar.from_file(poscar_path)
    elements = poscar.site_symbols
    element_counts = poscar.natoms

    # --- Prepare spin (MAGMOM)
    magmom_vals = [MAGMOM_VALUES.get(el, 0) for el in elements]
    magmom_block = "MAGMOM = " + " ".join(f"{n}*{m:.1f}" for n, m in zip(element_counts, magmom_vals))
    magmom_comment = " # " + " ".join(f"{el}={mag}μB" for el, mag in zip(elements, magmom_vals))

    # --- Prepare DFT+U
    ldaul_vals = [LDAUL_VALUES.get(el, -1) for el in elements]
    ldauu_vals = [LDAUU_VALUES.get(el, 0.0) for el in elements]
    ldauj_vals = ["0"] * len(elements)

    ldaul_comment = " # " + " ".join(f"{el}={v}" for el, v in zip(elements, ldaul_vals))
    ldauu_comment = " # " + " ".join(f"{el}={v}" for el, v in zip(elements, ldauu_vals))
    ldauj_comment = " # all 0"

    # --- Read Template
    with open(incar_template_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        stripped = line.strip()
        key = stripped.split("=")[0].strip() if "=" in stripped else ""

        # Handle spin-related MAGMOM
        if key == "MAGMOM":
            if spin:
                new_lines.append(magmom_block + magmom_comment)
            continue

        # Handle DFT+U related keys
        if key == "LDAUL":
            if dftu:
                new_lines.append("LDAUL = " + " ".join(map(str, ldaul_vals)) + ldaul_comment)
            continue
        elif key == "LDAUU":
            if dftu:
                new_lines.append("LDAUU = " + " ".join(map(str, ldauu_vals)) + ldauu_comment)
            continue
        elif key == "LDAUJ":
            if dftu:
                new_lines.append("LDAUJ = " + " ".join(ldauj_vals) + ldauj_comment)
            continue
        elif key == "LDAUTYPE":
            if dftu:
                new_lines.append("LDAUTYPE = 2")
            continue
        elif key == "LDAU":
            if dftu:
                new_lines.append("LDAU = .TRUE.")
            else:
                new_lines.append("LDAU = .FALSE.")
            continue

        # All other lines
        new_lines.append(line.rstrip())

    # --- Write final INCAR
    with open(output_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"[INFO] INCAR updated in-place with spin={spin}, dftu={dftu} → {output_path}")
