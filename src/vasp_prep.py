import subprocess
import os
import numpy as np
from pymatgen.io.vasp import Poscar
from ionic_data import MAGMOM_VALUES, LDAUL_VALUES, LDAUU_VALUES
import re
import shutil
import matplotlib.pyplot as plt
import glob
import pandas as pd


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
    output_path,
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
        poscar = Poscar.from_file(os.path.join(poscar_path,'POSCAR'))
        rec_lattice = poscar.structure.lattice.reciprocal_lattice
        lengths = [np.linalg.norm(vec) for vec in rec_lattice.matrix]

        if debug:
            print(f"[DEBUG] Reciprocal lattice lengths: {lengths}")
        
        mesh = [max(1, int(round(L / kspacing))) for L in lengths]

        if debug:
            print(f"[DEBUG] Computed mesh from kspacing={kspacing}: {mesh}")

    # Write KPOINTS
    with open(os.path.join(output_path,'KPOINTS'), "w") as f:
        f.write("Automatic mesh\n")
        f.write("0\n")
        f.write(f"{scheme}\n")
        f.write(f"{mesh[0]} {mesh[1]} {mesh[2]}\n")

    print(f"[INFO] KPOINTS written to {output_path} with mesh: {mesh}")
    return mesh

def generate_incar_from_template(
    poscar_path,
    incar_template_path,
    incar_template,
    output_path,
    spin=True,
    dftu=True
):
    poscar = Poscar.from_file(os.path.join(poscar_path,'POSCAR'))
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
    with open(os.path.join(incar_template_path,incar_template), "r") as f:
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
    with open(os.path.join(output_path,'INCAR'), "w") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"[INFO] INCAR updated in-place with spin={spin}, dftu={dftu} → {output_path}")


def run_kpoint_convergence_test(
    root_path,
    objective_path,
    kmesh_list=None,
    kspacing_list=None,
    scheme="Gamma",
    structure_source="POSCAR",  # New parameter
    incar_source="INCAR.static.template",
    prefix="static_kpoint",
    spin=True,
    dftu=True,
    debug=False
):
    if (kmesh_list is None and kspacing_list is None) or (kmesh_list and kspacing_list):
        raise ValueError("Provide either `kmesh_list` or `kspacing_list`, not both.")

    if structure_source not in ["POSCAR", "CONTCAR"]:
        raise ValueError("structure_source must be 'POSCAR' or 'CONTCAR'.")

    os.makedirs(objective_path, exist_ok=True)
    used_meshes = set()


    # ---- kmesh mode ----
    if kmesh_list:
        for mesh in kmesh_list:
            mesh_tuple = tuple(mesh)
            if mesh_tuple in used_meshes:
                print(f"[Warning] Duplicate kmesh {mesh_tuple} found. Skipping...")
                continue
            used_meshes.add(mesh_tuple)

            mesh_str = f"{mesh[0]}{mesh[1]}{mesh[2]}"
            subdir = os.path.join(objective_path, f"{prefix}{mesh_str}")
            if os.path.exists(subdir):
                print(f"[Warning] Directory {subdir} already exists. Skipping...")
                continue

            os.makedirs(subdir, exist_ok=True)
            #Copy INCAR from INCAR.template to destination path as INCAR
            generate_incar_from_template(os.path.join(root_path, structure_source), os.path.join(root_path, incar_source), output_path=os.path.join(subdir, "INCAR"), spin=spin, dftu=dftu)
            #Copy POTCAR to destination POTCAR
            shutil.copy2(os.path.join(root_path, 'POTCAR'), os.path.join(subdir, "POTCAR"))
            # Copy POSCAR or CONTCAR and rename to POSCAR
            shutil.copy2(os.path.join(root_path, structure_source), os.path.join(subdir, "POSCAR"))
            #copy_inputs(root_path, subdir, structure_source, incar_source)

            generate_kpoints(
                poscar_path=os.path.join(subdir, "POSCAR"),
                output_path=os.path.join(subdir, "KPOINTS"),
                kmesh=mesh,
                scheme=scheme,
                debug=debug
            )

            if debug:
                print(f"Generated kmesh {mesh_tuple} in {subdir}")

    # ---- kspacing mode (similar logic, omitted for brevity, reuse copy_inputs) ----
    # ...

    if debug:
        print("KPOINT convergence test setup completed.")


def plot_kpoint_convergence_results(base_dir, prefix="static_kpoint*", output_plot="kpoint_convergence.png", output_csv="kpoint_convergence.csv", debug=False):
    """
    Plot total energy vs. k-point mesh from OUTCARs in static_kpoint*/ or static_kspacing*/ subdirs.
    """
    data = []

    for path in sorted(glob.glob(os.path.join(base_dir, prefix))):
        outcar_path = os.path.join(path, "OUTCAR")
        if not os.path.isfile(outcar_path):
            if debug:
                print(f"[Warning] OUTCAR not found in {path}")
            continue

        # Extract mesh from directory name
        dirname = os.path.basename(path)
        if "kpoint" in dirname:
            mesh_str = dirname.replace("static_kpoint", "")
        elif "kspacing" in dirname:
            mesh_str = dirname.replace("static_kspacing", "")
        else:
            continue

        # Grep TOTEN value
        with open(outcar_path, "r") as f:
            lines = f.readlines()

        energy = None
        for line in reversed(lines):  # Grep from end (faster)
            if "free  energy   TOTEN" in line:
                match = re.search(r"TOTEN\s+=\s+(-?\d+\.\d+)", line)
                if match:
                    energy = float(match.group(1))
                    break

        if energy is None:
            print(f"[Warning] No TOTEN found in {outcar_path}")
            continue

        data.append((mesh_str, energy))

    if not data:
        print("[Error] No data found.")
        return

    # Sort by mesh size (e.g., "222" -> [2,2,2] -> product = 8)
    def mesh_product(mesh_str):
        try:
            return int(mesh_str[0]) * int(mesh_str[1]) * int(mesh_str[2])
        except:
            return float("inf")

    data.sort(key=lambda x: mesh_product(x[0]))

    meshes, energies = zip(*data)

    plt.figure(figsize=(8, 6))
    plt.plot(meshes, energies, marker="o", linestyle="-")
    plt.xlabel("K-point mesh")
    plt.ylabel("Total energy (eV)")
    plt.title("K-point Convergence Test")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, output_plot))
    plt.close()

    df = pd.DataFrame(data, columns=["kmesh", "TOTEN (eV)"])
    df.to_csv(os.path.join(base_dir, output_csv), index=False)

    if debug:
        print(f"[INFO] Plot saved to {output_plot}")
        print(f"[INFO] CSV saved to {output_csv}")
