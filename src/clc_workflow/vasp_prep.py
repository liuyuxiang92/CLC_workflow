import subprocess
import os
import numpy as np
from pymatgen.io.vasp import Poscar
from clc_workflow.ionic_data import MAGMOM_VALUES, LDAUL_VALUES, LDAUU_VALUES
import re
import shutil
import matplotlib.pyplot as plt
import glob
import pandas as pd
import dpdata


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
            generate_incar_from_template(root_path, root_path, incar_source, subdir, spin=spin, dftu=dftu)
            #Copy POTCAR to destination POTCAR
            shutil.copy2(os.path.join(root_path, 'POTCAR'), os.path.join(subdir, "POTCAR"))
            # Copy POSCAR or CONTCAR and rename to POSCAR
            shutil.copy2(os.path.join(root_path, structure_source), os.path.join(subdir, "POSCAR"))
            #copy_inputs(root_path, subdir, structure_source, incar_source)

            generate_kpoints(
                poscar_path=subdir,
                output_path=subdir,
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

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dpdata

def plot_convergence_results(
    base_dir, 
    prefix="kpoint*", 
    output_plot_prefix="convergence", 
    output_csv="convergence.csv", 
    debug=False
):
    """
    Plot convergence: delta energy, force, and virial norm vs. k-point mesh or ENCUT.
    Values are relative to the maximum kpoint/ENCUT result and in meV.
    """
    data = []

    for path in sorted(glob.glob(os.path.join(base_dir, prefix))):
        outcar_path = os.path.join(path, "OUTCAR")
        if not os.path.isfile(outcar_path):
            if debug:
                print(f"[Warning] OUTCAR not found in {path}")
            continue

        dirname = os.path.basename(path)
        if "kpoint" in dirname.lower():
            mode = "kpoint"
            mesh_str = dirname.lower().replace("kpoint", "")
            try:
                kx, ky, kz = int(mesh_str[0]), int(mesh_str[1]), int(mesh_str[2])
                x_value = (kx, ky, kz)
            except Exception as e:
                if debug:
                    print(f"[Warning] Cannot parse kpoint mesh from {dirname}: {e}")
                continue
        elif "energy" in dirname.lower():
            mode = "energy"
            mesh_str = dirname.lower().replace("energy", "")
            try:
                x_value = int(mesh_str)
            except Exception as e:
                if debug:
                    print(f"[Warning] Cannot parse ENCUT value from {dirname}: {e}")
                continue
        else:
            if debug:
                print(f"[Warning] Directory {dirname} does not match expected naming.")
            continue

        try:
            system = dpdata.LabeledSystem(outcar_path, fmt='outcar')
        except Exception as e:
            if debug:
                print(f"[Warning] Failed to read {outcar_path}: {e}")
            continue

        n_atoms = system.get_natoms()

        energy = system.data['energies'][-1] / n_atoms
        forces_flat = np.array(system.data['forces'][-1])
        virial_tensor = np.array(system.data['virials'][-1])

        force_norm = np.linalg.norm(forces_flat) / forces_flat.shape[0]/n_atoms
        virial_norm = np.linalg.norm(virial_tensor)/n_atoms

        if debug:
            print(f"Read: {dirname} -> Energy {energy:.6f} eV, Force {force_norm:.6f} eV/A, Virial {virial_norm:.6f} eV")

        data.append((x_value, energy, force_norm, virial_norm))

    if not data:
        print("[Error] No valid data found.")
        return

    # Sort data
    if mode == "kpoint":
        data.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))
    else:
        data.sort(key=lambda x: x[0])

    x_values, energies, force_norms, virial_norms = zip(*data)

    # Reference (maximum x) value for delta
    ref_energy = energies[-1]
    ref_force = force_norms[-1]
    ref_virial = virial_norms[-1]

    delta_energies = [(e - ref_energy) * 1000 for e in energies]     # eV -> meV
    delta_forces = [(f - ref_force) * 1000 for f in force_norms]
    delta_virials = [(v - ref_virial) * 1000 for v in virial_norms]

    if mode == "kpoint":
        x_labels = [f"{kx}{ky}{kz}" for (kx, ky, kz) in x_values]
        x_ticks = list(range(len(x_labels)))
    else:
        x_labels = [str(x) for x in x_values]
        x_ticks = x_values

    # --- Plot delta energy ---
    plt.figure(figsize=(8, 6))
    plt.plot(x_ticks, delta_energies, marker="o", linestyle="-")
    plt.xticks(x_ticks, x_labels)
    plt.xlabel("K-point mesh (kxkykz)" if mode == "kpoint" else "ENCUT (eV)")
    plt.ylabel("Delta Energy per atom (meV)")
    plt.title("Convergence: Energy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"{output_plot_prefix}_energy.png"))
    plt.close()

    # --- Plot delta force norm ---
    plt.figure(figsize=(8, 6))
    plt.plot(x_ticks, delta_forces, marker="o", linestyle="-")
    plt.xticks(x_ticks, x_labels)
    plt.xlabel("K-point mesh (kxkykz)" if mode == "kpoint" else "ENCUT (eV)")
    plt.ylabel("Delta Force (meV/Å/atom)")
    plt.title("Convergence: Force")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"{output_plot_prefix}_force.png"))
    plt.close()

    # --- Plot delta virial norm ---
    plt.figure(figsize=(8, 6))
    plt.plot(x_ticks, delta_virials, marker="o", linestyle="-")
    plt.xticks(x_ticks, x_labels)
    plt.xlabel("K-point mesh (kxkykz)" if mode == "kpoint" else "ENCUT (eV)")
    plt.ylabel("Delta Virial (meV/atom)")
    plt.title("Convergence: Virial")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"{output_plot_prefix}_virial.png"))
    plt.close()

    # --- Save CSV ---
    df = pd.DataFrame({
        "kmesh_or_energy": x_labels,
        "Delta TOTEN (meV)": delta_energies,
        "Delta Force Norm (meV/Å)": delta_forces,
        "Delta Virial Norm (meV)": delta_virials
    })
    df.to_csv(os.path.join(base_dir, output_csv), index=False)

    if debug:
        print(f"[INFO] Convergence plots and {output_csv} saved in {base_dir}.")

