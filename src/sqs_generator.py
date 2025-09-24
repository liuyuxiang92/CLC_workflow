import os
import subprocess
import numpy as np
import shutil
import random
import secrets
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure,Element


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def format_ratio(val):
    return int(round(val * 1000))


def build_directory_name(a_base, a_dopant, x, b_base, b_dopant, y):
    a_base_tag = f"{a_base}{format_ratio(1 - x):03d}"
    a_dop_tag = f"{a_dopant}{format_ratio(x):03d}"
    b_base_tag = f"{b_base}{format_ratio(1 - y):03d}"
    b_dop_tag = f"{b_dopant}{format_ratio(y):03d}"
    group_dir = f"{a_base_tag}_{a_dop_tag}_{b_base}_{b_dopant}"
    comp_dir = f"{a_base_tag}_{a_dop_tag}_{b_base_tag}_{b_dop_tag}"
    return group_dir, comp_dir


def get_kept_vacancy_indices(structure, site_species, vacancy_frac):
    if 2 not in site_species:
        return set()
    third_site_elements = [el for el, _ in site_species[2]]
    third_site_indices = [i for i, site in enumerate(structure) if site.specie.symbol in third_site_elements]
    n_third = len(third_site_indices)
    n_keep = int(round((1 - vacancy_frac) * n_third))
    return set(third_site_indices[:n_keep])


def detect_third_site_element(structure, site_species):
    known_elements = set(el for group in site_species.values() for el, _ in group)
    third_elements = [site.specie.symbol for site in structure if site.specie.symbol not in known_elements]
    return list(set(third_elements))


def generate_rndstr_in(structure, output_path, site_species, vacancy_frac):
    lattice = structure.lattice
    lines = []

    lengths = np.array(lattice.lengths)
    angles = lattice.angles
    lines.append(f"{lengths[0]:.6f} {lengths[1]:.6f} {lengths[2]:.6f} {angles[0]:.2f} {angles[1]:.2f} {angles[2]:.2f}")

    unit_vectors = np.array(lattice.matrix) / lengths[:, np.newaxis]
    for vec in unit_vectors:
        lines.append(" ".join(f"{x:.6f}" for x in vec))

    kept_third_site_indices = get_kept_vacancy_indices(structure, site_species, vacancy_frac)

    for i, site in enumerate(structure):
        symbol = site.specie.symbol

        # Skip third-site vacancies
        #if 2 in site_species and symbol in [el for el, _ in site_species[2]] and i not in kept_third_site_indices:
        #    continue

        dopants = None
        for group_dopants in site_species.values():
            if symbol in [el for el, _ in group_dopants]:
                dopants = group_dopants
                break

        if dopants is None:
            raise ValueError(f"Unknown site species: {symbol} at index {i}")

        frac = structure.frac_coords[i]
        # Only include dopants with non-zero concentration
        dop_str = ", ".join(
            f"{el}={frac_val:.6f}"
            for el, frac_val in dopants
            if frac_val > 1e-6
        )
        lines.append(f"{frac[0]:.6f} {frac[1]:.6f} {frac[2]:.6f} {dop_str}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def run_corrdump(directory, rndstr_filename="rndstr.in", r2=5.58, r3=4.24, r4=0):
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        subprocess.run([
            "mcsqs",
            f"-l={rndstr_filename}",
            f"-2={r2}", f"-3={r3}", f"-4={r4}"
        ], stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] mcsqs (cluster gen mode) failed in {directory}: {e}")
        print(e.stderr.decode())
    finally:
        os.chdir(cwd)


def run_mcsqs(directory, n_atoms, timeout_sec):
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        print("check run_mcsqs")
        print(f"mcsqs -n={n_atoms}")
        if isinstance(n_atoms, (tuple, list)):
            n_atoms = n_atoms[0]
        else:
            n_atoms = int(n_atoms)
        print(f"mcsqs -n={n_atoms}")
        print(f"timeout_sec = {timeout_sec}")

        # simple, secure, built-in seed: 1..(2**31-1)
        random_seed = 1 + secrets.randbelow(2**31 - 1)

        subprocess.run(["mcsqs", f"-n={n_atoms}", f"-sd={random_seed}"], check=True, timeout=timeout_sec,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("after run_mcsqs")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] mcsqs timed out in {directory} after {timeout_sec} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] mcsqs structure generation failed in {directory}: {e}")
    finally:
        os.chdir(cwd)

def run_mcsqs_rc(directory, timeout_sec):
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        subprocess.run(["mcsqs", f"-rc"], check=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] mcsqs -rc timed out in {directory} after {timeout_sec} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] mcsqs -rc structure generation failed in {directory}: {e}")
    finally:
        os.chdir(cwd)


def convert_bestsqs_to_poscar(directory):
    bestsqs_path = os.path.join(directory, "bestsqs.out")
    if not os.path.exists(bestsqs_path):
        print(f"[WARNING] bestsqs.out not found in {directory}. POSCAR not generated.")
        return
    try:
        structure = Structure.from_file(bestsqs_path).get_sorted_structure()
        structure.remove_species("X")
        poscar = Poscar(structure)
        poscar.write_file(os.path.join(directory, "POSCAR"))
    except Exception as e:
        print(f"[ERROR] Failed to convert bestsqs.out to POSCAR in {directory}: {e}")

def clean_directory_keep_poscar(directory):
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if filename != "POSCAR":
            try:
                if os.path.isfile(full_path):
                    os.remove(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
            except Exception as e:
                print(f"[WARNING] Failed to remove {full_path}: {e}")




def replace_sqscell_with_cubic(sqscell_dir, tolerance=1e-3):
    """
    From the given sqscell.out, select cubic matrices (equal vector lengths and 90-degree angles),
    and rewrite the file to only include these matrices.

    Args:
        sqscell_dir (str): Directory containing sqscell.out
        tolerance (float): Allowed numerical tolerance (default 1e-3)
    """
    sqscell_path = os.path.join(sqscell_dir, 'sqscell.out')
    backup_path = os.path.join(sqscell_dir, 'sqscell_old.out')

    if not os.path.isfile(sqscell_path):
        raise FileNotFoundError(f"{sqscell_path} not found.")

    shutil.move(sqscell_path, backup_path)

    with open(backup_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty file: {backup_path}")

    n_matrices = int(lines[0].strip())

    matrices_raw, current_matrix = [], []
    for line in lines[1:]:
        if line.strip() == '':
            continue
        current_matrix.append(line.rstrip())
        if len(current_matrix) == 3:
            matrices_raw.append(current_matrix)
            current_matrix = []

    if len(matrices_raw) != n_matrices:
        raise ValueError(f"Expected {n_matrices} matrices, found {len(matrices_raw)}")

    matrices_float = []
    for mat_raw in matrices_raw:
        mat = []
        for row in mat_raw:
            mat.append([float(x) for x in row.split()])
        matrices_float.append(np.array(mat))

    selected_raw = []
    for raw, mat in zip(matrices_raw, matrices_float):
#        lengths = np.linalg.norm(mat, axis=1)
#        if np.all(np.abs(lengths - lengths[0]) < tolerance):
            lengths = np.linalg.norm(mat, axis=1)
            cos01 = np.dot(mat[0], mat[1]) / (lengths[0] * lengths[1])
            cos02 = np.dot(mat[0], mat[2]) / (lengths[0] * lengths[2])
            cos12 = np.dot(mat[1], mat[2]) / (lengths[1] * lengths[2])
            if all(abs(cos) < tolerance for cos in [cos01, cos02, cos12]):
                selected_raw.append(raw)

    with open(sqscell_path, 'w') as f:
        f.write(f"{len(selected_raw)}\n\n")
        for idx, mat_lines in enumerate(selected_raw):
            for line in mat_lines:
                f.write(line + '\n')
            if idx != len(selected_raw) - 1:
                f.write('\n')

    print(f"Selected {len(selected_raw)} cubic matrices out of {n_matrices}.")


def generate_random_substitution_poscar(
    unitcell_poscar_path,
    site_species,
    output_path,
    seed,
):
    random.seed(seed)
    structure = Poscar.from_file(unitcell_poscar_path).structure.copy()

    # --- Perform random substitution ---
    for site_type, species_frac in site_species.items():
        base_elem = species_frac[0][0]
        indices = [i for i, site in enumerate(structure) if site.specie.symbol == base_elem]

        for dopant, frac in species_frac:
            num_replace = int(round(len(indices) * frac))
            if dopant == base_elem:
                continue
            replace_indices = random.sample(indices, num_replace)
            if dopant.upper().startswith("X"):  # vacancy: remove atoms
                for idx in sorted(replace_indices, reverse=True):
                    del structure[idx]
            else:
                for idx in replace_indices:
                    structure.replace(idx, dopant, coords=structure[idx].frac_coords)
            indices = [i for i in indices if i not in replace_indices]

    # --- Sort species alphabetically ---
    unique_species = sorted(set(site.specie.symbol for site in structure))
    ordered_sites = []
    for symbol in unique_species:
        ordered_sites.extend([site for site in structure if site.specie.symbol == symbol])

    # --- Create sorted structure and write POSCAR ---
    sorted_structure = Structure(
        structure.lattice,
        [site.species for site in ordered_sites],
        [site.frac_coords for site in ordered_sites]
    )

    poscar = Poscar(sorted_structure)
    poscar.write_file(os.path.join(output_path, "POSCAR"))


def generate_all_poscars(
    screened_df,
    unitcell_poscar_path,
    delta_list,
    n_atoms,
    timeout_sec,
    select_cubic=False,
    clean_dir=True,
    output_root="sqs_structures",
    use_random_substitution=False,
    seed=123
):
    structure = Poscar.from_file(unitcell_poscar_path).structure

    for row_idx, row in screened_df.iterrows():
        x, y = float(row['x']), float(row['y'])
        a_base, a_dop = row['A_base'], row['A_dopant']
        b_base, b_dop = row['B_base'], row['B_dopant']

        group_dir, comp_dir = build_directory_name(a_base, a_dop, x, b_base, b_dop, y)
        full_comp_path = os.path.join(output_root, group_dir, comp_dir)

        site_species_ab = {
            0: [(a_base, 1 - x), (a_dop, x)],
            1: [(b_base, 1 - y), (b_dop, y)]
        }

        third_site_elements = detect_third_site_element(structure, site_species_ab)
        if not third_site_elements:
            raise ValueError("Unable to infer third site element for vacancy control.")
        third_element = third_site_elements[0]

        for delta_idx, delta in enumerate(delta_list):
            o_tag = f"o{format_ratio(1 - delta):03d}"
            full_path = os.path.join(full_comp_path, o_tag)
            ensure_dir(full_path)

            site_species = site_species_ab.copy()
            vacancy_frac = delta / 3  # default fallback
            if vacancy_frac > 0:
                site_species[2] = [(third_element, 1 - vacancy_frac), ("X", vacancy_frac)]
            else:
                site_species[2] = [(third_element, 1.0)]

            if use_random_substitution:
                # Deterministic sub-seed from global seed, row index, and delta index
                sub_seed = hash((seed, row_idx, delta_idx)) % (2**32)
                generate_random_substitution_poscar(
                    unitcell_poscar_path,
                    site_species,
                    full_path,
                    sub_seed
                )
            else:
                rndstr_path = os.path.join(full_path, "rndstr.in")
                print(f"site_species = {site_species}")
                generate_rndstr_in(structure, rndstr_path, site_species, vacancy_frac)
                run_corrdump(full_path)
                run_mcsqs(full_path, n_atoms, timeout_sec)
                if select_cubic:
                    replace_sqscell_with_cubic(full_path)
                    print("what happen")
                    run_mcsqs_rc(full_path, timeout_sec)
                convert_bestsqs_to_poscar(full_path)
                if clean_dir:
                    clean_directory_keep_poscar(full_path)

            print(f"[INFO] Finished: {full_path}/POSCAR")



#generate rndstr.in for oxygen vacancy
def generate_rndstr_in_oxygen_vacancy(poscar_path: str, output_path: str = "rndstr.in", oxygen_fraction: float = 0.9):
    """
    Generate a rndstr.in file for MCSQS with fixed A/B cations and mixed O/Va sites.

    Args:
        poscar_path (str): Path to the POSCAR file.
        output_path (str): Output path for rndstr.in.
        oxygen_fraction (float): Fraction of O atoms to keep (e.g., 0.9 for 10% vacancy).
    """
    structure = Structure.from_file(poscar_path)
    lattice = structure.lattice
    lines = []

    lengths = np.array(lattice.lengths)
    angles = lattice.angles
    lines.append(f"{lengths[0]:.6f} {lengths[1]:.6f} {lengths[2]:.6f} {angles[0]:.2f} {angles[1]:.2f} {angles[2]:.2f}")

    unit_vectors = np.array(lattice.matrix) / lengths[:, np.newaxis]
    for vec in unit_vectors:
        lines.append(" ".join(f"{x:.6f}" for x in vec))

    for i, site in enumerate(structure):
        el = site.specie.symbol
        frac = site.frac_coords

        if el == "O":
            dop_str = f"O={oxygen_fraction:.6f}, Va={1.0 - oxygen_fraction:.6f}"
        else:
            dop_str = f"{el}=1.000000"

        lines.append(f"{frac[0]:.6f} {frac[1]:.6f} {frac[2]:.6f} {dop_str}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Successfully wrote {output_path} with O/Va mixing (O fraction = {oxygen_fraction})")

