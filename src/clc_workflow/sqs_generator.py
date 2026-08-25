import os
import subprocess
import numpy as np
import shutil
import random
import secrets
from functools import lru_cache
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure,Element


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def format_ratio(val):
    return int(round(val * 1000))


def build_directory_name(a_base, a_dopant, x, b_base, b_dopant, y,
                         a_vac=0.0, b_vac=0.0):
    """
    Group/composition directory names, e.g.
        ("Sr875_Ca125_Fe_Mn", "Sr875_Ca125_Fe875_Mn125")

    A None dopant is rendered as the literal string "None" for backward
    compatibility with existing trees; pass the family's dopant symbol instead
    (with a 000 tag) to keep families in separate directories -- see
    build_directory_name_for_family().

    x and y are fractions of the A and B SITES, so the base tag is 1 - x - a_vac: a
    cation-deficient cell reads directly off the name (Sr750_Ca200 is 5% A-vacant, since
    the two do not sum to 1000).  A non-zero vacancy also appends an explicit _vA###
    / _vB### tag, on the A side to the group as well because it is part of the A-site
    composition.  Both tags are omitted at zero, so vacancy-free trees keep the names
    they have.
    """
    a_base_tag = f"{a_base}{format_ratio(1 - x - a_vac):03d}"
    a_dop_tag = f"{a_dopant}{format_ratio(x):03d}"
    b_base_tag = f"{b_base}{format_ratio(1 - y - b_vac):03d}"
    b_dop_tag = f"{b_dopant}{format_ratio(y):03d}"
    a_vac_tag = f"_vA{format_ratio(a_vac):03d}" if a_vac else ""
    b_vac_tag = f"_vB{format_ratio(b_vac):03d}" if b_vac else ""
    group_dir = f"{a_base_tag}_{a_dop_tag}{a_vac_tag}_{b_base}_{b_dopant}"
    comp_dir = f"{a_base_tag}_{a_dop_tag}{a_vac_tag}_{b_base_tag}_{b_dop_tag}{b_vac_tag}"
    return group_dir, comp_dir


def build_directory_name_for_family(a_base, a_dopant, x, b_base, b_dopant, y,
                                    a_vac=0.0, b_vac=0.0):
    """
    Same layout, but the dopant symbol is always used even at zero concentration.

    With several families in one tree, the legacy behaviour renders A_dopant=None as
    the literal "None000", so every family's x=0 column would collide in a single
    Sr1000_None000_* directory.  Using the configured dopant name keeps them apart:
        x=0, Ca/Mn -> Sr1000_Ca000_Fe_Mn
        x=0, Ba/Co -> Sr1000_Ba000_Fe_Co
    """
    if not a_dopant or not b_dopant:
        raise ValueError("family directory names require both dopant symbols")
    return build_directory_name(a_base, a_dopant, x, b_base, b_dopant, y,
                                a_vac=a_vac, b_vac=b_vac)


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
    """Generate clusters.out.  Uses cwd= so it is safe inside a process pool."""
    try:
        subprocess.run([
            "mcsqs",
            f"-l={rndstr_filename}",
            f"-2={r2}", f"-3={r3}", f"-4={r4}"
        ], cwd=str(directory), stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        print("[ERROR] mcsqs not found on PATH")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] mcsqs (cluster gen mode) failed in {directory}: {e}")
        if e.stderr:
            print(e.stderr.decode(errors="replace"))
        return False


def run_mcsqs(directory, n_atoms, timeout_sec):
    """
    Search over supercells of n_atoms sites.  Uses cwd= so it is pool-safe.

    A timeout is not a failure: mcsqs rewrites bestsqs.out whenever it improves, so
    the best structure found so far is still on disk.
    """
    if isinstance(n_atoms, (tuple, list)):
        n_atoms = n_atoms[0]
    n_atoms = int(n_atoms)
    random_seed = 1 + secrets.randbelow(2**31 - 1)
    try:
        subprocess.run(["mcsqs", f"-n={n_atoms}", f"-sd={random_seed}"],
                       cwd=str(directory), check=True, timeout=timeout_sec,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        print("[ERROR] mcsqs not found on PATH")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] mcsqs structure generation failed in {directory}: {e}")

def run_mcsqs_rc(directory, timeout_sec):
    """
    Optimise within the supercell(s) listed in sqscell.out (mcsqs -rc).  Pool-safe.

    As with run_mcsqs, a timeout leaves the best-so-far bestsqs.out in place.
    """
    random_seed = 1 + secrets.randbelow(2**31 - 1)
    try:
        subprocess.run(["mcsqs", "-rc", f"-sd={random_seed}"],
                       cwd=str(directory), check=True, timeout=timeout_sec,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        print("[ERROR] mcsqs not found on PATH")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] mcsqs -rc structure generation failed in {directory}: {e}")


def poscar_atom_count(path):
    """Atoms declared by a POSCAR's counts line, or None if it cannot be read."""
    try:
        with open(path) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        for ln in lines[5:7]:                  # species names then counts (or just counts)
            toks = ln.split()
            if toks and all(t.lstrip("+-").isdigit() for t in toks):
                return sum(int(t) for t in toks)
    except OSError:
        pass
    return None


def convert_bestsqs_to_poscar(directory, expect_atoms=None):
    """
    bestsqs.out -> POSCAR.  Returns (ok, detail).

    `expect_atoms` is not optional in practice, and the reason is the main failure mode of
    this whole stage: mcsqs rewrites bestsqs.out every time it improves, and a timeout
    KILLS it -- so a run that times out mid-write leaves a truncated file.  Truncation in
    the 6-line header raises here.  Truncation anywhere in the atom block, which is 98% of
    the file, does NOT: pymatgen parses what is there and hands back a structure with
    fewer atoms, which then becomes a perfectly valid-looking POSCAR of the wrong
    composition.  Counting the sites before writing is the only thing standing between
    that and 200 ps of MD on a cell that is quietly missing atoms.
    """
    bestsqs_path = os.path.join(directory, "bestsqs.out")
    if not os.path.exists(bestsqs_path):
        return False, "bestsqs.out not found"
    try:
        structure = Structure.from_file(bestsqs_path).get_sorted_structure()
    except Exception as e:
        # IndexError here is the header-truncation case, not a bug in the parser
        return False, (f"bestsqs.out unreadable ({type(e).__name__}: {e}) -- truncated, "
                       f"most likely killed mid-write by the mcsqs timeout")
    n_sites = len(structure)
    structure.remove_species("X")
    if expect_atoms is not None and len(structure) != int(expect_atoms):
        return False, (f"bestsqs.out holds {n_sites} sites -> {len(structure)} atoms, "
                       f"expected {int(expect_atoms)} -- truncated, most likely killed "
                       f"mid-write by the mcsqs timeout")
    try:
        Poscar(structure).write_file(os.path.join(directory, "POSCAR"))
    except Exception as e:
        return False, f"POSCAR write failed ({type(e).__name__}: {e})"
    return True, ""

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



# =====================================================================================
#  nv-based, pool-safe single-structure generation (used by `clc sqs`)
# =====================================================================================
#
#  The legacy path wrote delta = nv/64 in one file and converted it back with
#  vacancy_frac = delta/3 in another, which only worked because the supercell happened
#  to have 192 oxygen sites.  Here the vacancy COUNT is primary and both fractions are
#  derived from it exactly:
#
#      n_O_sites    = 24 * prod(supercell)     # 192 for [2,2,2]
#      vacancy_frac = nv / n_O_sites           # what mcsqs is told
#      delta        = 3 * nv / n_O_sites       # = nv/n_fu; used for the o-tag
#
@lru_cache(maxsize=8)
def base_cell_counts(base_poscar=None):
    """
    (A, B, O) sites in ONE base cell.

    None keeps the historical answer for the 40-site 2x2x2 base, (8, 8, 24), so nothing
    that already calls supercell_site_counts(supercell) changes.  Given a path, the counts
    are read from the file instead -- which is what makes a 5-atom primitive base usable.
    That matters because the supercell shape is not free: reaching a given site count from
    a 40-site base costs a very elongated cell (400 atoms = [5,2,1], 38.6 x 15.4 x 7.7 A),
    while the same 400 atoms from a 5-atom base is [5,4,4], 19.3 x 15.4 x 15.4 A.

    The A/B assignment is positional -- first non-O species is A, second is B -- which is
    the order every POSCAR in this pipeline is written in.  Cached because
    generate_one_sqs asks per task.
    """
    if base_poscar is None:
        return 8, 8, 24
    struct = Poscar.from_file(str(base_poscar)).structure
    order, counts = [], {}
    for site in struct:
        s = site.specie.symbol
        if s not in counts:
            order.append(s)
            counts[s] = 0
        counts[s] += 1
    cations = [s for s in order if s != "O"]
    if len(cations) != 2 or "O" not in counts:
        raise ValueError(f"{base_poscar}: expected an ABO3 base cell with two cation "
                         f"species and O, found {counts}")
    return counts[cations[0]], counts[cations[1]], counts["O"]


def supercell_site_counts(supercell, base_poscar=None):
    """(n_A, n_B, n_O, n_atoms, n_formula_units) for `supercell` of the base cell."""
    mult = int(supercell[0]) * int(supercell[1]) * int(supercell[2])
    a_cell, b_cell, o_cell = base_cell_counts(base_poscar)
    n_A = a_cell * mult
    n_B = b_cell * mult
    n_O = o_cell * mult
    return n_A, n_B, n_O, n_A + n_B + n_O, n_A


def site_occupancies(x, y, a_vac=0.0, b_vac=0.0):
    """
    Sublattice occupancies as ((A_base, A_dop, A_vac), (B_base, B_dop, B_vac)).

    x and y are fractions of the A and B SITES, so the base species gets 1 - x - a_vac.
    That convention reads straight off a formula: Sr0.75Ca0.2Fe0.875Co0.125O3 is
    x = 0.2, a_vac = 0.05, and it degenerates to the old 1 - x when there is no vacancy.
    """
    for name, dop, vac in (("A", x, a_vac), ("B", y, b_vac)):
        if dop < -1e-9 or vac < -1e-9:
            raise ValueError(f"{name}-site: negative dopant/vacancy fraction "
                             f"({dop}, {vac})")
        if dop + vac > 1.0 + 1e-9:
            raise ValueError(f"{name}-site: dopant + vacancy = {dop + vac:.4f} > 1; "
                             f"x/y are fractions of the SITES, so the base species gets "
                             f"1 - dopant - vacancy")
    return (1.0 - x - a_vac, x, a_vac), (1.0 - y - b_vac, y, b_vac)


def integral_site_counts(fractions, n_sites, tol=1e-6):
    """
    fractions -> integer site counts, or None if any of them is not realisable.

    mcsqs is handed concentrations, not counts: ask for 0.05 on 64 sites and ATAT will
    quietly build the nearest thing it can (3.2 -> 3 sites = 0.047), so the cell that
    comes back is a different compound from the one requested, with no error anywhere.
    Checking here turns that into a failure at task-build time, before any GPU hour.
    """
    counts = []
    for f in fractions:
        v = f * n_sites
        if abs(v - round(v)) > tol:
            return None
        counts.append(int(round(v)))
    return counts


def vacancy_fraction(nv, n_O_sites):
    """Vacancy concentration on the oxygen sublattice (what rndstr.in declares)."""
    if nv < 0 or nv > n_O_sites:
        raise ValueError(f"nv={nv} outside 0..{n_O_sites}")
    return float(nv) / float(n_O_sites)


def delta_from_nv(nv, n_O_sites):
    """Oxygen deficiency per formula unit, delta = 3*nv/n_O_sites."""
    return 3.0 * float(nv) / float(n_O_sites)


def write_sqscell(directory, supercell):
    """
    Write sqscell.out containing the single diagonal supercell matrix, so that
    `mcsqs -rc` optimises inside a fixed cell instead of searching over shapes.

    For a cubic base cell this guarantees an axis-aligned cube every time, which keeps
    the NPT-aniso a/b/c readings comparable across the whole sweep.
    """
    a, b, c = (int(v) for v in supercell)
    lines = ["1", "", f"{a} 0 0", f"0 {b} 0", f"0 0 {c}"]
    path = os.path.join(str(directory), "sqscell.out")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _is_fully_ordered(site_species):
    """True when every sublattice has a single species at unit occupancy."""
    for group in site_species.values():
        occupied = [(el, frac) for el, frac in group if frac > 1e-6]
        if len(occupied) != 1 or abs(occupied[0][1] - 1.0) > 1e-9:
            return False
    return True


def write_ordered_supercell(base_structure, supercell, site_species, out_dir):
    """
    Zero-disorder short-circuit.

    At nv=0 with x,y in {0,1} every concentration is 1.0, so mcsqs has no clusters to
    optimise and exits with an error.  Write the plain supercell instead, substituting
    each sublattice's single species.
    """
    struct = base_structure.copy()
    sub_map = {}
    for group in site_species.values():
        occupied = [(el, frac) for el, frac in group if frac > 1e-6]
        el = occupied[0][0]
        for existing, _ in group:
            sub_map[existing] = el
    for i, site in enumerate(struct):
        target = sub_map.get(site.specie.symbol)
        if target and target != site.specie.symbol:
            struct.replace(i, target, coords=site.frac_coords)
    struct.make_supercell([int(v) for v in supercell])
    Poscar(struct.get_sorted_structure()).write_file(os.path.join(str(out_dir), "POSCAR"))
    return True


def _task_id(task):
    return {"out_dir": str(task["out_dir"]), "x": task["x"], "y": task["y"],
            "nv": task["nv"]}


def _finish(out_dir, task):
    if task.get("clean_dir", True):
        clean_directory_keep_poscar(out_dir)


# ------------------------------------------------------------------ random occupation
def composition_seed(seed_key, base_seed=0):
    """
    A stable integer seed for one (set, composition), derived from its name.

    hashlib rather than hash(): Python salts str hashing per process, so hash() would
    give a different structure in every worker and every rerun.  This has to be
    reproducible across processes, machines and reruns, because it is the *only* thing
    tying the o0000, o0001, ... directories of one composition together.
    """
    import hashlib
    h = hashlib.sha256(f"{base_seed}|{seed_key}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _sublattice_indices(struct, a_base, b_base):
    """(A, B, anion) site indices of a supercell, keyed off the base cell's symbols."""
    a_idx, b_idx, o_idx = [], [], []
    for i, site in enumerate(struct):
        sym = site.specie.symbol
        (a_idx if sym == a_base else b_idx if sym == b_base else o_idx).append(i)
    return a_idx, b_idx, o_idx


def generate_one_random(task):
    """
    Build one structure by random occupation instead of by mcsqs.

    Cations are shuffled onto their sublattice and oxygens are removed at random, but
    **the randomness is seeded from the composition alone, never from nv**, and that is
    the whole point:

        rng = seed(set, group, comp)          # nv is NOT in the key
        cation arrangement = rng.permutation(A sites), rng.permutation(B sites)
        removal order      = rng.permutation(O sites)
        this directory     = remove the FIRST nv oxygens of that order

    So every o* directory of one composition shares one cation arrangement, and its
    vacancies are NESTED: o0002 is o0001 with one more oxygen gone, o0003 is o0002 with
    one more, all the way to the last.  The delta-G of one vacancy is then a difference
    along a single continuous removal path rather than between two unrelated cells,
    which is what makes the pair deltas comparable at all -- with independently drawn
    structures the arrangement noise (several eV, as the G_spread column shows) sits on
    top of a signal of order the vacancy formation energy itself.

    Nothing here is written per nv, so the nesting is a property of the construction and
    cannot drift: two directories built months apart, on different machines, in
    different worker processes, still lie on the same path.

    Extra task keys beyond generate_one_sqs's: seed_key (required), random_seed.
    Returns status: skipped | stale | random | failed
    """
    out_dir = str(task["out_dir"])
    poscar_path = os.path.join(out_dir, "POSCAR")
    supercell = [int(v) for v in task["supercell"]]
    n_A, n_B, n_O_sites, _n_sites, _ = supercell_site_counts(supercell,
                                                             task["base_poscar"])
    x, y, nv = float(task["x"]), float(task["y"]), int(task["nv"])
    a_vac = float(task.get("a_vac") or 0.0)
    b_vac = float(task.get("b_vac") or 0.0)

    a_fracs, b_fracs = site_occupancies(x, y, a_vac, b_vac)
    a_counts = integral_site_counts(a_fracs, n_A)
    b_counts = integral_site_counts(b_fracs, n_B)
    if a_counts is None or b_counts is None:
        side = "A" if a_counts is None else "B"
        return {**_task_id(task), "status": "failed",
                "detail": f"{side} site composition does not land on whole sites"}

    expect_atoms = (a_counts[0] + a_counts[1] + b_counts[0] + b_counts[1]
                    + (n_O_sites - nv))

    if os.path.isfile(poscar_path):
        have = poscar_atom_count(poscar_path)
        if have == expect_atoms:
            if task.get("audit") or task.get("skip_existing", True):
                return {**_task_id(task), "status": "skipped", "detail": "POSCAR exists"}
        else:
            detail = f"POSCAR has {have} atoms, expected {expect_atoms}"
            if task.get("audit"):
                return {**_task_id(task), "status": "stale", "detail": detail}
            print(f"[WARNING] {out_dir}: {detail} -- rebuilding")
    elif task.get("audit"):
        return {**_task_id(task), "status": "failed", "detail": "no POSCAR"}

    if nv > n_O_sites:
        return {**_task_id(task), "status": "failed",
                "detail": f"nv={nv} exceeds {n_O_sites} oxygen sites"}

    struct = Poscar.from_file(task["base_poscar"]).structure
    struct.make_supercell(supercell)
    a_base, b_base = task["a_base"], task["b_base"]
    a_idx, b_idx, o_idx = _sublattice_indices(struct, a_base, b_base)
    if len(a_idx) != n_A or len(b_idx) != n_B or len(o_idx) != n_O_sites:
        return {**_task_id(task), "status": "failed",
                "detail": (f"supercell has {len(a_idx)}/{len(b_idx)}/{len(o_idx)} "
                           f"A/B/O sites, expected {n_A}/{n_B}/{n_O_sites}")}

    seed_key = task.get("seed_key")
    if not seed_key:
        return {**_task_id(task), "status": "failed",
                "detail": "random method needs a seed_key (set/group/comp)"}
    rng = np.random.default_rng(composition_seed(seed_key,
                                                 int(task.get("random_seed", 0) or 0)))

    # Draw order matters and must not depend on nv: A, then B, then the oxygen removal
    # order.  nv only slices the last one.
    a_perm = list(rng.permutation(a_idx))
    b_perm = list(rng.permutation(b_idx))
    o_perm = list(rng.permutation(o_idx))

    drop = []
    for perm, counts, base_el, dop_el in ((a_perm, a_counts, a_base, task["a_dopant"]),
                                          (b_perm, b_counts, b_base, task["b_dopant"])):
        n_base, n_dop, n_vac = counts
        for i in perm[n_base:n_base + n_dop]:
            struct.replace(int(i), dop_el, coords=struct[int(i)].frac_coords)
        drop.extend(int(i) for i in perm[n_base + n_dop:n_base + n_dop + n_vac])

    drop.extend(int(i) for i in o_perm[:nv])       # <- the nested vacancy sequence
    if drop:
        struct.remove_sites(sorted(set(drop)))

    if len(struct) != expect_atoms:
        return {**_task_id(task), "status": "failed",
                "detail": f"built {len(struct)} atoms, expected {expect_atoms}"}

    ensure_dir(out_dir)
    Poscar(struct.get_sorted_structure()).write_file(poscar_path)
    _finish(out_dir, task)
    return {**_task_id(task), "status": "random", "detail": ""}


def generate_one_structure(task):
    """Dispatch on task['method']: 'mcsqs' (default) or 'random'."""
    method = str(task.get("method") or "mcsqs").lower()
    if method == "random":
        return generate_one_random(task)
    if method == "mcsqs":
        return generate_one_sqs(task)
    return {**_task_id(task), "status": "failed",
            "detail": f"unknown sqs.method {method!r} (use 'mcsqs' or 'random')"}


def generate_one_sqs(task):
    """
    Build a single SQS structure.  Picklable and self-contained, so it is safe as a
    multiprocessing.Pool target (all three mcsqs helpers use cwd= rather than chdir).

    task keys: out_dir, base_poscar, supercell, a_base, a_dopant, x, b_base, b_dopant,
               y, nv, cutoffs {r2,r3,r4}, timeout_sec, clean_dir, skip_existing,
               and optionally a_vac / b_vac (cation vacancy fractions, default 0),
               max_attempts (mcsqs retries, default 2), audit (check only, write nothing)

    Returns a dict whose "status" is one of
        skipped | stale | ordered | sqs | sqs_fallback | failed
    """
    out_dir = str(task["out_dir"])
    poscar_path = os.path.join(out_dir, "POSCAR")
    supercell = [int(v) for v in task["supercell"]]
    n_A, n_B, n_O_sites, n_sites, _ = supercell_site_counts(supercell,
                                                            task["base_poscar"])

    x, y, nv = float(task["x"]), float(task["y"]), int(task["nv"])
    a_vac = float(task.get("a_vac") or 0.0)
    b_vac = float(task.get("b_vac") or 0.0)
    vac_frac = vacancy_fraction(nv, n_O_sites)
    # What a correct POSCAR must contain.  Checked against every POSCAR this function
    # writes AND against every one it is asked to skip, because a truncated bestsqs.out
    # produces a short POSCAR silently -- see convert_bestsqs_to_poscar.
    expect_atoms = ((n_A - int(round(a_vac * n_A)))
                    + (n_B - int(round(b_vac * n_B)))
                    + (n_O_sites - nv))

    if os.path.isfile(poscar_path):
        have = poscar_atom_count(poscar_path)
        if have == expect_atoms:
            if task.get("audit") or task.get("skip_existing", True):
                return {**_task_id(task), "status": "skipped", "detail": "POSCAR exists"}
        else:
            detail = (f"POSCAR has {have} atoms, expected {expect_atoms} -- rebuilding"
                      if not task.get("audit") else
                      f"POSCAR has {have} atoms, expected {expect_atoms}")
            if task.get("audit"):
                return {**_task_id(task), "status": "stale", "detail": detail}
            print(f"[WARNING] {out_dir}: {detail}")
    elif task.get("audit"):
        return {**_task_id(task), "status": "failed", "detail": "no POSCAR"}

    ensure_dir(out_dir)
    structure = Poscar.from_file(task["base_poscar"]).structure

    # A vacancy is an explicit "X" species on its sublattice, exactly as on the anion
    # sublattice: mcsqs optimises the correlations with X as an ordinary type and
    # convert_bestsqs_to_poscar's remove_species("X") deletes every one of them,
    # whichever sublattice it sits on.
    (a_occ, _, _), (b_occ, _, _) = site_occupancies(x, y, a_vac, b_vac)
    site_species = {
        0: [(task["a_base"], a_occ), (task["a_dopant"], x)]
           + ([("X", a_vac)] if a_vac > 0 else []),
        1: [(task["b_base"], b_occ), (task["b_dopant"], y)]
           + ([("X", b_vac)] if b_vac > 0 else []),
    }
    third = detect_third_site_element(structure, site_species)
    if not third:
        return {**_task_id(task), "status": "failed",
                "detail": "could not infer the anion sublattice"}
    anion = third[0]
    site_species[2] = ([(anion, 1.0 - vac_frac), ("X", vac_frac)] if vac_frac > 0
                       else [(anion, 1.0)])

    if _is_fully_ordered(site_species):
        write_ordered_supercell(structure, supercell, site_species, out_dir)
        _finish(out_dir, task)
        return {**_task_id(task), "status": "ordered", "detail": "plain supercell"}

    generate_rndstr_in(structure, os.path.join(out_dir, "rndstr.in"),
                       site_species, vac_frac)
    cut = task.get("cutoffs", {}) or {}
    if not run_corrdump(out_dir, r2=cut.get("r2", 5.58), r3=cut.get("r3", 4.24),
                        r4=cut.get("r4", 0)):
        return {**_task_id(task), "status": "failed",
                "detail": "cluster generation failed"}

    timeout = task.get("timeout_sec", 240)
    write_sqscell(out_dir, supercell)
    bestsqs = os.path.join(out_dir, "bestsqs.out")

    # Retry rather than give up.  Every way this loop fails is a truncated or missing
    # bestsqs.out, i.e. the mcsqs timeout landing on top of a write -- a race, not a
    # property of the composition, so the same search with a fresh seed usually just
    # works.  One extra mcsqs run on the ~0.3% that fail is far cheaper than either
    # losing the structure or, worse, keeping a short one.
    attempts = max(1, int(task.get("max_attempts", 2) or 1))
    status, detail = "failed", "mcsqs produced no bestsqs.out"
    for attempt in range(1, attempts + 1):
        # A corrupt bestsqs.out from the previous attempt (or from an earlier run of the
        # whole pipeline, since a failed task is never cleaned) would otherwise be
        # converted again instead of being replaced -- the retry would be a no-op.
        for stale in ("bestsqs.out", "bestcorr.out"):
            try:
                os.remove(os.path.join(out_dir, stale))
            except OSError:
                pass

        status = "sqs"
        run_mcsqs_rc(out_dir, timeout)
        if not os.path.isfile(bestsqs):
            run_mcsqs(out_dir, n_sites, timeout)
            status = "sqs_fallback"
        if not os.path.isfile(bestsqs):
            detail = "mcsqs produced no bestsqs.out"
            continue

        ok, detail = convert_bestsqs_to_poscar(out_dir, expect_atoms=expect_atoms)
        if ok:
            _finish(out_dir, task)
            return {**_task_id(task), "status": status,
                    "detail": f"succeeded on attempt {attempt}" if attempt > 1 else ""}

    return {**_task_id(task), "status": "failed",
            "detail": f"{detail} (after {attempts} attempt(s))"}
