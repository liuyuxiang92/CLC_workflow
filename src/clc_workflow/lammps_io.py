"""
LAMMPS data-file / trajectory I/O for the CLC workflow.

Consolidates the three near-duplicate writers that used to live in
code/md/SrCaFeCoO/{make_conf.py, vasp2lmp.py, gen_md_runs.py}.  Differences from
those: the atom-type map is passed in per family instead of being hardcoded to
Sr/Ca/Fe/Co/O, and masses come from pymatgen rather than a hardcoded dict.

The type map is a list of element symbols; LAMMPS atom types are 1..len(type_map)
in that order.  Types with zero atoms are still declared, so the map stays
constant across a family and fixed-index analysis keeps working.
"""
import os
import numpy as np
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.vasp import Poscar

__all__ = [
    "masses_for", "cell_to_box", "write_conf_lmp", "render_input_lammps",
    "read_last_frame", "read_frames_since", "average_frames",
    "write_poscar_from_frame", "INPUT_TEMPLATE",
]


def masses_for(type_map):
    """Atomic masses (amu) for a type map, from pymatgen."""
    return [float(Element(el).atomic_mass) for el in type_map]


def cell_to_box(cell):
    """
    Cell vectors (3x3) -> LAMMPS box (ax, by, cz, xy, xz, yz).

    Same convention as the original vasp2lmp.py / gen_md_runs.py.
    """
    cell = np.asarray(cell, dtype=float)
    ax = np.linalg.norm(cell[0])
    bx = np.dot(cell[1], cell[0]) / ax
    by = np.linalg.norm(np.cross(cell[0], cell[1])) / ax
    cx = np.dot(cell[2], cell[0]) / ax
    cy = (np.dot(cell[1], cell[2]) - bx * cx) / by
    cz = np.sqrt(max(np.dot(cell[2], cell[2]) - cx**2 - cy**2, 0.0))
    return ax, by, cz, bx, cx, cy


def write_conf_lmp(out_path, structure, type_map, source=""):
    """
    Write a LAMMPS data file for `structure` (a pymatgen Structure).

    All elements of `structure` must appear in `type_map`; unused types are still
    declared with their masses so the type numbering is family-constant.
    """
    type_of = {el: i + 1 for i, el in enumerate(type_map)}
    missing = sorted({s.specie.symbol for s in structure} - set(type_of))
    if missing:
        raise ValueError(f"{out_path}: elements {missing} are not in type_map {type_map}")

    ax, by, cz, xy, xz, yz = cell_to_box(structure.lattice.matrix)
    tilt = max(abs(xy), abs(xz), abs(yz)) > 1e-8
    masses = masses_for(type_map)

    lines = [f"conf.lmp from {source or 'structure'} ({' '.join(type_map)})", ""]
    lines.append(f"{len(structure)} atoms")
    lines.append(f"{len(type_map)} atom types")
    lines.append("")
    lines.append(f"   0.0 {ax:.10f} xlo xhi")
    lines.append(f"   0.0 {by:.10f} ylo yhi")
    lines.append(f"   0.0 {cz:.10f} zlo zhi")
    if tilt:
        lines.append(f"   {xy:.10f} {xz:.10f} {yz:.10f} xy xz yz")
    lines += ["", "Masses", ""]
    for el, m in zip(type_map, masses):
        lines.append(f"{type_of[el]} {m}  # {el}")
    lines += ["", "Atoms # atomic", ""]

    # cartesian coordinates expressed in the LAMMPS box frame
    frac = np.asarray(structure.frac_coords, dtype=float) % 1.0
    box = np.array([[ax, 0.0, 0.0], [xy, by, 0.0], [xz, yz, cz]], dtype=float)
    cart = frac @ box
    for aid, (site, r) in enumerate(zip(structure, cart), 1):
        lines.append(f"{aid} {type_of[site.specie.symbol]} "
                     f"{r[0]:.10f} {r[1]:.10f} {r[2]:.10f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


INPUT_TEMPLATE = """variable        NSTEPS          equal {nsteps}
variable        THERMO_FREQ     equal {thermo_freq}
variable        DUMP_FREQ       equal {dump_freq}
variable        TEMP            equal {temperature:.6f}
variable        PRES            equal {pressure:.6f}
variable        TAU_T           equal {tau_t:.6f}
variable        TAU_P           equal {tau_p:.6f}

units           metal
boundary        p p p
atom_style      {atom_style}

neighbor        1.0 bin

atom_modify     map yes
read_data       conf.lmp            # masses {type_comment} come from the data file

{plugin_line}{head_comment}pair_style      {pair_style} {model}
pair_coeff      * * {elements}

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${{THERMO_FREQ}}

dump            2 all custom ${{DUMP_FREQ}} traj.lammpstrj id type x y z

velocity        all create ${{TEMP}} {seed}
fix             1 all npt temp ${{TEMP}} ${{TEMP}} ${{TAU_T}} aniso ${{PRES}} ${{PRES}} ${{TAU_P}}

timestep        {timestep}
run             ${{NSTEPS}} upto
"""


def freeze_command(model_path, model_head, checkpoint="model.ckpt.pt"):
    """The `dp freeze` that must have produced a head-specific LAMMPS model file."""
    return f"dp --pt freeze -c {checkpoint} -o {model_path} --head {model_head}"


def render_input_lammps(type_map, model_path, temperature_k, nsteps, timestep_ps,
                        thermo_freq=10, dump_freq=100, pressure_bar=0.0,
                        tau_t=0.1, tau_p=0.5, seed=443428, model_head=None,
                        pair_style="deepmd", atom_style="atomic", plugin=None):
    """
    NPT-aniso DeePMD input deck, templated from the proven clc_300k/input.lammps.

    `pair_style` / `plugin` are what the LAMMPS build dictates, and the two travel
    together.  No `plugin load` line is emitted unless `plugin` is given: a build with
    the pair style compiled in (dpa4C's `dpa4spin`) errors on it outright, and only a
    stock deepmd-kit build -- which loads `libdeepmd_lmp.so` at runtime to get
    `pair_style deepmd` -- has anything to load.

    `model_head` is provenance only.  A multi-task head CANNOT be selected here the way
    dp_opt.py / dp_phonon.py select it via DP(model=..., head=...): no deepmd-kit 3.x
    pair style takes a head keyword, so the head has to be frozen into the model file
    beforehand.  Recording it in the deck is what makes an accidental swap to the wrong
    branch visible when reading a run directory months later.
    """
    type_comment = " ".join(f"{i+1}={el}" for i, el in enumerate(type_map))
    plugin_line = f"plugin load     {plugin}\n" if plugin else ""
    head_comment = (
        f"# model frozen from multi-task head '{model_head}':\n"
        f"#   {freeze_command(model_path, model_head)}\n"
        f"# (pair_style {pair_style} takes no head argument -- it is baked into the file)\n"
    ) if model_head else ""
    return INPUT_TEMPLATE.format(
        nsteps=int(nsteps), thermo_freq=int(thermo_freq), dump_freq=int(dump_freq),
        temperature=float(temperature_k), pressure=float(pressure_bar),
        tau_t=float(tau_t), tau_p=float(tau_p),
        type_comment=f"({type_comment})", model=model_path, head_comment=head_comment,
        plugin_line=plugin_line, pair_style=pair_style, atom_style=atom_style,
        elements=" ".join(type_map), seed=int(seed), timestep=float(timestep_ps),
    )


# ---------------------------------------------------------------- trajectory
def _tail_bytes(path, nbytes):
    size = os.path.getsize(path)
    start = max(0, size - nbytes)
    with open(path, "rb") as f:
        f.seek(start)
        return start, f.read()


def read_last_frame(traj_path, max_tail=1 << 26):
    """
    Return the last *complete* frame of a LAMMPS custom dump as a dict:
        {step, n_atoms, cell (3x3), lo (3,), types (N,), coords (N,3), triclinic}

    Reads from the end of the file, growing the window until a complete frame is
    found, so cost is independent of trajectory length.  A truncated final frame
    (killed job) is skipped in favour of the previous complete one.
    """
    marker = b"ITEM: TIMESTEP"
    size = os.path.getsize(traj_path)
    window = 1 << 20
    while True:
        start, blob = _tail_bytes(traj_path, window)
        offsets = []
        idx = blob.find(marker)
        while idx != -1:
            offsets.append(idx)
            idx = blob.find(marker, idx + 1)
        for off in reversed(offsets):
            frame = _parse_frame(blob[off:].decode("utf-8", "replace"))
            if frame is not None:
                return frame
        if start == 0 or window >= max_tail:
            raise ValueError(f"{traj_path}: no complete frame found")
        window *= 4


def _frame_offsets(blob, marker=b"ITEM: TIMESTEP"):
    """Byte offsets of every frame header in `blob`."""
    offsets, idx = [], blob.find(marker)
    while idx != -1:
        offsets.append(idx)
        idx = blob.find(marker, idx + 1)
    return offsets


def _parse_window(blob):
    """
    Every complete frame in `blob`, in file order.

    Each frame is decoded from its own slice rather than from `blob[off:]`, which is what
    read_last_frame does -- that is fine when you stop at the first success, but decoding
    the whole tail once per frame is quadratic when you want all of them.
    """
    offsets = _frame_offsets(blob)
    bounds = zip(offsets, offsets[1:] + [len(blob)])
    frames = [_parse_frame(blob[a:b].decode("utf-8", "replace")) for a, b in bounds]
    return [f for f in frames if f is not None], offsets


def read_frames_since(traj_path, min_step, max_tail=1 << 31):
    """
    Every complete frame with step >= `min_step`, in step order.

    Reads from the end like read_last_frame, so the cost follows the length of the window
    asked for rather than the length of the trajectory -- the point of averaging only the
    last 20 ps of a 200 ps run is lost if the whole file has to be parsed to find it.

    The tail window is grown until it reaches back past `min_step` (or to the start of the
    file).  After the first read the frame spacing in bytes and in steps is known, so the
    next window is sized to the number of frames actually wanted instead of blindly
    quadrupling; the usual case is therefore two reads.  A truncated final frame is
    dropped, as in read_last_frame.
    """
    window = 1 << 20
    while True:
        start, blob = _tail_bytes(traj_path, window)
        frames, offsets = _parse_window(blob)
        if not frames:
            if start == 0 or window >= max_tail:
                raise ValueError(f"{traj_path}: no complete frame found")
            window *= 4
            continue
        # start == 0 means the window is the whole file: nothing older exists to find.
        if frames[0]["step"] <= min_step or start == 0 or window >= max_tail:
            break
        window = min(_next_window(window, blob, frames, offsets, min_step), max_tail)

    sel = [f for f in frames if f["step"] >= min_step]
    if not sel:                                # min_step past the end of the run
        sel = frames[-1:]
    n_atoms = {f["n_atoms"] for f in sel}
    if len(n_atoms) > 1:
        raise ValueError(f"{traj_path}: atom count changes within the window: "
                         f"{sorted(n_atoms)}")
    return sorted(sel, key=lambda f: f["step"])


def _next_window(window, blob, frames, offsets, min_step):
    """Tail size that should reach `min_step`, from the frame spacing just measured."""
    if len(frames) < 2 or len(offsets) < 2:
        return window * 4
    per_frame_bytes = (len(blob) - offsets[0]) / len(offsets)
    per_frame_steps = frames[1]["step"] - frames[0]["step"]
    if per_frame_steps <= 0 or per_frame_bytes <= 0:
        return window * 4
    n_wanted = (frames[-1]["step"] - min_step) / per_frame_steps + 2
    return max(int(n_wanted * per_frame_bytes * 1.1), window * 4)


def _frac_coords(frame):
    """Fractional coordinates of a frame, wrapped into [0, 1)."""
    cart = np.asarray(frame["coords"], dtype=float) - np.asarray(frame["lo"], dtype=float)
    cell = np.asarray(frame["cell"], dtype=float)
    return np.linalg.solve(cell.T, cart.T).T % 1.0


def average_frames(frames, rms_max=0.5):
    """
    Time-averaged ("mean") structure of `frames`, as (frame, stats).

    Positions are averaged in FRACTIONAL coordinates -- under `fix npt` the cell breathes,
    so a Cartesian average would mix coordinates measured against different boxes -- and
    against the first frame of the window with the minimum-image convention, so an atom
    that crosses a periodic boundary mid-window does not smear across the cell.  The cell
    itself is averaged too, and the mean fractional positions are expressed in it.

    This removes thermal libration and leaves the static distortion; the same computation
    on 20 ps windows is what code/md/SrCaFeCoO/feofe_windows.py reports as the
    mean-structure Fe-O-Fe angle, several degrees away from the instantaneous one.

    Averaging is only meaningful for an atom that stays on its site.  A vacancy-mediated
    oxygen hop puts one atom at two different sites during the window, and their mean is a
    point on no site at all -- typically a bottleneck between two B cations, where the
    relaxation starts with a large force on it.  Such atoms are detected by their RMS
    spread about their own mean (~0.1 A for vibration, >= ~1 A for a hop, so the two
    populations separate cleanly) and take their LAST-frame position instead, leaving the
    rest of the cell with the clean mean.  `rms_max=None` disables the fallback.
    """
    if not frames:
        raise ValueError("average_frames: no frames given")
    n_atoms = frames[0]["n_atoms"]
    types = frames[0]["types"]
    if any(f["n_atoms"] != n_atoms for f in frames):
        raise ValueError("average_frames: frames disagree on the atom count")
    if any(not np.array_equal(f["types"], types) for f in frames):
        # _parse_frame sorts by id, so this means the dump itself is inconsistent.
        raise ValueError("average_frames: atom types change between frames")

    ref = _frac_coords(frames[0])
    dsum = np.zeros((n_atoms, 3))
    d2sum = np.zeros((n_atoms, 3))
    cell_sum = np.zeros((3, 3))
    for f in frames:
        ds = _frac_coords(f) - ref
        ds -= np.round(ds)                     # minimum image, in fractional space
        dsum += ds
        d2sum += ds * ds
        cell_sum += np.asarray(f["cell"], dtype=float)

    n = len(frames)
    cell_avg = cell_sum / n
    dmean = dsum / n
    frac_mean = (ref + dmean) % 1.0

    # Per-atom RMS about its own mean.  Converting the fractional variance with the cell
    # edge lengths is exact for an orthogonal box, which is what `fix npt ... aniso`
    # produces, and a fair approximation for the triclinic case that should not arise.
    var = np.maximum(d2sum / n - dmean ** 2, 0.0)
    rms = np.sqrt(var @ (np.linalg.norm(cell_avg, axis=1) ** 2))

    hoppers = (rms > float(rms_max)) if rms_max else np.zeros(n_atoms, dtype=bool)
    if hoppers.any():
        frac_mean[hoppers] = _frac_coords(frames[-1])[hoppers]

    frame = {"step": frames[-1]["step"], "n_atoms": n_atoms, "cell": cell_avg,
             "lo": np.zeros(3), "types": types, "coords": frac_mean @ cell_avg,
             "triclinic": any(f["triclinic"] for f in frames)}
    stats = {"n_frames": n, "step_first": frames[0]["step"], "step_last": frames[-1]["step"],
             "n_hoppers": int(hoppers.sum()), "max_rms": float(rms.max()),
             "mean_rms": float(rms.mean())}
    return frame, stats


def _parse_frame(text):
    """Parse one frame from text starting at 'ITEM: TIMESTEP'.  None if incomplete."""
    lines = text.splitlines()
    try:
        if not lines[0].startswith("ITEM: TIMESTEP"):
            return None
        step = int(lines[1])
        if not lines[2].startswith("ITEM: NUMBER OF ATOMS"):
            return None
        n = int(lines[3])
        if not lines[4].startswith("ITEM: BOX BOUNDS"):
            return None
        triclinic = "xy" in lines[4]
        bounds = [list(map(float, lines[5 + i].split())) for i in range(3)]
        i = 8
        if not lines[i].startswith("ITEM: ATOMS"):
            return None
        cols = lines[i].split()[2:]
        i += 1
        rows = lines[i:i + n]
        if len(rows) < n:
            return None                      # truncated frame
        arr = np.array([r.split() for r in rows], dtype=float)
    except (IndexError, ValueError):
        return None

    ci = {name: k for k, name in enumerate(cols)}
    for need in ("id", "type", "x", "y", "z"):
        if need not in ci:
            return None
    arr = arr[np.argsort(arr[:, ci["id"]])]
    types = arr[:, ci["type"]].astype(int)
    coords = arr[:, [ci["x"], ci["y"], ci["z"]]]

    lo = np.array([b[0] for b in bounds], dtype=float)
    lx, ly, lz = (b[1] - b[0] for b in bounds)
    if triclinic:
        # LAMMPS writes xlo_bound etc. for triclinic; recover the true edges
        xy, xz, yz = (b[2] for b in bounds)
        lx -= max(0.0, xy, xz, xy + xz) - min(0.0, xy, xz, xy + xz)
        ly -= max(0.0, yz) - min(0.0, yz)
        cell = np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]], dtype=float)
    else:
        cell = np.diag([lx, ly, lz]).astype(float)

    return {"step": step, "n_atoms": n, "cell": cell, "lo": lo,
            "types": types, "coords": coords, "triclinic": triclinic}


def write_poscar_from_frame(out_path, frame, type_map, comment=""):
    """Write a POSCAR from a trajectory frame, mapping LAMMPS types via type_map."""
    types = frame["types"]
    if types.min() < 1 or types.max() > len(type_map):
        raise ValueError(f"{out_path}: types {types.min()}..{types.max()} outside "
                         f"type_map of length {len(type_map)}")
    species = [type_map[t - 1] for t in types]
    cart = np.asarray(frame["coords"], dtype=float) - np.asarray(frame["lo"], dtype=float)
    structure = Structure(Lattice(frame["cell"]), species, cart,
                          coords_are_cartesian=True)
    poscar = Poscar(structure.get_sorted_structure(),
                    comment=comment or f"step {frame['step']}")
    poscar.write_file(out_path)
    return out_path
