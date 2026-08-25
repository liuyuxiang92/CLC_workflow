"""
Config loading and the family/task expansion shared by every driver.

One config.yaml drives all stages.  Per-family overrides are merged over the
top-level defaults, so the common case is one line per family.
"""
import copy
import json
import os
from pathlib import Path

import yaml

__all__ = ["load_config", "Family", "expand_families", "set_names",
           "MANIFEST_NAME", "STAGE_IO", "DISPATCH_OVERRIDES",
           "resolve_dispatch", "dispatch_summary", "REL_PREFIX", "stage_model"]

MANIFEST_NAME = "manifest.csv"

# Depth of set_NN/<family>/<comp>/o#### below output_root.  Every task directory sits at
# exactly this depth, so anything staged at the work base is reachable from a task by a
# fixed prefix -- that is what lets the model and the worker scripts be uploaded once per
# submission instead of once per structure.
REL_PREFIX = "../../../.."


def stage_model(cfg, stage):
    """
    (path to write into the command or the input deck, filename to stage) for `stage`.

    A relative model_path is resolved against config.yaml's own directory first, then
    output_root -- the same rule as dispatch.machine_json, because both are inputs you
    hand the pipeline rather than things it produces.  So `model_path: model.pt2` means
    "the file sitting next to this config", which is what it looks like it means.

    Both the LAMMPS deck and the dp_opt/dp_phonon command line are read with cwd = the
    task's o#### directory, where nothing puts a model, so a file that is found gets
    referenced through REL_PREFIX and shipped as a dpdispatcher common file.  An absolute
    path, or a name found nowhere locally, is passed through untouched -- that is the
    "model already lives on the compute node" case.

    Shared rather than duplicated because `clc md` bakes this path into input.lammps at
    generation time while `clc submit` decides what to upload at submission time.  If those
    two ever disagreed the failure would be LAMMPS's "Cannot open file", on the node,
    after the queue wait.
    """
    model = str(cfg[stage]["model_path"])
    if os.path.isabs(model):
        return model, None

    root = os.path.abspath(cfg["output_root"])
    for base in (cfg["_config_dir"], cfg["output_root"]):
        found = os.path.join(base, model)
        if not os.path.isfile(found):
            continue
        # dpdispatcher resolves forward_common_files against work_base (= output_root),
        # so the name it is given has to be relative to that -- which also makes the
        # REL_PREFIX path the task sees correct by construction.
        rel = os.path.relpath(os.path.abspath(found), root)
        if rel.startswith(os.pardir):
            raise ValueError(
                f"{stage}.model_path = {model} resolves to {os.path.abspath(found)}, "
                f"which is outside output_root ({root}).\n"
                f"dpdispatcher can only upload files under the work base, so copy it "
                f"into {root} or give its absolute path on the compute node instead.")
        return f"{REL_PREFIX}/{rel}", rel

    # Not here YET.  It is still a relative path, and a relative model can only ever be
    # at the work base on the node: the o#### directory is created fresh and receives
    # nothing but forward_files.  So reference it through REL_PREFIX anyway.  Writing the
    # bare name instead -- which this used to do -- bakes "compressed_model.pt2" into
    # input.lammps, and that is not a path that becomes correct later; it is one that
    # fails with "Cannot open file" on the node no matter what turns up at output_root.
    #
    # This matters because the deck is written at stage 2 and the model is staged at
    # submission, which can be days apart: a checkpoint that arrives from the model store
    # in between used to leave a deck that had already been baked wrong.  Nothing is
    # staged (second element stays None), so the caller still warns about the upload.
    norm = os.path.normpath(model)
    if norm.startswith(os.pardir):
        # Escapes output_root, so no prefix can make it reachable -- and dpdispatcher
        # could not upload it either.  Pass it through and let the caller complain.
        return model, None
    return f"{REL_PREFIX}/{norm}", None

# Keys config.yaml may set to override machine.json / resources.json, and where each
# one lands in that JSON.  Overrides are applied to an in-memory copy at submit time --
# the files themselves are never rewritten, so machine.json stays the single place the
# Bohrium credentials and the batch/context type live.
DISPATCH_OVERRIDES = {
    "machine_type": ("machine", ["remote_profile", "input_data", "machine_type"]),
    "image_name":   ("machine", ["remote_profile", "input_data", "image_name"]),
    "job_name":     ("machine", ["remote_profile", "input_data", "job_name"]),
    "disk_size":    ("machine", ["remote_profile", "input_data", "disk_size"]),
    "on_demand":    ("machine", ["remote_profile", "input_data", "on_demand"]),
    "platform":     ("machine", ["remote_profile", "input_data", "platform"]),
    "group_size":   ("resources", ["group_size"]),
    # How many of a group's tasks run AT THE SAME TIME on the node.  group_size alone puts
    # N tasks in one job and runs them one after another; para_deg makes dpdispatcher
    # launch them in background batches with a `wait` between, so a 400-atom DP job that
    # leaves a V100 half idle can have two or four of them resident at once.  Wall time per
    # job is then ceil(group_size / para_deg) x per-task, not group_size x per-task.
    # Leave it at 1 for phonon, which is the memory-hungry stage.
    "para_deg":     ("resources", ["para_deg"]),
    # Shell setup dpdispatcher writes into the job script's header, before the command.
    # `source` of a path that is not in the image prints "No such file or directory" for
    # every job, so these belong next to image_name rather than in a file you forget to
    # revisit when the image changes.  [] means "emit nothing".
    "source_list":  ("resources", ["source_list"]),
    "module_list":  ("resources", ["module_list"]),
    "envs":         ("resources", ["envs"]),
    # Arbitrary shell lines, emitted verbatim before (and after) the task command.
    # source_list can only `source` a file, so anything that is a COMMAND rather than a
    # file -- `conda activate <env>` being the usual one -- has to go here.  Keep the
    # environment setup here rather than folding it into dispatch.<stage>.command: setting
    # `command` replaces the one `clc submit` builds from the stage's own config, so the
    # model path, mesh and supercell would stop following config.yaml.
    "prepend_script": ("resources", ["prepend_script"]),
    "append_script":  ("resources", ["append_script"]),
}

# stage -> (required input file, produced output file used for "is it done")
STAGE_IO = {
    "sqs":     ("POSCAR", "POSCAR"),
    "md":      ("conf.lmp", "traj.lammpstrj"),
    "md_avg":  ("traj.lammpstrj", "POSCAR_md_avg"),
    "opt":     ("POSCAR_md_avg", "opt_result.json"),
    "optimized": ("opt_result.json", "optimized_POSCAR"),
    "phonon":  ("optimized_POSCAR", "stability.txt"),
}

_DEFAULTS = {
    "base_poscar": "./POSCAR",
    "output_root": ".",
    # Directories to pull a relative model_path from when the case does not have it.
    # Checked after config.yaml's own directory, before $CLC_MODEL_DIR and ~/.clc/models.
    # The file is hardlinked into output_root at submit time (copied across filesystems),
    # because dpdispatcher will only upload what is under its work base -- so one stored
    # checkpoint can back any number of cases without costing its bytes each time.
    "model_store": [],
    "n_sets": 1,
    "set_prefix": "set",
    "a_base": "Sr",
    "b_base": "Fe",
    "x_values": [round(0.125 * i, 3) for i in range(9)],
    "y_values": [round(0.125 * i, 3) for i in range(9)],
    "vacancies": {"n_min": 0, "n_max": 32},
    # Cation vacancies, fractions of the A / B sites.  Normally set per family block, not
    # here -- an off-stoichiometric composition is one compound, not a sweep.
    "a_vac": 0.0,
    "b_vac": 0.0,
    "screen_charge_neutrality": True,
    "screening_delta_range": [0.0, 0.5],
    "families": [{"a_dopant": "Ca", "b_dopant": "Mn"}],
    "sqs": {
        # "mcsqs"  -- correlation-optimised special quasirandom structures (the default)
        # "random" -- plain random occupation, no ATAT needed.  Much faster, and the
        #             oxygen vacancies of one composition are NESTED: o0002 is o0001
        #             with one more oxygen removed, so a vacancy delta-G is a step along
        #             one removal path instead of a difference between unrelated cells.
        "method": "mcsqs",
        # Shifts the whole random ensemble.  The per-structure seed is
        # sha256(random_seed | set/group/comp), so changing this redraws every
        # structure while keeping the nesting intact; leaving it alone makes reruns
        # reproduce the tree exactly.  Ignored by mcsqs.
        "random_seed": 0,
        "supercell": [2, 2, 2],
        "cutoffs": {"r2": 5.58, "r3": 4.24, "r4": 0},
        "timeout_sec": 240,
        "n_procs": 4,
        "pin_cpus": False,
        "clean_dir": True,
        "skip_existing": True,
        # Refuse a composition that cannot be laid down on the supercell exactly.  mcsqs
        # is given concentrations, not counts, so an unrealisable one is rounded silently
        # and the tree fills up with the wrong compound.  Turn it off only when you have
        # decided to accept the rounding -- gen_sqs then prints what it rounded to.
        "require_commensurate": True,
        # mcsqs runs until the timeout kills it, and a kill on top of one of its
        # bestsqs.out rewrites truncates the file.  Truncation in the header raises;
        # truncation in the atom block parses SILENTLY into a structure with fewer atoms.
        # Both are caught by the atom-count check, and both are cured by running the
        # search again with a fresh seed, which is what this buys.
        "max_attempts": 2,
    },
    "md": {
        "model_path": "model.pth",
        # Multi-task head.  Unlike opt/phonon -- where deepmd.calculator.DP takes head=
        # at load time -- LAMMPS cannot select a head: pair_style deepmd (v3.0.0) accepts
        # only out_freq/out_file/fparam/aparam/ttm/fparam_from_compute/
        # aparam_from_compute/atomic/relative/relative_v/virtual_len/spin_norm.  So the
        # head must already be baked into model_path by
        #     dp --pt freeze -c <ckpt> -o model.pth --head <HEAD>
        # This field records WHICH head that file was frozen from: it is written into
        # input.lammps as provenance and checked for consistency, never emitted as a
        # pair_style argument.
        "model_head": None,
        # What the LAMMPS *build* provides, as opposed to what the run should do.
        # No `plugin load` by default: the dpa4C build has its pair style compiled in and
        # treats the line as a hard error.  Only a stock deepmd-kit LAMMPS needs it, and
        # then it is one line -- plugin: libdeepmd_lmp.so, with pair_style: deepmd.
        "pair_style": "deepmd",
        "atom_style": "atomic",
        "plugin": None,
        "temperature_k": 300,
        "run_ps": 200,
        "timestep_ps": 0.002,
        "thermo_freq": 10,
        "dump_freq": 100,
        "pressure_bar": 0.0,
        "tau_t": 0.1,
        "tau_p": 0.5,
        "velocity_seed": "auto",
        # Stage 4 averages atomic positions over this much of the END of the trajectory to
        # build POSCAR_md_avg.  20 ps is long enough to average out libration (the period
        # of an octahedral tilt mode is well under a ps) and short enough to stay inside
        # the equilibrated tail of a 200 ps run.  0 = write the last frame instead.
        "average_last_ps": 20.0,
        # Per-atom RMS spread about its own mean, in A, above which an atom is taken to
        # have changed sites during the window -- its mean would be a point on no site at
        # all, so it keeps its last-frame position.  Vibration gives ~0.1 A and a hop
        # >= ~1 A, so anything in this range separates them.  null disables the fallback.
        "avg_rms_max": 0.5,
    },
    "opt": {
        "model_path": "model.ckpt.pt",
        "model_head": "DOWNSTREAM_DATA",
        "fmax": 0.005,
        "pressure_gpa": 0,
        "nsteps": 2000,
        "traj_interval": 5,
    },
    "phonon": {
        "model_path": "model.ckpt.pt",
        "model_head": "DOWNSTREAM_DATA",
        "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "displacement": 0.01,
        "mesh": [20, 20, 20],
        "temperatures": {"start": 670.0, "stop": 1070.0, "num": 5},
        "delete_band_yaml_after_stability": True,
    },
    "analysis": {
        "avg_mode": "simple",
        "temps": [670, 1070],
        "s_config_a": 2.0,
        "mu_G": {670: -5.1768, 1070: -5.6699},
        "mu_H": {670: -4.37519, 1070: -4.304685},
    },
    "dispatch": {
        "machine_json": "machine.json",
        "resources_json": "resources.json",
        # Every DISPATCH_OVERRIDES key is spliced in below, at this level and under each
        # stage, defaulting to None -- "leave the JSON file's value alone".  Generating
        # them rather than listing them keeps the two in step when the table grows.
        "md": {"command": "lmp -i input.lammps > log.lammps",
               "forward_files": ["conf.lmp", "input.lammps"],
               "backward_files": ["log.lammps", "traj.lammpstrj"],
               "forward_common_files": []},
        "opt": {"command": None,      # built from opt.* config
                "forward_files": ["POSCAR_md_avg"],
                "backward_files": ["opt.traj", "opt_result.json"],
                "forward_common_files": ["dp_opt.py"]},
        "phonon": {"command": None,   # built from phonon.* config
                   "forward_files": ["optimized_POSCAR"],
                   "backward_files": ["energy.txt", "thermal_properties.yaml",
                                      "stability.txt"],
                   "forward_common_files": ["dp_phonon.py"]},
    },
}

for _key in DISPATCH_OVERRIDES:
    _DEFAULTS["dispatch"].setdefault(_key, None)
    for _stage in ("md", "opt", "phonon"):
        _DEFAULTS["dispatch"][_stage].setdefault(_key, None)
del _key, _stage


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Family:
    """One (A_dopant, B_dopant) family with its resolved grids."""

    def __init__(self, cfg, spec):
        self.a_base = spec.get("a_base", cfg["a_base"])
        self.b_base = spec.get("b_base", cfg["b_base"])
        self.a_dopant = spec["a_dopant"]
        self.b_dopant = spec["b_dopant"]
        self.x_values = [float(v) for v in spec.get("x_values", cfg["x_values"])]
        self.y_values = [float(v) for v in spec.get("y_values", cfg["y_values"])]
        # Cation vacancies.  Fractions of the A / B SITES, so the base species gets
        # 1 - x - a_vac: Sr0.75Ca0.2Fe0.875Co0.125O3 is x=0.2, a_vac=0.05.  Unlike x and y
        # these are per BLOCK rather than a grid -- an off-stoichiometric composition is a
        # single compound, not a row of them, so a block is one (a_vac, b_vac) and the
        # x_values/y_values it carries are that compound's cation ratios.
        self.a_vac = float(spec.get("a_vac", cfg.get("a_vac", 0.0)) or 0.0)
        self.b_vac = float(spec.get("b_vac", cfg.get("b_vac", 0.0)) or 0.0)
        for site, vac, dops in (("a", self.a_vac, self.x_values),
                                ("b", self.b_vac, self.y_values)):
            if not 0.0 <= vac < 1.0:
                raise ValueError(f"{self.name}: {site}_vac = {vac} outside [0, 1)")
            bad = [d for d in dops if d + vac > 1.0 + 1e-9]
            if bad:
                raise ValueError(
                    f"{self.name}: {site}_vac = {vac} leaves no room for "
                    f"{'x' if site == 'a' else 'y'} = {bad} -- x/y are fractions of the "
                    f"SITES, so dopant + vacancy must not exceed 1")
        vac = _deep_merge(cfg["vacancies"], spec.get("vacancies", {}))
        self.nv_min = int(vac["n_min"])
        self.nv_max = int(vac["n_max"])
        self.screen = bool(spec.get("screen_charge_neutrality",
                                    cfg["screen_charge_neutrality"]))
        self.screen_delta_range = tuple(spec.get("screening_delta_range",
                                                 cfg["screening_delta_range"]))

    @property
    def name(self):
        return f"{self.a_base}{self.a_dopant}_{self.b_base}{self.b_dopant}"

    @property
    def type_map(self):
        """LAMMPS atom types 1..5 for this family: [A_base, A_dop, B_base, B_dop, O]."""
        return [self.a_base, self.a_dopant, self.b_base, self.b_dopant, "O"]

    def nv_range(self, n_O_sites):
        if self.nv_max > n_O_sites:
            raise ValueError(f"{self.name}: n_max={self.nv_max} exceeds the "
                             f"{n_O_sites} oxygen sites in the supercell")
        return list(range(self.nv_min, self.nv_max + 1))

    def __repr__(self):
        return (f"Family({self.a_base}/{self.a_dopant}-{self.b_base}/{self.b_dopant}, "
                f"nx={len(self.x_values)}, ny={len(self.y_values)}, "
                f"nv={self.nv_min}..{self.nv_max})")


def load_config(path):
    """Read config.yaml, merge over defaults, resolve paths, validate."""
    path = Path(path).expanduser().resolve()
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _deep_merge(_DEFAULTS, user)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)

    root = Path(cfg["output_root"])
    cfg["output_root"] = str((path.parent / root).resolve() if not root.is_absolute()
                             else root.resolve())
    bp = Path(cfg["base_poscar"])
    cfg["base_poscar"] = str((path.parent / bp).resolve() if not bp.is_absolute()
                             else bp.resolve())

    sc = list(cfg["sqs"]["supercell"])
    if len(sc) != 3 or any(int(v) < 1 for v in sc):
        raise ValueError(f"sqs.supercell must be three positive integers, got {sc}")
    cfg["sqs"]["supercell"] = [int(v) for v in sc]

    if int(cfg["n_sets"]) < 1:
        raise ValueError("n_sets must be >= 1")
    if not cfg["families"]:
        raise ValueError("no families configured")
    return cfg


def dispatch_json_path(cfg, key):
    """Absolute path of dispatch.machine_json / dispatch.resources_json."""
    p = cfg["dispatch"][key]
    return p if os.path.isabs(p) else os.path.join(cfg["_config_dir"], p)


def resolve_dispatch(cfg, stage):
    """
    machine.json + resources.json for `stage`, with config.yaml's overrides applied.

    Returns (machine_dict, resources_dict, applied), where `applied` is a list of
    (key, value, source) for every override that took effect so --dry-run can show
    exactly what the stage will run on.  Neither JSON file is modified on disk: they
    remain the only copy of the Bohrium credentials and of batch_type/context_type.

    Precedence, most specific first:
        dispatch.<stage>.<key>   >   dispatch.<key>   >   the value already in the JSON
    """
    dsp = cfg["dispatch"]
    scfg = dsp.get(stage) or {}

    loaded = {}
    for kind, key in (("machine", "machine_json"), ("resources", "resources_json")):
        path = dispatch_json_path(cfg, key)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"dispatch.{key} = {cfg['dispatch'][key]} not found at {path} -- "
                f"dpdispatcher needs it")
        with open(path) as f:
            try:
                loaded[kind] = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: not valid JSON ({e})") from None

    applied = []
    for key, (kind, route) in DISPATCH_OVERRIDES.items():
        if scfg.get(key) is not None:
            value, source = scfg[key], f"dispatch.{stage}.{key}"
        elif dsp.get(key) is not None:
            value, source = dsp[key], f"dispatch.{key}"
        else:
            continue
        node = loaded[kind]
        for k in route[:-1]:
            node = node.setdefault(k, {})
        node[route[-1]] = value
        applied.append((key, value, source))

    gs = loaded["resources"].get("group_size", 1)
    if int(gs) < 1:
        raise ValueError(f"group_size must be >= 1, got {gs} (stage {stage})")
    return loaded["machine"], loaded["resources"], applied


def dispatch_summary(machine_dict, resources_dict):
    """Effective value of every overridable key -- walks the same routes as the setter,
    so what --dry-run prints cannot drift from what is actually submitted."""
    src = {"machine": machine_dict, "resources": resources_dict}
    out = {}
    for key, (kind, route) in DISPATCH_OVERRIDES.items():
        node = src.get(kind)
        for k in route:
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        out[key] = node
    return out


def expand_families(cfg):
    return [Family(cfg, spec) for spec in cfg["families"]]


def set_names(cfg):
    n = int(cfg["n_sets"])
    return [f"{cfg['set_prefix']}_{i:02d}" for i in range(1, n + 1)]


def manifest_path(cfg):
    return os.path.join(cfg["output_root"], MANIFEST_NAME)
