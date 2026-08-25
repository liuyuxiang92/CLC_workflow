#!/usr/bin/env python3
"""
Stage 2 -- write conf.lmp + input.lammps into every o* directory listed in manifest.csv.

    clc md config.yaml [--limit N] [--force]

NPT aniso at md.temperature_k for md.run_ps, matching the proven
code/md/SrCaFeCoO/clc_300k/input.lammps.  The atom-type map is per family
([A_base, A_dop, B_base, B_dop, O] -> types 1..5); types with zero atoms are still
declared so the numbering stays constant within a family and fixed-index analysis
(analyze_runs.py treats types 3,4 as the B site) keeps working.
"""
import argparse
import os
import sys
import zlib
from pathlib import Path

import pandas as pd
from pymatgen.io.vasp import Poscar

from clc_workflow.clc_config import load_config, stage_model, MANIFEST_NAME
from clc_workflow.lammps_io import freeze_command, render_input_lammps, write_conf_lmp
from clc_workflow.staging import ensure_model, model_store_dirs

def velocity_seed(cfg, rel_path):
    """Deterministic per-structure seed, so a rerun reproduces the same MD."""
    spec = cfg["md"].get("velocity_seed", "auto")
    if str(spec).lower() != "auto":
        return int(spec)
    return 1 + (zlib.crc32(rel_path.encode()) % (2**31 - 2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="rewrite conf.lmp/input.lammps even if they already exist")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = cfg["output_root"]
    man_path = os.path.join(root, MANIFEST_NAME)
    if not os.path.isfile(man_path):
        sys.exit(f"{man_path} not found -- run `clc sqs` first")
    man = pd.read_csv(man_path)
    if args.limit:
        man = man.head(args.limit)

    md = cfg["md"]
    nsteps = int(round(float(md["run_ps"]) / float(md["timestep_ps"])))
    print(f"[*] {len(man)} structures; {md['run_ps']} ps / {md['timestep_ps']} ps "
          f"= {nsteps} steps at {md['temperature_k']} K, NPT aniso")
    # LAMMPS opens this path with cwd = the o#### task directory, so a model sitting at
    # output_root has to be named through REL_PREFIX -- and `clc submit` uploads it as a
    # common file on the strength of the same call.  Deciding it here rather than writing
    # md.model_path verbatim is what stops the deck and the upload from drifting apart.
    # Pull the checkpoint in from the shared store first, so stage_model resolves it
    # relative to output_root and the deck names a path the node will actually have.
    ensure_model(cfg, "md")
    try:
        model_in_deck, staged = stage_model(cfg, "md")
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")
    print(f"[*] pair_style {md['pair_style']}, atom_style {md['atom_style']}"
          + (f", plugin load {md['plugin']}" if md.get("plugin") else ", no plugin load"))
    print(f"[*] model written into input.lammps: {model_in_deck}")
    if staged:
        print(f"    found in {root} -- `clc submit` will ship it as a common file")
    elif os.path.isabs(model_in_deck):
        print(f"    absolute -- it must already exist on the compute node")
    else:
        # A relative name that is not here cannot be made true: the task directory is
        # created fresh on the node and only receives forward_files.  Say it now; the
        # alternative is LAMMPS reporting "Cannot open file" after the queue wait.
        stores = ", ".join(str(d) for d in model_store_dirs(cfg)) or "none configured"
        print(f"[WARNING] {md['model_path']} was not found next to {args.config}, in")
        print(f"          {root}, or in the model store ({stores}),")
        print(f"          so the deck names it relative to each task directory -- where")
        print(f"          nothing puts it, and LAMMPS will fail with 'Cannot open file'.")
        print(f"          Either put it beside the config or in a model_store directory,")
        print(f"          or set md.model_path to its absolute path on the node, then")
        print(f"          rerun with --force.")

    head = md.get("model_head")
    # A training checkpoint is not a frozen model; pair_style deepmd will fail to load it
    # on the compute node, i.e. after the jobs are queued.  Catch it here instead.
    if str(md["model_path"]).endswith((".ckpt.pt", ".ckpt")):
        sys.exit(
            f"[ERROR] md.model_path = {md['model_path']} looks like a training checkpoint.\n"
            f"        LAMMPS needs a frozen model (.pth or .pb); only dp_opt.py and\n"
            f"        dp_phonon.py can read a checkpoint directly.  Freeze it first:\n"
            f"            {freeze_command('model.pth', head or '<HEAD>', md['model_path'])}\n"
            f"        then set md.model_path: model.pth"
            + ("" if head else "\n        (and md.model_head to the head you froze)"))
    if head:
        print(f"[*] multi-task head: {head}")
        print(f"    LAMMPS cannot select a head -- {md['model_path']} must already be")
        print(f"    frozen from it:  {freeze_command(md['model_path'], head, cfg['opt']['model_path'])}")

    written = skipped = missing = 0
    for _, row in man.iterrows():
        d = os.path.join(root, row["path"])
        poscar = os.path.join(d, "POSCAR")
        if not os.path.isfile(poscar):
            missing += 1
            continue
        conf, inp = os.path.join(d, "conf.lmp"), os.path.join(d, "input.lammps")
        if not args.force and os.path.isfile(conf) and os.path.isfile(inp):
            skipped += 1
            continue

        type_map = str(row["lammps_type_map"]).split()
        structure = Poscar.from_file(poscar).structure
        write_conf_lmp(conf, structure, type_map, source=os.path.relpath(poscar, root))
        with open(inp, "w") as f:
            f.write(render_input_lammps(
                type_map=type_map, model_path=model_in_deck,
                temperature_k=md["temperature_k"], nsteps=nsteps,
                timestep_ps=md["timestep_ps"], thermo_freq=md["thermo_freq"],
                dump_freq=md["dump_freq"], pressure_bar=md["pressure_bar"],
                tau_t=md["tau_t"], tau_p=md["tau_p"], model_head=head,
                pair_style=md["pair_style"], atom_style=md["atom_style"],
                plugin=md.get("plugin"),
                seed=velocity_seed(cfg, row["path"])))
        written += 1

    print(f"[*] written={written} skipped={skipped} missing_POSCAR={missing}")
    if written:
        print("[*] next: clc submit md config.yaml --dry-run")


if __name__ == "__main__":
    main()
