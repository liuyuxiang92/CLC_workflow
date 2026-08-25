#!/usr/bin/env python3
"""
The one submitter: stages 3, 5 and 7 through dpdispatcher.

    clc submit {md|opt|phonon} config.yaml [--dry-run] [--limit N]
                     [--sets set_01] [--redo]

Replaces job_submit.py, execute_parrallel_opt.sh, execute_parrallel_phonon.sh,
prepare_mpi_phonon.py, prepare_submit.py and prepare_submit_phonon.py.

All of those existed to split a path list into chunks, copy a script per chunk, patch a
hardcoded line number in the copy, and submit it.  dpdispatcher's `group_size` in
resources.json already does the chunking, so none of it is needed: one Task per o*
directory, and group_size decides how many share a job.

Task selection is "has the stage's input, lacks its output", so a resubmission after
partial failure is just a re-run -- which is what collect_uncal.py and
find_uncalculated_dir.py were doing by hand.

Shared files (the worker script, optionally the model) go in forward_common_files and are
referenced from the command with the constant relative prefix ../../../.. , since every
o* directory sits at the same depth set_NN/<group>/<comp>/o####.  If your dpdispatcher
lacks forward_common_files, pass --stage-files-per-task to put them in each task's
forward_files instead (job_submit.py already proves '../model.ckpt.pt' works).
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from clc_workflow.clc_config import (load_config, dispatch_summary, resolve_dispatch,
                                     stage_model, MANIFEST_NAME, REL_PREFIX)
from clc_workflow.staging import ensure_model, ensure_workers, model_store_dirs

STAGE_INPUTS = {
    "md": ["conf.lmp", "input.lammps"],
    "opt": ["POSCAR_md_avg"],
    "phonon": ["optimized_POSCAR"],
}
STAGE_DONE = {
    "md": "traj.lammpstrj",
    "opt": "opt_result.json",
    "phonon": "stability.txt",
}


def build_command(cfg, stage):
    """The per-task shell command.  Parameters are baked in, so workers stay stateless."""
    d = cfg["dispatch"].get(stage, {}) or {}
    if d.get("command"):
        return d["command"]

    if stage == "opt":
        o = cfg["opt"]
        parts = [f"python {REL_PREFIX}/dp_opt.py",
                 f"--model {stage_model(cfg, 'opt')[0]}",
                 f"--fmax {o['fmax']}",
                 f"--nsteps {int(o['nsteps'])}",
                 f"--pressure-gpa {o['pressure_gpa']}",
                 f"--traj-interval {int(o['traj_interval'])}"]
        if o.get("model_head"):
            parts.insert(2, f"--head {o['model_head']}")
        return " ".join(parts) + " > dp_opt.log 2>&1"

    if stage == "phonon":
        p = cfg["phonon"]
        sc = p["supercell_matrix"]
        diag = [sc[0][0], sc[1][1], sc[2][2]] if isinstance(sc[0], (list, tuple)) else list(sc)
        t = p["temperatures"]
        parts = [f"python {REL_PREFIX}/dp_phonon.py",
                 f"--model {stage_model(cfg, 'phonon')[0]}",
                 f"--supercell {diag[0]} {diag[1]} {diag[2]}",
                 f"--displacement {p['displacement']}",
                 f"--mesh {p['mesh'][0]} {p['mesh'][1]} {p['mesh'][2]}",
                 f"--temps {t['start']}:{t['stop']}:{int(t['num'])}"]
        if p.get("model_head"):
            parts.insert(2, f"--head {p['model_head']}")
        parts.append("--delete-band-yaml" if p.get("delete_band_yaml_after_stability", True)
                     else "--keep-band-yaml")
        return " ".join(parts) + " > dp_phonon.log 2>&1"

    raise ValueError(f"no command configured for stage {stage!r}")


def _from_dict(cls, d, what):
    """
    Machine/Resources from an already-merged dict.

    load_from_dict is the normal API and is what load_from_json calls internally; the
    temp-file fallback is only for a dpdispatcher old enough to lack it, since writing
    the dict out would otherwise leave a copy of machine.json's password on disk.
    """
    if hasattr(cls, "load_from_dict"):
        return cls.load_from_dict(d)
    import json
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=f"_{what}.json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(d, f)
        os.chmod(tmp, 0o600)
        return cls.load_from_json(tmp)
    finally:
        os.unlink(tmp)


def select_tasks(cfg, stage, sets=None, redo=False, limit=None):
    root = cfg["output_root"]
    man_path = os.path.join(root, MANIFEST_NAME)
    if not os.path.isfile(man_path):
        sys.exit(f"{man_path} not found -- run `clc sqs` first")
    man = pd.read_csv(man_path)
    if sets:
        man = man[man["set"].isin(sets)]

    needed = STAGE_INPUTS[stage]
    done_file = STAGE_DONE[stage]
    ready, missing_input, already = [], 0, 0
    for _, row in man.iterrows():
        d = os.path.join(root, row["path"])
        if not all(os.path.isfile(os.path.join(d, f)) for f in needed):
            missing_input += 1
            continue
        if not redo and os.path.isfile(os.path.join(d, done_file)):
            already += 1
            continue
        ready.append(row["path"])
    if limit:
        ready = ready[:limit]
    return ready, {"total": len(man), "missing_input": missing_input,
                   "already_done": already, "ready": len(ready)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["md", "opt", "phonon"])
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true",
                    help="print tasks, command and staged files; import nothing")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sets", default=None, help="comma list of set names")
    ap.add_argument("--redo", action="store_true",
                    help="resubmit even where the output already exists")
    ap.add_argument("--stage-files-per-task", action="store_true",
                    help="put shared files in each task's forward_files instead of "
                         "Submission(forward_common_files=...)")
    ap.add_argument("--ignore-missing-model", action="store_true",
                    help="submit even though model_path is relative and not present "
                         "locally (only if the node provides it some other way)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    stage = args.stage
    sets = [s.strip() for s in args.sets.split(",")] if args.sets else None
    paths, stats = select_tasks(cfg, stage, sets=sets, redo=args.redo, limit=args.limit)

    dsp = cfg["dispatch"]
    scfg = dsp.get(stage, {}) or {}
    command = build_command(cfg, stage)
    fwd = list(scfg.get("forward_files") or STAGE_INPUTS[stage])
    bwd = list(scfg.get("backward_files") or [])
    common = list(scfg.get("forward_common_files") or [])

    # Put the uploadable things under output_root before anything looks for them there.
    # dpdispatcher's work base IS output_root and it will not reach outside it, so a
    # worker that ships in the package and a checkpoint that lives in a shared store both
    # have to be materialised here -- which is what used to make a new case start with a
    # copy of the whole directory.
    ensure_workers(cfg, common, dry=args.dry_run)
    ensure_model(cfg, stage, dry=args.dry_run)

    # Ship the model alongside the worker script when it is here rather than in the
    # image, so the path referenced on the node and the file uploaded cannot disagree --
    # a mismatch fails only on the compute node, after the queue wait.
    #
    # md is staged even when it has a custom command, unlike opt/phonon: its model is
    # named in input.lammps (written by `clc md`, via the same stage_model), not on the
    # command line, so overriding the command does not take the model with it.
    model_warning = None
    if stage == "md" or not scfg.get("command"):
        model = str(cfg[stage]["model_path"])
        try:
            used_path, to_stage = stage_model(cfg, stage)
        except ValueError as e:
            sys.exit(f"[ERROR] {e}")
        if to_stage and to_stage not in common:
            common.append(to_stage)
            print(f"[*] staging model  : {to_stage}  (found in {cfg['output_root']})")
        elif os.path.isabs(model):
            print(f"[*] model          : {model}  (absolute -- must exist on the node)")
        elif model not in fwd and model not in common:
            # Relative, not here, and not being uploaded: the task directory is created
            # fresh on the node and only receives forward_files, so nothing can put this
            # file where the deck/command will look.  It is a guaranteed failure, but one
            # that only surfaces after the queue wait, so say so before submitting.
            model_warning = (
                f"model {model!r} will not exist on the compute node.\n"
                f"    The command references it at the work base, which is right -- but "
                f"nothing\n"
                f"    is uploading it there, so the task will not find it.\n"
                f"    It was looked for next to {args.config}, in "
                f"{cfg['output_root']}, and in the model store\n"
                f"    ({', '.join(str(d) for d in model_store_dirs(cfg)) or 'no store configured'}), "
                f"and is in none of them.\n"
                f"    Fix by either:\n"
                f"      - putting {model} beside the config, or in a directory named\n"
                f"        by model_store / $CLC_MODEL_DIR (it is then hardlinked in\n"
                f"        and uploaded once per submission), or\n"
                f"      - setting {stage}.model_path to its absolute path on the node.\n"
                f"    Then rerun `clc md --force`, since the path is baked into "
                f"input.lammps." if stage == "md" else
                f"model {model!r} will not exist on the compute node "
                f"(relative, and not in {cfg['output_root']})."
            )

    print(f"[*] stage       : {stage}")
    print(f"[*] work_base   : {cfg['output_root']}")
    print(f"[*] manifest    : {stats['total']} rows -> ready {stats['ready']}, "
          f"missing input {stats['missing_input']}, already done {stats['already_done']}")
    print(f"[*] command     : {command}")
    print(f"[*] forward     : {fwd}")
    print(f"[*] backward    : {bwd}")
    print(f"[*] common      : {common}   (referenced as {REL_PREFIX}/<name>)")

    # Resolve machine/resources now, even for --dry-run: machine_type, image_name and
    # group_size are exactly the settings that are expensive to get wrong, so they have
    # to be visible before anything is submitted.
    machine_dict = resources_dict = None
    try:
        machine_dict, resources_dict, applied = resolve_dispatch(cfg, stage)
    except (FileNotFoundError, ValueError) as e:
        if not args.dry_run:
            sys.exit(f"[ERROR] {e}")
        applied = []
        print(f"[!] {e}")

    print(f"[*] machine     : {dsp['machine_json']}")
    print(f"[*] resources   : {dsp['resources_json']}")
    if machine_dict is not None:
        eff = dispatch_summary(machine_dict, resources_dict)
        by_key = {k: src for k, _, src in applied}
        shown = ("machine_type", "image_name", "group_size", "para_deg", "job_name")
        for key in shown:
            origin = by_key.get(key, f"from {os.path.basename(dsp['machine_json'])}"
                                     if key not in ("group_size", "para_deg")
                                     else f"from {os.path.basename(dsp['resources_json'])}")
            print(f"[*]   {key:<13}: {eff[key]}   ({origin})")
        group_size = int(eff["group_size"] or 1)
        para_deg = int(eff["para_deg"] or 1)
        n_jobs = -(-len(paths) // group_size)
        waves = -(-group_size // para_deg)
        note = (f", {para_deg} at a time -> {waves} wave(s) of wall time per job"
                if para_deg > 1 else " (one after another)")
        print(f"[*]   -> {len(paths)} task(s) in {n_jobs} job(s) of "
              f"<= {group_size}{note}")
        # para_deg above group_size is not an error, just wasted intent: a job only ever
        # holds group_size tasks, so nothing extra can run beside them.
        if para_deg > group_size:
            print(f"[!]   para_deg {para_deg} > group_size {group_size}: a job never holds "
                  f"more than {group_size} task(s), so only {group_size} run concurrently. "
                  f"Raise group_size to fill the node.")
        extra = [f"{k}={v}" for k, v, _ in applied if k not in shown]
        if extra:
            print(f"[*]   also overridden: {', '.join(extra)}")

    if args.stage_files_per_task and common:
        fwd = fwd + [f"{REL_PREFIX}/{c}" for c in common]
        common = []
        print(f"[*] per-task staging: forward_files -> {fwd}")

    if not paths:
        print("\n[*] nothing to submit.")
        return
    print(f"\n[*] first tasks:")
    for p in paths[:5]:
        print(f"      {p}")
    if len(paths) > 5:
        print(f"      ... and {len(paths) - 5} more")

    if model_warning:
        print(f"\n[WARNING] {model_warning}")

    if args.dry_run:
        print(f"\n[dry-run] {len(paths)} task(s); nothing submitted.")
        return

    # Submitting anyway would burn the queue wait to arrive at "Cannot open file".  The
    # escape hatch is for a setup this check cannot see, e.g. a model injected by the
    # image's entrypoint into each working directory.
    if model_warning and not args.ignore_missing_model:
        sys.exit(f"[ERROR] refusing to submit: {model_warning.splitlines()[0]}\n"
                 f"        See the fixes above, or pass --ignore-missing-model if the "
                 f"node really does provide it.")

    try:
        from dpdispatcher import Machine, Resources, Task, Submission
    except ImportError:
        sys.exit("[ERROR] dpdispatcher is not installed here. Use --dry-run to inspect, "
                 "or run this on the submitting host.")

    # load_from_dict, not load_from_json: the overrides above live only in these dicts,
    # so that config.yaml can retune a stage without editing (or duplicating) a file
    # that contains the Bohrium password.
    machine = _from_dict(Machine, machine_dict, "machine")
    resources = _from_dict(Resources, resources_dict, "resources")

    task_list = [Task(command=command, task_work_path=p,
                      forward_files=fwd, backward_files=bwd) for p in paths]

    kwargs = dict(work_base=cfg["output_root"], machine=machine,
                  resources=resources, task_list=task_list)
    if common:
        try:
            submission = Submission(forward_common_files=common, **kwargs)
        except TypeError:
            print("[warn] this dpdispatcher has no forward_common_files; "
                  "falling back to per-task forward_files")
            fwd = fwd + [f"{REL_PREFIX}/{c}" for c in common]
            task_list = [Task(command=command, task_work_path=p,
                              forward_files=fwd, backward_files=bwd) for p in paths]
            kwargs["task_list"] = task_list
            submission = Submission(**kwargs)
    else:
        submission = Submission(**kwargs)

    print(f"\n[*] submitting {len(task_list)} task(s) ...")
    submission.run_submission()
    print("[*] submission finished.")


if __name__ == "__main__":
    main()
