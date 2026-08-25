"""
Putting into output_root the two kinds of file that have to be there.

dpdispatcher only uploads what sits under its work base, which is output_root.  That is
the reason a case directory used to need its own copy of dp_opt.py, dp_phonon.py and an
8 MB checkpoint: the uploader cannot reach outside the tree, so everything it ships had
to be duplicated per case by hand.

Nothing about that requires the duplication to be MANUAL, though.  The workers now ship
inside the package and the checkpoint can live in one shared store; both are materialised
into output_root here, at submit time, right before dpdispatcher looks for them.  The
worker on the node is then the installed version by construction, which the hand-copying
could not guarantee -- a case directory cut months ago kept running whatever dp_opt.py
was current the day it was created.
"""
import os
from pathlib import Path

from clc_workflow import pkgfiles

__all__ = ["model_store_dirs", "ensure_workers", "ensure_model"]


def model_store_dirs(cfg):
    """
    Where a checkpoint may live besides the case itself, most specific first:
    config.yaml's `model_store`, then $CLC_MODEL_DIR (os.pathsep-separated), then
    ~/.clc/models.  Relative entries in the config are relative to config.yaml.
    """
    out = []
    store = cfg.get("model_store") or []
    if isinstance(store, (str, os.PathLike)):
        store = [store]
    for s in store:
        p = Path(s).expanduser()
        out.append(p if p.is_absolute() else Path(cfg["_config_dir"]) / p)
    for s in (os.environ.get("CLC_MODEL_DIR") or "").split(os.pathsep):
        if s.strip():
            out.append(Path(s).expanduser())
    out.append(Path.home() / ".clc" / "models")
    return [p for p in out if p.is_dir()]


def ensure_workers(cfg, names, verbose=True, dry=False):
    """
    Materialise any packaged worker named in forward_common_files into output_root.

    A name that is not a packaged worker is left alone -- that is a file of yours, and
    "I don't ship it" is not a reason to touch it.  A packaged worker already present is
    refreshed when it differs, so editing the installed package (editable install) is
    picked up on the next submission instead of silently not being.
    """
    root = Path(cfg["output_root"])
    for name in names:
        try:
            src = pkgfiles.worker(Path(name).name)
        except FileNotFoundError:
            continue
        if dry:
            if verbose:
                print(f"[*] would stage    : {name}  (packaged worker -> {root})")
            continue
        what = pkgfiles.place(src, root / name)
        if verbose and what != "current":
            print(f"[*] staging worker : {name}  ({what} from the installed package)")


def ensure_model(cfg, stage, verbose=True, dry=False):
    """
    Make cfg[stage].model_path resolvable under output_root, pulling it from the shared
    store if that is where it lives.

    Absolute paths mean "already on the compute node" and are left untouched.  A model
    that is already under output_root is left untouched.  Otherwise it is looked for next
    to config.yaml and then in the store, and hardlinked (or copied) into output_root so
    stage_model's relpath -- and therefore the ../../../.. the node sees -- resolves.

    Returns the source it came from, or None if nothing was staged.
    """
    model = str(cfg[stage]["model_path"])
    if os.path.isabs(model):
        return None

    root = Path(cfg["output_root"])
    if (root / model).is_file():
        return None

    for base in [Path(cfg["_config_dir"]), *model_store_dirs(cfg)]:
        cand = base / model
        if not cand.is_file():
            continue
        if dry:
            if verbose:
                print(f"[*] would stage    : {model}  (from {base} -> {root})")
            return cand
        what = pkgfiles.place(cand, root / model)
        if verbose:
            print(f"[*] staging model  : {model}  ({what} from {base})")
        return cand
    return None
