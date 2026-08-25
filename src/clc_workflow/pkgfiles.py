"""
Locating the files that ship INSIDE the package: the node-side workers, the config
templates and run_all.sh.

These used to be loose files next to the drivers, which is why a new case had to start
with a copy of the whole directory.  Now they travel with the install and are
materialised into a case only when something actually needs them on disk there --
dpdispatcher uploading a worker, or `clc init` writing a starting config.
"""
import os
import shutil
from importlib.resources import files
from pathlib import Path

__all__ = ["worker", "template", "run_all_sh", "list_templates", "place"]


def _res(*parts):
    """Absolute path of a packaged file.  Installs here are always real directories --
    editable or plain `pip install` -- so a plain path is enough and callers can hand it
    to shutil or to bash without an as_file() context to keep alive."""
    p = Path(str(files("clc_workflow").joinpath(*parts)))
    if not p.exists():
        raise FileNotFoundError(f"packaged file missing: {'/'.join(parts)} (looked at {p})")
    return p


def worker(name):
    """dp_opt.py / dp_phonon.py -- the scripts that run on the compute node."""
    return _res("workers", name)


def template(name):
    return _res("templates", name)


def run_all_sh():
    return _res("run_all.sh")


def list_templates():
    d = Path(str(files("clc_workflow").joinpath("templates")))
    return sorted(p.name for p in d.iterdir()
                  if p.is_file() and p.suffix in (".yaml", ".json"))


def place(src, dest, label=""):
    """
    Put `src` at `dest`, preferring a hardlink and falling back to a copy.

    A hardlink so that a shared model store costs its bytes once no matter how many
    cases reference it; a copy when the link cannot be made (different filesystem --
    Dropbox and an external scratch disk being the usual pair -- or a filesystem with no
    hardlinks at all).  Either way dpdispatcher sees an ordinary file under work_base,
    which is the only thing it will upload.

    Returns "linked", "copied" or "current".
    """
    src, dest = Path(src), Path(dest)
    if dest.exists():
        # same inode, or same content-length-and-mtime: nothing to do.  Comparing
        # st_ino first matters for the model, where the alternative is hashing 8 MB on
        # every submission.
        s, d = src.stat(), dest.stat()
        if (s.st_dev, s.st_ino) == (d.st_dev, d.st_ino):
            return "current"
        if s.st_size == d.st_size and int(s.st_mtime) == int(d.st_mtime):
            return "current"
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        return "linked"
    except OSError:
        shutil.copy2(src, dest)
        return "copied"
