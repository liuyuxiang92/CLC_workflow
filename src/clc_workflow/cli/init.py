#!/usr/bin/env python3
"""
`clc init` -- start a new case directory.

    clc init                     scaffold in the current directory
    clc init runs/feco_sweep     scaffold there, creating it if needed
    clc init --template feco     start from config_feco.yaml instead
    clc init --local             use machine_local.json / resources_local.json
    clc init --list              show the available templates

Writes ONLY the things that are genuinely per-case: config.yaml, machine.json and
resources.json.  The packaged machine.json carries PLACEHOLDER credentials, because it
lives in a git repo; put a filled-in one at ~/.clc/machine.json and that is used instead.  No code, no worker scripts, no model checkpoint -- those come from the
installed package or the shared model store, and are put under output_root at submit
time.  A case directory is therefore three text files plus whatever the run produces.

Existing files are never overwritten; pass --force to replace them.
"""
import argparse
import shutil
import sys
from pathlib import Path

from clc_workflow import pkgfiles


# Credentials belong off the repo, so the packaged machine.json is a placeholder.  A
# filled-in one at ~/.clc/machine.json is preferred when it exists, which keeps the real
# password in exactly one place and still lets a new case start ready to submit.
PERSONAL_MACHINE = Path.home() / ".clc" / "machine.json"


def _machine_source(suffix):
    if not suffix and PERSONAL_MACHINE.is_file():
        return PERSONAL_MACHINE
    return pkgfiles.template(f"machine{suffix}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?", default=".",
                    help="where to scaffold (default: the current directory)")
    ap.add_argument("--template", default="config",
                    help="config template: 'config' (default) or 'feco', or a filename")
    ap.add_argument("--local", action="store_true",
                    help="use the local-run machine/resources JSONs instead of Bohrium's")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--list", action="store_true", help="list templates and exit")
    args = ap.parse_args(argv)

    if args.list:
        print("templates shipped with the package:")
        for name in pkgfiles.list_templates():
            print(f"  {name}")
        return 0

    name = args.template
    if not name.endswith((".yaml", ".yml")):
        name = "config.yaml" if name == "config" else f"config_{name}.yaml"
    try:
        cfg_src = pkgfiles.template(name)
    except FileNotFoundError:
        print(f"clc init: no template {name!r}.  Available:", file=sys.stderr)
        for t in pkgfiles.list_templates():
            print(f"  {t}", file=sys.stderr)
        return 2

    suffix = "_local" if args.local else ""
    wanted = [(cfg_src, "config.yaml"),
              (_machine_source(suffix), "machine.json"),
              (pkgfiles.template(f"resources{suffix}.json"), "resources.json")]

    dest_dir = Path(args.directory).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    wrote, skipped = [], []
    for src, out in wanted:
        dest = dest_dir / out
        if dest.exists() and not args.force:
            skipped.append(out)
            continue
        shutil.copyfile(src, dest)
        wrote.append(out)

    print(f"[*] case directory : {dest_dir}")
    for f in wrote:
        print(f"    wrote   {f}")
    for f in skipped:
        print(f"    kept    {f}  (already there; --force to replace)")

    if "machine.json" in wrote:
        if not suffix and PERSONAL_MACHINE.is_file():
            print(f"            credentials from {PERSONAL_MACHINE}")
        elif not suffix:
            print(f"            ^ PLACEHOLDER credentials -- fill in email/password/program_id,")
            print(f"              or put a filled-in machine.json at {PERSONAL_MACHINE}")
            print(f"              and future `clc init` runs will use that instead.")

    print("\nNext:")
    print(f"    $EDITOR {dest_dir / 'config.yaml'}      # base_poscar, families, model_path")
    print( "    clc run config.yaml --dry-run           # check what stage 1 would do")
    print( "    clc run config.yaml")
    print("\nThe POSCAR named by base_poscar is the one thing you still supply yourself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
