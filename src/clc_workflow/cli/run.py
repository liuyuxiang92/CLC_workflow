#!/usr/bin/env python3
"""
`clc run` -- stages 1-9 unattended.

    clc run config.yaml [--from N] [--to N] [--dry-run] [--bob] ...
    nohup clc run config.yaml > run_all.out 2>&1 &      # survives logout

A thin exec of the packaged run_all.sh, whose retry / min-fraction / per-stage-log logic
is worth keeping in one place rather than reimplementing.  It is handed $CLC pointing
back at this entry point, so each stage runs through the same installed package, and it
inherits YOUR working directory -- pipeline_logs/ lands where you started it.

Every option is passed through untouched; run `clc run --help` for the full list.
"""
import os
import shutil
import sys

from clc_workflow import pkgfiles


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    script = pkgfiles.run_all_sh()

    bash = shutil.which("bash")
    if bash is None:
        sys.exit("[ERROR] clc run needs bash on PATH")

    # How the stages call back into the package.  It has to be ONE executable, because
    # run_all.sh invokes it as "$CLC" <sub> <config>.  main() puts the console script's
    # own path here, which pins the stages to THIS install rather than to whatever PATH
    # resolves -- the case that matters when several conda envs each have a clc.
    clc = os.environ.get("CLC") or shutil.which("clc")
    if not clc:
        sys.exit("[ERROR] cannot find the `clc` executable to run the stages with.\n"
                 "        `clc run` shells out to run_all.sh, which calls back in per\n"
                 "        stage, so the console script has to be on PATH.  Either\n"
                 "        `pip install -e /path/to/CLC_workflow`, or set CLC=/path/to/clc.")

    env = dict(os.environ)
    env["CLC"] = clc
    env.setdefault("PYTHON", sys.executable)

    os.execve(bash, [bash, str(script), *argv], env)


if __name__ == "__main__":
    main()
