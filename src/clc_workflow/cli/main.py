#!/usr/bin/env python3
"""
`clc` -- the single entry point for the whole pipeline.

    clc init [DIR]              scaffold a new case directory
    clc run config.yaml         stages 1-9 unattended (run_all.sh)
    clc sqs|md|... config.yaml  one stage
    clc status config.yaml      what is done, what is pending

Nothing here needs to live next to your data.  The package is installed once
(`pip install -e CLC_workflow`) and every subcommand reads its paths from the config
file you name, resolved against THAT FILE's directory -- so the working directory is
free and no code has to be copied into a case.

Subcommands are imported lazily.  `clc status` should not pay for pandas-plus-matplotlib
because `clc plot` exists, and more to the point it should still work in an environment
where the plotting stack is missing.
"""
import importlib
import os
import sys

# subcommand -> (module under clc_workflow.cli, one-line help)
COMMANDS = {
    "init":     ("init",                "scaffold config.yaml + machine.json in a new case dir"),
    "run":      ("run",                 "run stages 1-9 unattended (wraps run_all.sh)"),
    "sqs":      ("gen_sqs",             "stage 1 -- generate SQS structures, write manifest.csv"),
    "md":       ("gen_md",              "stage 2 -- write conf.lmp / input.lammps"),
    "submit":   ("submit",              "stages 3/5/7 -- submit md|opt|phonon via dpdispatcher"),
    "collect-md":  ("collect_md",       "stage 4 -- average MD trajectories"),
    "collect-opt": ("collect_opt",      "stage 6 -- gather optimisation results"),
    "collect":  ("collect_results",     "stage 8 -- join everything into results.csv"),
    "plot":     ("plot_heatmap",        "stage 9 -- heatmaps and series plots"),
    "status":   ("status",              "what is done and what is pending across the tree"),
    "check-bob":("check_bob",           "stage 6.5 -- B-O-B distribution report"),
    "delta":    ("build_delta_dataset", "build the measured-vs-computed delta dataset"),
    "convergence": ("convergence_x0",   "convergence analysis at x0"),
    "compare":  ("compare_exp_theory",  "compare experiment against theory"),
}

# accepted, but not advertised in the listing: the stage numbers people already use
ALIASES = {"1": "sqs", "2": "md", "8": "collect", "9": "plot"}


def _usage(stream=sys.stdout):
    print(__doc__.strip(), file=stream)
    print("\nSubcommands:", file=stream)
    for name, (_, blurb) in COMMANDS.items():
        print(f"  {name:<14} {blurb}", file=stream)
    print("\n`clc <subcommand> --help` for a subcommand's own options.", file=stream)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        _usage()
        return 0
    if argv[0] in ("-V", "--version"):
        from clc_workflow import __version__
        print(f"clc-workflow {__version__}")
        return 0

    name = ALIASES.get(argv[0], argv[0])
    if name not in COMMANDS:
        print(f"clc: unknown subcommand {argv[0]!r}\n", file=sys.stderr)
        _usage(sys.stderr)
        return 2

    modname, _ = COMMANDS[name]
    mod = importlib.import_module(f"clc_workflow.cli.{modname}")

    # Remember the real entry point before rewriting argv[0] for argparse: `clc run`
    # hands this to run_all.sh so the stages call back into THIS install rather than
    # whatever `clc` PATH happens to resolve to.
    if os.path.isfile(sys.argv[0]) and os.access(sys.argv[0], os.X_OK):
        os.environ.setdefault("CLC", os.path.abspath(sys.argv[0]))

    # argparse takes prog from basename(sys.argv[0]), so the subcommand's own --help
    # says "usage: clc sqs ..." rather than "usage: gen_sqs.py ...".
    sys.argv[0] = f"clc {argv[0]}"
    return mod.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
