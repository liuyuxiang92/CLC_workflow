# CLC_workflow

A screening pipeline for perovskite oxygen carriers in **chemical looping beyond
combustion**, and the dataset tooling that turns its results — and measured properties —
into training sets for machine-learned potentials.

Everything is driven by one command, `clc`, and one config file. The package is installed
once; a case directory holds only `config.yaml`, `machine.json` and the data the pipeline
writes into it. **No code is ever copied into a case.**

```bash
pip install -e .            # from this directory
clc                         # list the subcommands
clc init my_case            # scaffold config.yaml + machine.json
clc run my_case/config.yaml # stages 1-9, unattended
```

## Two tracks

**The screening pipeline** (stages 1–9) generates SQS structures across a composition
grid, relaxes them with a DP model, runs phonons, and joins everything into
`results.csv` with ΔG of vacancy formation per composition and temperature. Stages 3, 5
and 7 run on remote GPUs through dpdispatcher; the rest run locally.

**The property-dataset tools** build dpdata training sets from measurements and split
them for cross-validation. They share no state with the pipeline beyond reading its
output, and `clc kfold`, `clc decorate` and `clc tasks` work on any dpdata set at all.

## Subcommands

| | | |
|---|---|---|
| **Setup** | `clc init [DIR]` | scaffold a case from the packaged templates |
| | `clc config-from-xlsx` | derive a `families:` block from a measurement spreadsheet |
| **Pipeline** | `clc run config.yaml` | stages 1–9 unattended (wraps `run_all.sh`) |
| 1 | `clc sqs` | generate SQS structures, write `manifest.csv` |
| 2 | `clc md` | write `conf.lmp` / `input.lammps` |
| 3/5/7 | `clc submit md\|opt\|phonon` | submit to the cluster via dpdispatcher |
| 4 | `clc collect-md` | average MD trajectories → `POSCAR_md_avg` |
| 6 | `clc collect-opt` | gather relaxations → `optimized_POSCAR` |
| 6.5 | `clc check-bob` | B–O–B distribution report |
| 8 | `clc collect` | join everything into `results.csv` |
| 9 | `clc plot` | heatmaps and ΔG series |
| | `clc status` | what is done and what is pending across the tree |
| **Datasets** | `clc delta` | measured Δδ + structures → dpdata, optionally K-folded |
| | `clc decorate` | template POSCAR + formula/label sheet → dpdata |
| | `clc kfold` | split an existing dpdata set into K folds |
| | `clc tasks` | folds → `data/task.000y/{train,valid}` per fold |
| **Analysis** | `clc convergence`, `clc compare` | convergence at x₀; experiment vs theory |

`clc <subcommand> --help` for a subcommand's own options. Stage numbers `1 2 8 9` are
accepted as aliases.

## Building a property training set

Three commands, each usable on its own.

**`clc delta`** joins an oxygen-capacity spreadsheet to the structures the pipeline
generated. One frame per (structure, measurement), with the pressure window and
temperature carried as `fparam` because Δδ is a change *between* two states rather than a
property of one:

```bash
clc delta config_feco.yaml --xlsx measurements.xlsx --source opt --out delta_dataset
```

**`clc decorate`** needs no pipeline output at all — just a template POSCAR and a sheet of
formulas and measured values:

```bash
clc decorate --poscar NiOOH.vasp --xlsx data.xlsx \
    --label-col exp --label-name overpotential --n-configs 5 --kfold 5
```

Each formula's substitutable sites are relabelled `--n-configs` times with different random
decorations, all sharing that formula's label. A formula names an ensemble, not a
structure; several decorations are what tell the model which details of one arrangement
are irrelevant to the property. Site fractions become whole site counts by largest
remainder, and a cell that cannot represent them is an error rather than a silent
approximation.

**`clc kfold`** folds a dataset that already exists, whatever wrote it:

```bash
clc kfold delta_dataset -k 5 --out kfold_dataset
```

Point it at the parent of an existing `train/` + `valid/` split and it pools both sides and
redeals them. It slices the `.npy` files directly rather than round-tripping through
dpdata, so every array survives — including ones this code has never heard of.

### The split is by group, never by frame

All three use the same rule, and it is the one thing here worth understanding.

A compound contributes many frames: every temperature, every pressure window, every SQS
realisation or random decoration. They share a label and differ only in `fparam` or in how
the atoms are arranged. Splitting those frames at random would put a compound's 500 K point
in training and its 600 K point in validation — four folds out of five trained on a
near-duplicate of what they are scored on, and a validation error that measures
interpolation between near-identical rows rather than transfer to an unseen material.

So whole groups move together. `clc delta` groups by compound, `clc decorate` by formula,
`clc kfold` by composition — which for this pipeline recovers the same partition from the
data alone, since every realisation of one composition has the same atom counts.

Folds are balanced on **frames**, not on group count: groups carry very different frame
counts, and dealing them round-robin would leave the K scores measuring different amounts
of data. Perfect balance is not reachable — a fold cannot hold half a group — so weight the
K scores by fold size when you average them.

Each fold is written **once**, as `fold_0/ … fold_{K-1}/`. A fold's training set is the
other K−1 directories, and `folds.json` names the system directories for both sides of
every run.

### Laying out the training runs

`clc tasks` turns folds into one directory per training run:

```bash
clc tasks . --out data --input-template input.json --model dpa4.ckpt.pt
```

```
data/task.0000/
  input.json    systems.json    dpa4.ckpt.pt
  valid/iter_1/fold_0   train/iter_1/fold_1 fold_2 fold_3 fold_4
```

Fold *y* of every iteration is the validation set of `task.000y`. For an iterative
campaign, round N trains on iterations 1..N and each fold continues from the model that
same fold produced last round:

```bash
clc tasks . --upto iter_2 --out data_iter2 --input-template input.json \
    --task-model 'data_iter1/task.{task}/model.ckpt.pt'
```

Chaining fold-to-fold is not a convenience. One shared model from the previous round would
have trained on data from every fold, so every task's validation fold would already be in
its model's history and the five scores would stop being a cross-validation.

**Pass the systems explicitly.** Every task writes `systems.json` naming its system
directories one by one, and `--input-template` fills a copy of your `input.json` from that
list. Handing deepmd the parent directory relies on a tree walk, and a walk does not
descend into a symlink — the training set would come back empty with nothing obviously
wrong.

## How paths resolve

**Every path in `config.yaml` resolves against that file's directory, not your working
directory.** This is what frees the working directory: `clc sqs ~/cases/feco/config.yaml`
does the same thing from anywhere. `output_root: .` means "beside the config".

Subcommands are imported lazily, so `clc status` does not pay for pandas-plus-matplotlib
because `clc plot` exists — and still works in an environment where the plotting stack is
missing.

## Credentials

`machine.json` holds cluster credentials. The packaged template carries **placeholders**;
put a filled-in copy at `~/.clc/machine.json` (mode 600) and `clc init` uses that instead,
saying which source it used. Never copy a working `machine.json` into `src/clc_workflow/
templates/` — that directory is in this repo, and the file would go with it.

## Layout

```
src/clc_workflow/
  clc_config.py         config loading; every path is config-relative
  sqs_generator.py      composition.py  neutrality.py  ionic_data.py
  lammps_io.py  opt.py  phonon_analysis.py  thermo.py  bob_angles.py
  kfold.py              group K-fold: fold assignment + npy-level splitting
  rl_builder.py         MD-averaged builder for rl-matdesign's reward
  staging.py            puts workers + model under output_root
  pkgfiles.py           locates the packaged workers/templates
  cli/                  one module per subcommand; main.py dispatches
  workers/              dp_opt.py, dp_phonon.py — run on the compute node
  templates/            what `clc init` copies out
  run_all.sh            stages 1-9 in order, driven by `clc run`
```

## Optional dependencies

The base install stays slim so `clc status` and `clc submit` work in a minimal
environment:

```bash
pip install -e '.[submit]'    # dpdispatcher — the submitting machine
pip install -e '.[plot]'      # matplotlib, scipy
pip install -e '.[analysis]'  # phonopy, ase, openpyxl — spreadsheets, local phonon work
pip install -e '.[rl_reward]' # ase — rl_builder.py's MD-averaged builder
pip install -e '.[all]'
```

The dataset commands sit at different points on that scale. `clc kfold` and `clc tasks`
need nothing beyond numpy — they read and write the `.npy` files directly. `clc decorate`
and `clc delta` additionally need `openpyxl` (from `analysis`) to read the spreadsheet and
`dpdata` to read a POSCAR; `dpdata` is not declared here, since anywhere you are building
a deepmd training set already has it.

## Longer documentation

The case-level `README.md` that ships beside a working case goes much deeper: the
composition grid, why stage 4 averages rather than snapshots, how the workers reach the
compute node, GPU packing through `group_size` × `para_deg`, and the reasoning behind
each stage's "has my input, lacks my output" selection rule.
