# Trace: CLC_workflow

<!-- concepts: perovskite-oxygen-vacancy-thermodynamics, sqs-structure-generation, dp-md-workflow-orchestration -->

## 2026-07-31 — Pipeline refactor: shared library modules

Building stages 1–8 of the SQS → MD → DP-opt → energy+phonon pipeline. Shared code goes
in `CLC_workflow/src/` so the local drivers in `structures/` stay thin.

**Design constraint discovered while planning the submitter:** the two worker scripts that
dpdispatcher ships to compute nodes (`dp_opt.py`, `dp_phonon.py`) must be **self-contained** —
they cannot import from `CLC_workflow/src`, because only the script itself is forwarded. So
their parameters travel as CLI flags baked into the job command by `submit.py`, keeping the
workers stateless and the command self-documenting. Everything else (local drivers) imports
freely from `src/`.

**New modules:**
- `lammps_io.py` — one home for the LAMMPS data writer that was triplicated across
  `md/SrCaFeCoO/{make_conf.py, vasp2lmp.py, gen_md_runs.py}`. Type map is now a per-family
  argument instead of hardcoded Sr/Ca/Fe/Co/O; masses from `pymatgen.Element`. Also holds
  `read_last_frame()`, which seeks backward from EOF in growing windows so cost is independent
  of trajectory length, and skips a truncated final frame from a killed job.

**Analysis-script corrections completed earlier this session** (see
`../plot_heatmap_avg_series_bolz_linear.py`, original kept as `.orig`): S_config had a sign
error, a spurious ×1000, and a formula-unit count frozen at 8; δ was being recovered from the
rounded `o####` directory label instead of `nv/n_fu`; μ(T) silently fell back to the 670 K
value and was added once per pair regardless of Δnv. Net effect on ΔG was −0.6 to −1.6 eV
typical, ≈ −7 eV for pairs spanning a gap. 16/16 checks pass against a synthetic
known-answer tree.

**Key numeric identity worth remembering:** `0.008314 [kJ/mol/K] * 0.01036427 [eV per kJ/mol]`
= `8.6169e-5` = k_B in eV/K exactly. Any further scaling of that product is a unit bug — this
is what made the spurious ×0.001 provable rather than a judgement call.

## 2026-07-31 — Stages 1–8 built and tested end to end

All ten drivers written plus five library modules. Verified on a 120-structure synthetic
tree (2 families × 2 sets × nv 0–3) with a stand-in `mcsqs`.

**Two things the test caught that are worth remembering:**

1. **ATAT `bestsqs.out` is not `rndstr.in`.** pymatgen's parser branches on the first
   line: 6 fields → lattice parameters, coord system from a/b/c/α/β/γ, atoms start at
   line 4 **in units of that coord system**; otherwise 3+3 header lines. My stub first
   wrote Cartesian coordinates under a 6-field header, which pymatgen read as fractional
   and produced a structure with 1.15 Å min distance instead of 1.9255 Å. Real mcsqs is
   fine — but the same trap applies to anything hand-writing these files.

2. **`collect_opt.py`'s safe-distance screen caught that bad geometry immediately**, which
   was an accidental but genuine validation of the KD-tree port: it rejected every
   structure with the correct Fe–O / Mn–O threshold and message.

**Gap handling has two distinct regimes** (worth knowing when reading the dnv column):
an o-directory that exists but lacks artifacts keeps the pair sequence contiguous with
NaN values (safe); only a *genuinely absent* directory makes a pair span dnv > 1, and
that is where μ scaling matters. Confirmed: removing an o-dir produced `o953 - o984`
with dnv=2 charged 2μ = −10.3536 eV, against the legacy single μ — a 5.18 eV error.

**Resumability works by construction:** every stage keys off artifact presence, so a
rerun reports `skipped=120` / `ready 0, already done 120` and `status.py --pending
<stage>` lists exactly what remains.

## 2026-07-31 — Stage 9: numeric x axis, pO2 shift, sign comparison

`plot_heatmap_avg_series_bolz_linear.py` reworked: heatmaps are opt-in (`--heatmaps`), the
line-plot x axis is now the numeric vacancy fraction `x_pair = nv_lo/n_fu` instead of the
`o_lo - o_hi` label, and ΔG=0 is solved at several oxygen partial pressures via the source
paper's eq. (1), `ΔG(T,pO2) = ΔG^DFT(T,P0) + ½ kB T ln(pO2/P0)`.

**Verification anchor worth keeping:** that formula reproduces the paper's own correction
table to 5 decimals — 670 K gives −0.00644 / −0.30064 eV per vacancy at 0.8 / 3e-5 atm,
1070 K gives −0.01029 / −0.48013. Any future refactor of `dG_pO2_shift` should be checked
against those four numbers rather than re-derived.

**The pitfall this exposed.** "Shift the 1 atm fitting line and re-read the crossing" gives
`x0(p) = -(b + Δ)/m` with the slope unchanged — but only when *every* pair has `dnv = 1`.
Across a missing `o*` directory the correction is `dnv·Δ`, so it is no longer a constant:
both slope and intercept move. Measured on the synthetic gapped series, the closed form was
off by 0.021 in x (≈1.4 vacancies) and the slope moved 6.85 → 6.41. So the code refits at
each pressure, and `summarize_pO2_thresholds` only uses the exact delta-method `dx0_std`
when `all_dnv_1`, falling back to quadrature and flagging `dx0_std_exact=False` otherwise.
Same `dnv` trap as the μ scaling found earlier — worth assuming it recurs anywhere a
"per-vacancy" quantity is added to a pair.

**Deployment pitfall (hit on the cluster first run).** `python gen_sqs.py config.yaml` on
`/share/perovskite/srcafeco_md` died with `ModuleNotFoundError: No module named 'clc_config'`:
the drivers had been copied over but `CLC_workflow/` had not, and the old one-liner
`sys.path.insert(0, __file__.parent/"CLC_workflow"/"src")` fails silently in that case.
All seven local drivers now resolve the package through `_clc_src()` — `$CLC_WORKFLOW_SRC`
first, then `CLC_workflow/src` in `__file__.parents` — and exit with the rsync command to
run rather than a bare import error. Note `pip install -e CLC_workflow` is **not** a route:
`find_packages(where="src")` returns `[]` because the modules are top-level `.py` files with
no sub-packages, so the install succeeds and provides nothing.

**Sign comparison is now structural, not a side script.** `F_state` is kept in
`state_series_raw.csv` so `recompute_state_G()` can replay the whole averaging → pairs →
threshold chain for a = −2 / +2 / 0 from one collection pass. On the real fitted data
(`thresholds_DG_continuous.csv`, 81 comps × 31 points) the correct sign roughly doubles
dΔG/dnv (0.053 → 0.096 eV/vacancy at 670 K) and therefore *halves* the pO2-driven vacancy
swing, 5.5 → 3.1 vacancies at 670 K and 8.7 → 3.8 at 1070 K; the wrong sign collapses the
slope (R² 1.00 → 0.41) and the swing becomes meaningless (−3997 … +529 vacancies).

### EARS — Progress (2026-07-31 18:12)
<!-- concepts: composition-grid-enumeration, perovskite-doping-site-degeneracy -->

**Bug found by the user: `composition.generate_valid_combinations` invents grid points.**
`x_values: [0.625]`, `y_values: [0.750]` should give one compound; it gave 9. The function
split enumeration into interior / edges / corners and the edge+corner blocks called
`add_entry` with literal `0.0` and `1.0` regardless of whether those values appeared in
`x_vals` / `y_vals` — so every run silently gained the four undoped/fully-doped corners
plus the x=0, x=1, y=0, y=1 edges.

**Why it went unnoticed:** the default config *is* the full 0…1 grid, where the injected
endpoints coincide with requested ones. 9×9 gives 81 rows either way. The bug only shows
on a sub-grid that omits an endpoint — i.e. exactly the "just run this one composition"
case. Charge neutrality then prunes the phantoms unevenly, so the count looks plausible.
Generalisable: **an enumerator that hardcodes boundary values can only be validated by a
grid that excludes them** — a full-range test is structurally blind to this class of bug.

**What the split was actually for** (kept in the rewrite): at `x == 0` the A dopant is
absent, so N dopants would label the same compound → collapse to one row with
`A_dopant=None`. At `x == 1` it is the *base* that is absent and each dopant is a genuinely
distinct compound → keep all. Asymmetric, and easy to lose in a naive `product()` rewrite.
Now one `product(x_vals, y_vals)` loop with per-endpoint collapse of the dopant choices.

**Second, latent bug in the same function:** `add_entry` tested the enclosing loop
variables `A` / `B` instead of its own `A_dopant` / `B_dopant` parameters. Benign only
because the `x != 0.0` guard happened to dominate at every call site, and because the
`for A in A_dopants` loops left A/B bound; it would have become a `NameError` or silent
mislabel under an empty dopant list or any reordering. Now uses the parameters.

Verified: user case 1 row; 9×9 unchanged at 81/0 dup; `include_undoped=False` 80;
2 A-dopants × 2 B-dopants 289 with no duplicates; interior-only and endpoints-only grids
both exact. End-to-end `gen_sqs.py --dry-run` reports 1 composition × 3 nv = 3 structures.

### EARS — Progress (2026-07-31, MD multi-task head)
<!-- concepts: deepmd-multitask-heads, lammps-deepmd-interface, dp-md-workflow-orchestration -->

**The MD stage needs the multi-task head too — but it cannot be passed the way opt/phonon
pass it.** `dp_opt.py` / `dp_phonon.py` select a branch at load time with
`DP(model=..., head=...)`. LAMMPS has no equivalent: checked the v3.0.0 pair style source
(`source/lmp/pair_deepmd.cpp`), and the keyword list is exactly `out_freq, out_file,
fparam, aparam, ttm, fparam_from_compute, aparam_from_compute, atomic, relative,
relative_v, virtual_len, spin_norm`. No `head`, no `branch`. The docs agree, but the
source is the check worth repeating on a version bump.

So for MD the head is a **build-time** property of the model file, not a runtime argument:

    dp --pt freeze -c model.ckpt.pt -o model.pth --head DOWNSTREAM_DATA

`md.model_head` is therefore recorded, not emitted — it goes into `input.lammps` as a
provenance comment naming the head and the exact freeze command. Emitting it as a
pair_style argument would have been the natural-looking change and would fail on the
compute node after the jobs are queued.

**Guard added because the failure is late and expensive:** `md.model_path` ending in
`.ckpt.pt` / `.ckpt` now aborts `gen_md.py` with the freeze command to run. A checkpoint
works fine for opt/phonon and silently *cannot* work for MD, so the same filename being
valid in two of three stages is exactly the trap. Cluster runs would only surface it
after the queue wait.

Deck for `model_head: null` is byte-identical to before, so existing trees are unaffected.

### EARS — Progress (2026-08-03 10:19)
<!-- concepts: dpdispatcher-config-layering, bohrium-job-sizing, secret-hygiene -->
Making `machine_type` / `image_name` / `group_size` per-stage from config.yaml.

**Decision: override in memory, never rewrite the JSONs.** machine.json holds the
Bohrium email+password. Templating per-stage copies of it would multiply the number
of files containing the credential by three. So config.yaml overrides are applied to
a dict loaded at submit time; `machine.json` stays the one copy of the secret and of
`batch_type`/`context_type`, which nobody wants to vary per stage anyway.

**Precedence chosen:** `dispatch.<stage>.<key>` > `dispatch.<key>` > the JSON's value.
Three levels rather than two because the common edit is "new image everywhere" (one
line at dispatch level) while the common *tuning* is per stage (phonon wants a
different machine type from md).

**Deliberately did not change defaults.** Tempting to ship `group_size: 8` for md and
`20` for opt, since group_size 1 means one Bohrium job per structure and the queue
wait dominates a 5-minute opt. But I have no measured runtimes here, and silently
regrouping someone's submission changes what lands in the queue. Shipped the current
values (group_size 1, existing image/machine_type) and exposed the knob with guidance.

**Route table over ad-hoc setattr:** `DISPATCH_OVERRIDES` maps key -> (which JSON, path
into it). Adding a knob is one line, and `--dry-run` can print effective values by
walking the same table, so the printout cannot drift from what is actually applied.

### EARS — Progress (2026-08-03, reorganization)
<!-- concepts: dpdispatcher-file-staging, workflow-layout-constraints, repo-hygiene -->
Separated live pipeline code from superseded code. Two findings worth keeping:

**The layout is not free — `forward_common_files` pins it.** dpdispatcher resolves
common files relative to `work_base` (= `output_root` = the drivers' directory), and the
task command reaches them with a fixed `../../../..` because every task dir sits at
`set_NN/<family>/<comp>/o####`. So `dp_opt.py`/`dp_phonon.py` cannot be tidied into a
`workers/` subdir without also editing the prefix, and the failure would land on the
compute node after the queue wait. Kept the top level flat and wrote the reason into
README rather than discovering it twice.

**Latent bug surfaced by the file sweep:** `build_command` emitted `--model
model.ckpt.pt`, which `DP()` resolves against cwd = the `o####` task dir. The model was
in neither `forward_files` nor `forward_common_files`, so stages 5/7 could only ever
have worked if the checkpoint were baked into the image at that exact relative name.
Moving `model.ckpt.pt` into `old_version_code/` made it visible. Fixed with
`stage_model()`: if the model exists at `output_root`, reference it via REL_PREFIX and
append it to common files; otherwise pass through untouched (the in-image case). Both
branches print what they chose, so `--dry-run` distinguishes them.

**Did not move anything out of `CLC_workflow/src`** — it is a separate git repo with a
GitHub remote, so a tidy-up there would register as deletions in that repo. Seven
modules are dormant for this pipeline but belong to the wider screening work; listed
them in README instead. Separately: its `.pyc` and `egg-info` are *tracked*, which is
why `git status` there is permanently dirty. Added `.gitignore`; untracking is left to
the human since it is their repo and their commit.

### EARS — Progress (2026-08-03 11:04)
<!-- concepts: dpdispatcher-job-environment, lammps-deck-portability, config-driven-defaults -->

MD jobs failed with `/opt/intel/oneapi/setvars.sh: No such file or directory` at line 12
of the generated `.sub`. That path came from `source_list` in resources.json — a leftover
from a VASP/Intel-MPI template. The legacy `old_version_code/job.json` never sourced it,
confirming it was never needed here; the DP image has no oneAPI.

Decisions:

- Repointed `source_list` at `/root/dp-outisli-env/bin/activate` (the venv the dpa4C
  toolchain lives in) and added `source_list` / `module_list` / `envs` to
  `DISPATCH_OVERRIDES`. Rule being applied consistently: **anything whose correct value
  is a function of `image_name` belongs next to `image_name` in config.yaml**, not in a
  JSON file that nobody rereads when the image changes. The `is not None` test in
  `resolve_dispatch` means `[]` is a real override ("emit nothing"), distinct from
  omitting the key.
- Replaced the hand-listed override keys in `_DEFAULTS["dispatch"]` with a loop over
  `DISPATCH_OVERRIDES`. Three copies of the key list (table, top level, three stages) had
  to be edited in lockstep to add one key; a missing entry fails silently as "override
  ignored", which is the worst kind — it only shows up as a job that ran on the wrong
  hardware.
- Deck is moving from `pair_style deepmd` + `plugin load libdeepmd_lmp.so` to
  `pair_style dpa4spin/kk` on a custom LAMMPS build launched under Kokkos. Made
  `pair_style` / `atom_style` / the plugin line config keys instead of hardcoding the new
  values, keeping the stock deepmd values as defaults — the two builds have to coexist.

Open: whether `dpa4spin/kk` needs `atom_style spin` (which would change conf.lmp's Atoms
section, the dump columns, and the trajectory parser) or tolerates `atomic`. Defaulting to
`atomic` and flagging it rather than guessing at the data-file format.

### EARS — Progress (2026-08-03 13:55)
<!-- concepts: dpdispatcher-file-staging, single-source-of-truth, lammps-deck-portability -->

MD failed on the node with `DeePMD-kit Error: Cannot open file: compressed_model.pt2`
(pair_dpa4spin.cpp:241). Same class of bug already fixed for opt/phonon: the model is
named as a bare relative path, and everything runs with cwd = the task's `o####` dir,
where nothing puts a model. `dispatch.md.forward_common_files` was `[]` with only a
comment suggesting the user add it by hand.

Fix: moved `REL_PREFIX` / `stage_model()` out of submit.py into clc_config.py and taught
gen_md.py to use it. What makes MD different from opt/phonon, and why sharing was
necessary rather than tidy:

- opt/phonon name the model **in the command**, which submit.py builds at submission
  time — one writer, so a local helper sufficed.
- MD names it **in input.lammps**, which gen_md.py writes at generation time, while
  submit.py separately decides what to upload. Two writers, two different moments. Any
  drift between them surfaces only as LAMMPS failing to open the file on the node, after
  the queue wait — never locally.

So the rule is: a path that one program writes into a file and another program has to
make true is a shared function, not a convention. Also dropped the
`not scfg.get("command")` guard for md when auto-staging — md always has a command
(`$LMP ...`), yet still needs its model staged, because the model isn't in the command.

### EARS — Session Start (2026-08-05 09:53)
<!-- concepts: md-trajectory-averaging, pipeline-artifact-naming, thermal-vs-static-distortion -->
- Task: stage 4 should hand DP optimisation the structure obtained by averaging atomic
  positions over the LAST 20 ps of traj.lammpstrj, not the single last MD frame.
- Why: a snapshot carries every atom's instantaneous thermal displacement at a random
  phase of its libration; the 20 ps mean structure keeps only the static distortion, so
  the relaxation starts from a representative geometry rather than a noisy one. The
  contrast is already quantified for these materials in
  `code/md/SrCaFeCoO/feofe_windows.py` (mean-structure vs instantaneous Fe-O-Fe angles).
- Decisions taken up front: new artifact name `POSCAR_md_avg` (never overwrite the
  last-frame `POSCAR_md_final`, so the two provenances cannot mix silently); atoms whose
  RMS spread over the window exceeds 0.5 A — vacancy-mediated O hops, which average two
  real sites into a fake midpoint — fall back per atom to their last-frame position.

### EARS — Progress (2026-08-05 09:57)
<!-- concepts: md-trajectory-averaging, tail-seek-trajectory-io, pipeline-artifact-naming -->

Implemented `read_frames_since` + `average_frames` in `lammps_io.py` and rewired stage 4.
Three things worth recording:

- `read_last_frame`'s "decode `blob[off:]` for each candidate offset" is fine because it
  returns on the first success, but reused as-is for a whole window it is quadratic:
  100 frames x an 8 MB tail decoded per frame. `_parse_window` slices each frame between
  consecutive offsets instead, so the tail is decoded once.
- The growing-tail loop does not blindly quadruple. After the first read both the byte
  spacing and the STEP spacing between frames are known, so the next window is sized to
  the number of frames actually wanted -- two reads instead of log4(fraction of file).
- Averaging must be done in FRACTIONAL coordinates against a per-frame cell: `fix npt`
  breathes the box, so a Cartesian average mixes coordinates measured against different
  boxes. The mean cell is averaged alongside and the mean fractional positions expressed
  in it. Minimum image (`ds -= round(ds)`) against the window's first frame keeps an atom
  that crosses a boundary from smearing across the cell. All three points are inherited
  from `code/md/SrCaFeCoO/feofe_windows.py`, which is the independent implementation the
  new code will be cross-checked against.

Artifact renamed `POSCAR_md_final` -> `POSCAR_md_avg` rather than overwritten. The reason
is not tidiness: stage 4 skips a directory whose output already exists, so keeping the
name would leave a tree silently half last-frame and half averaged, with nothing on disk
to tell the two apart. Read-only consumers (`collect_results.py`, `thermo.py`) still
accept the old name so existing result trees keep parsing.

### EARS — Progress (2026-08-05 12:11)
<!-- concepts: octahedral-tilt-analysis, structure-validation, periodic-neighbour-search -->

Added `bob_angles.py` to the package: B-O-B angle distributions from a POSCAR, to check
`POSCAR_md_avg` against `optimized_POSCAR` (driver `check_bob.py`, stage 6.5). Ported from
`code/md/SrCaFeCoO/bob_angle_poscar.py`, with three changes forced by the pipeline:

- The B site cannot be hard-coded `(Fe, Co)`. config.yaml drives several families, so the
  B species come per structure from the manifest's `b_base`/`b_dopant` columns.
- The per-O loop over all B atoms x 27 shifts is O(nB) per O; at 2560 atoms that is ~1500 O
  x 500 B x 27. Replaced by one cKDTree over the 27-image B sublattice, queried k=12 with
  `distance_upper_bound=rcut` — the same trick `phonon_analysis.min_pair_violation` uses.
- KD-tree images must be collapsed to their parent atom before taking "the two nearest B",
  or a cell small enough for one B to bridge to itself would report a fake 2-neighbour O.

An O with <2 B neighbours yields no angle. The old script dropped those silently; here they
are counted (`n_underco`), because that is exactly the signature of a relaxation that pulled
an octahedron apart — the failure this check exists to catch.

### EARS — Progress (2026-08-05 16:18)
<!-- concepts: sqs-composition-encoding, cation-vacancy-modelling, supercell-commensurability -->

Extending stage 1 to cation (A-site / B-site) vacancies, for a target list whose
compositions are off-stoichiometric in A/B (A/B = 0.80-1.05), not just in oxygen.

The blocker was never the config: `generate_one_sqs` hard-codes three sublattices, and the
only one that can hold an `X` is the anion — so `vacancies:` is an OXYGEN vacancy count and
nothing in config.yaml can put a vacancy on A or B. Writing `Sr0.75Ca0.2...` in the old
schema silently renormalises to full occupancy, i.e. generates the wrong compound with no
error. That silence is the reason this needs a real feature rather than a fudged x value.

Two decisions:

- **x stays the fraction of SITES, not of occupied cations.** With `a_vac`, the base gets
  `1 - x - a_vac`. That reads straight off the formula (Sr0.75Ca0.2 -> x=0.2, a_vac=0.05,
  Sr=0.75) and keeps the vacancy-free case identical to what it is today.
- **Commensurability is checked, not left to mcsqs.** ATAT is told fractions; if they are
  not realisable on the lattice it rounds, which is the same silent-wrong-compound failure.
  A hard check that every species count lands on an integer site count turns it into an
  error at task-build time, before any GPU hour is spent.

Cell size follows from the arithmetic, not from taste: 0.05 steps need `n_A` divisible by
20, and the 0.7125 of (Sr0.75Ba0.25)0.95 needs 80 -> 80 A sites -> 400 atoms. The 40-site
base can only reach that as [5,2,1] (38.6 x 15.4 x 7.7 A), whose short axis is below the
5.58 A pair cutoff corrdump is given. A 5-atom primitive base reaches the same 400 atoms as
[5,4,4] = 19.3 x 15.4 x 15.4 A instead, which is why `supercell_site_counts` had to stop
hard-coding 8/8/24 per cell and read the base POSCAR.

### EARS — Progress (2026-08-05 17:40)
<!-- concepts: sqs-generation-robustness, silent-data-corruption, atat-mcsqs-behaviour -->

7 of ~2050 structures failed with "POSCAR conversion failed". Diagnosed, and the visible
failures turned out to be the *lucky* ones.

mcsqs rewrites `bestsqs.out` on every improvement, and `run_mcsqs_rc` kills it on timeout,
so a kill landing on top of a write truncates the file. What happens next depends on WHERE
it was truncated, and the two cases behave completely differently:

- truncated in the 6-line header  -> pymatgen raises IndexError -> "POSCAR conversion
  failed", the message actually seen;
- truncated anywhere in the atom block -> **pymatgen parses it happily** and returns a
  structure with fewer atoms, which becomes a valid-looking POSCAR of the wrong
  composition, status `sqs`, and goes straight into 200 ps of MD.

The atom block is 98% of the file, so the silent mode is the more likely one. 7 header hits
implies the tree may hold silently-short POSCARs among the "successes". Verified locally:
cutting the last line of a bestsqs.out gives `PARSED, 9 atoms` instead of 10, no exception.

Fixes: `convert_bestsqs_to_poscar` now takes `expect_atoms` and returns (ok, detail) rather
than swallowing the exception; the expected count is checked on every POSCAR written *and*
on every POSCAR skipped, so a resumable rerun re-audits the whole tree; and the generator
retries with a fresh seed, deleting the stale `bestsqs.out` first -- without that the retry
converts the same corrupt file and is a no-op.

Deliberately NOT done: falling back to a random decoration when mcsqs fails. That swaps a
correct-but-interrupted search for a structure with different short-range order, in a tree
where nothing downstream could tell the two apart. The failures are a write race, not a
composition mcsqs cannot handle.

### EARS — Session Start (2026-08-18 14:30)
<!-- concepts: config-schema, perovskite-composition-families -->
- Task: Show how to express a set of pasted composition equations as `families:` entries in config.yaml (read-only advice; do not edit config.yaml).
- Why: User wants to extend the SQS generation sweep to new A/B dopant systems without breaking the existing set_* tree.

### EARS — Session Start (2026-08-25 11:11)
<!-- concepts: python-packaging, cli-entry-points, workflow-portability -->
- Task: Make the CLC workflow runnable from any working directory without copying CLC_workflow/ + driver scripts into each case dir.
- Why: User runs many independent cases; per-case duplication of code makes updates and provenance painful.

### EARS — Decision: run-from-anywhere via a real package (2026-08-25)
<!-- concepts: python-packaging, cli-entry-points, workflow-portability -->
- **Surprise**: the workflow was already 90% location-independent. `load_config` resolves
  `output_root` / `machine_json` / `resources_json` / `base_poscar` against *config.yaml's
  own directory*, and no driver writes cwd-relative output. The copying was never required
  by path resolution — only by four concrete things.
- **The four**: (1) `_clc_src()` duplicated in 9 drivers walked up from `__file__` to find
  `CLC_workflow/src`; (2) `run_all.sh` hardcoded `$HERE/*.py` and did `cd "$HERE"`;
  (3) dpdispatcher resolves `forward_common_files` against work_base, so `dp_opt.py` /
  `dp_phonon.py` had to sit *at* `output_root`; (4) `stage_model` errors if the checkpoint
  resolves outside `output_root`, so every case duplicated an 8 MB `.ckpt.pt`.
- **`setup.py` had never worked**: `find_packages(where="src")` over a directory of *flat*
  modules finds zero packages — `clc_workflow.egg-info/top_level.txt` was empty, so
  `pip install` installed nothing. Nobody noticed because everything ran by path.
- **Fix**: `src/*.py` → `src/clc_workflow/`, drivers → `clc_workflow/cli/` behind one
  `clc` console script, workers/templates/run_all.sh as package data. Only **3**
  intra-package import lines existed, so the move was near-free.
- **Pitfall hit**: the flat-import rewrite regex covered the 16 library modules but not
  *sibling CLI* modules — `convergence_x0.py` did `import plot_heatmap_avg_series_bolz_linear`
  after a `sys.path.insert(HERE)`. It compiled fine and only failed on import. Byte-compiling
  is not enough after a package move; import every module.
- **Pitfall hit**: stripping `# noqa: E402` with `[ \t]*# noqa: E402\s*$` — `\s*` ate the
  following newlines and glued `def` onto the import block. Use `[^\S\n]*` when trimming
  to end-of-line.
- **Staging**: `clc submit` now hardlinks workers (from the package) and the checkpoint
  (from `model_store` / `$CLC_MODEL_DIR` / `~/.clc/models`) into `output_root`, copying
  only across filesystems. Verified same inode, 2 links. Side benefit the manual copy
  never had: the node always runs the *installed* worker.
