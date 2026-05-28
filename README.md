# Hybrid Book Scanning Solver

High-performance hybrid solver for Google Hash Code 2020 Book Scanning instances. The default mode runs the current GA while periodically improving elite GA candidates with parallel ILS workers.

## Installation

```bash
pip install -r requirements.txt
```

`numba` is used for accelerated scoring. The first run in a fresh Python process may spend extra time compiling scoring kernels; longer runs amortize that cost.

## Default Mode

The default parameter set is `default`, which uses embedded parallel ILS:

- GA runs for the full search budget.
- ILS is called inside GA generations.
- ILS runs every generation.
- 6 elite candidates are improved with 6 workers.
- The best improved candidate is inserted back into the GA population.

## Run One Instance

```bash
python app.py --input input/google_hashcode/e_so_many_books.txt --output output/google_hashcode/e_so_many_books.txt --time-limit 600
```

Equivalent explicit command:

```bash
python app.py \
  --parameter-set default \
  --input input/google_hashcode/e_so_many_books.txt \
  --output output/google_hashcode/e_so_many_books.txt \
  --time-limit 600
```

Useful single-instance options:

```bash
python app.py \
  --input input/google_hashcode/e_so_many_books.txt \
  --output output/google_hashcode/e_so_many_books.txt \
  --time-limit 600 \
  --ils-workers 6 \
  --ils-candidates 6 \
  --seed 1 \
  --log-dir logs \
  --validate
```

## Run Batch

Run every `.txt` file under `input`, including category folders, and write matching outputs into `output` with the same relative paths:

```bash
python app.py --input-dir input --output-dir output --time-limit 600
```

Use a separate output folder for a named experiment:

```bash
python app.py \
  --input-dir input \
  --output-dir output/default_parallel \
  --time-limit 600 \
  --seed 1 \
  --log-dir logs/default_parallel
```

Batch validation:

```bash
python app.py --input-dir input --output-dir output/default_parallel --time-limit 600 --validate
```

## Hybrid Modes

### Default: GA With Parallel ILS

Parallel ILS is embedded inside GA generations.

```bash
python app.py --input input/google_hashcode/e_so_many_books.txt --time-limit 600 --hybrid-mode ga_with_parallel_ils
```

This is the default mode. Increase or reduce workers based on available CPU and memory:

```bash
python app.py \
  --input input/google_hashcode/e_so_many_books.txt \
  --time-limit 600 \
  --ils-workers 8 \
  --ils-candidates 8
```

### GA With Inline ILS

Runs the same embedded idea, but with one ILS worker. GA and ILS are still interleaved, just not parallel.

```bash
python app.py --input input/google_hashcode/e_so_many_books.txt --time-limit 600 --hybrid-mode ga_with_ils
```

### GA Then ILS

Runs GA first, then starts ILS from the best GA solution.

```bash
python app.py --input input/google_hashcode/e_so_many_books.txt --time-limit 600 --hybrid-mode ga_then_ils
```

You can tune the GA share in this mode:

```bash
python app.py --input input/google_hashcode/e_so_many_books.txt --time-limit 600 --hybrid-mode ga_then_ils --ga-ratio 0.55
```

## Seed From Existing Output

An existing output file can be used as an additional GA seed. The solver rebuilds the schedule under the current evaluator before using it.

```bash
python app.py \
  --input input/google_hashcode/e_so_many_books.txt \
  --output output/google_hashcode/e_so_many_books.txt \
  --seed-solution output/previous/google_hashcode/e_so_many_books.txt
```

Use only provided output seeds and skip generated constructors:

```bash
python app.py \
  --input input/google_hashcode/e_so_many_books.txt \
  --output output/google_hashcode/e_so_many_books.txt \
  --seed-solution output/previous/google_hashcode/e_so_many_books.txt \
  --seed-only
```

## Parameter Sets

Named parameter sets are defined in `parameter_sets.py`. Each set has three layers:

- `hybrid`: mode, time split, ILS worker settings, and initial-construction budget.
- `ga`: population, generation, crossover, mutation, selection, and GA operator weights.
- `improvement`: ILS acceptance, restart, perturbation, local-search, and constructor parameters.

The GA and ILS operator pools both include the conditional uniform-instance operators `coverage_exchange` and `paired_choice_flip`. They are not part of the GA tuning config, are automatically disabled on non-uniform instances, and use the same ILS implementation and instance-adjusted weighting in local refinement. Bulk offspring mutation skips them so regular generation progress is not blocked.

```bash
python app.py --parameter-set default --input input/google_hashcode/e_so_many_books.txt
python app.py --parameter-set ga_with_parallel_ils --input input/google_hashcode/e_so_many_books.txt
python app.py --parameter-set ga_with_ils --input input/google_hashcode/e_so_many_books.txt
python app.py --parameter-set ga_then_ils --input input/google_hashcode/e_so_many_books.txt
```

Common tunables:

- `--time-limit`: search budget after initial construction.
- `--init-max-time`: construction budget for the initial GA seed pool.
- `--seed`: defaults to `54`, matching the ILS experiments.
- `--hybrid-mode`: `ga_with_parallel_ils`, `ga_with_ils`, or `ga_then_ils`.
- `--ils-workers`: parallel ILS worker count in `ga_with_parallel_ils` mode.
- `--ils-candidates`: number of elite GA candidates sent to ILS.
- `--ils-every-generations`: generation interval for ILS calls inside GA.
- `--ils-time-limit`: time budget per ILS call.
- `--ils-reserve-time`: reserve time before stopping ILS calls inside GA.
- `--ils-merge-policy`: `best` inserts only the best improved candidate; `all` inserts every improved elite.
- `--population-size`, `--generations`, `--mutation-prob`, `--crossover-rate`: GA controls.

Compatibility aliases are still accepted: `embedded_parallel`, `sequential`, `after_gens`, and the old `--embedded-improvement-*` flags.

## Validation

Console validation works without the GUI dependency:

```bash
python validator/validator.py input/google_hashcode/e_so_many_books.txt output/google_hashcode/e_so_many_books.txt
```
