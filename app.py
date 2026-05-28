import argparse
import os
import random
import sys
import time

from models import Parser
from models.hybrid_solver import HybridSolver
from models.initial_solution import InitialSolution
from models.solution import Solution
from parameter_sets import DEFAULT_PARAMETER_SET_NAME, available_parameter_sets, get_parameter_set


DEFAULT_RANDOM_SEED = 54


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid solver for Google Hash Code 2020 Book Scanning instances."
    )
    parser.add_argument("--input", "--file", dest="input_path")
    parser.add_argument("--output", dest="output_path")
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--parameter-set",
        default=DEFAULT_PARAMETER_SET_NAME,
        choices=available_parameter_sets(),
    )
    parser.add_argument("--time-limit", type=float)
    parser.add_argument(
        "--hybrid-mode",
        metavar="MODE",
        help=(
            "ga_with_parallel_ils runs ILS workers inside GA; "
            "ga_with_ils runs one inline ILS candidate inside GA; "
            "ga_then_ils runs GA first, then ILS. "
            "Hyphenated forms and old names embedded_parallel, sequential, and after_gens are accepted."
        ),
    )
    parser.add_argument("--ga-ratio", type=float)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--seed-solution", action="append", default=[])
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="use only --seed-solution files and skip generated initial solutions",
    )
    parser.add_argument("--init-max-time", type=float)
    parser.add_argument("--grasp-max-time", type=float)
    parser.add_argument("--refine-initial-count", type=int)
    parser.add_argument("--log-dir")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--no-improvement", action="store_true")
    parser.add_argument(
        "--ils-every-generations",
        "--embedded-improvement-interval",
        dest="ils_every_generations",
        type=int,
    )
    parser.add_argument(
        "--ils-candidates",
        "--embedded-improvement-count",
        dest="ils_candidates",
        type=int,
    )
    parser.add_argument(
        "--ils-time-limit",
        "--embedded-improvement-time",
        dest="ils_time_limit",
        type=float,
    )
    parser.add_argument(
        "--ils-reserve-time",
        "--embedded-improvement-min-remaining",
        dest="ils_reserve_time",
        type=float,
    )
    parser.add_argument(
        "--ils-workers",
        "--embedded-improvement-workers",
        dest="ils_workers",
        type=int,
    )
    parser.add_argument(
        "--ils-merge-policy",
        "--embedded-improvement-insert-policy",
        dest="ils_merge_policy",
        choices=["all", "best"],
    )

    add_ga_args(parser)
    add_improvement_args(parser)
    return parser.parse_args()


def add_ga_args(parser):
    group = parser.add_argument_group("genetic parameters")
    group.add_argument("--population-size", type=int)
    group.add_argument("--generations", type=int)
    group.add_argument("--mutation-steps", type=int)
    group.add_argument("--mutation-prob", type=float)
    group.add_argument("--crossover-rate", type=float)
    group.add_argument("--immigrant-frac", type=float)
    group.add_argument("--local-refine-steps", type=int)
    group.add_argument("--elite-refine-interval", type=int)
    group.add_argument("--elite-refine-count", type=int)
    group.add_argument("--selection-tournament-size", type=int)


def add_improvement_args(parser):
    group = parser.add_argument_group("improvement parameters")
    group.add_argument("--variant", default=None)
    group.add_argument("--accept-worse-prob", type=float)
    group.add_argument("--restart-threshold", type=int)
    group.add_argument("--perturb-strength-base", type=int)
    group.add_argument("--perturb-strength-growth", type=int)
    group.add_argument("--local-no-improve-limit", type=int)
    group.add_argument("--ls-order-weight", type=float)
    group.add_argument("--ls-insert-weight", type=float)
    group.add_argument("--ls-strategic-weight", type=float)
    group.add_argument("--perturb-replace-bias", type=float)
    group.add_argument("--restart-fresh-probability", type=float)


def main():
    args = parse_args()
    config = get_parameter_set(args.parameter_set)
    apply_cli_overrides(config, args)

    if args.seed is not None:
        random.seed(args.seed)

    if args.input_path:
        output_path = args.output_path or default_output_path(
            args.input_path,
            args.input_dir,
            args.output_dir,
        )
        result = solve_instance(args.input_path, output_path, config, args)
        print(f"Final score for {os.path.basename(args.input_path)}: {result.solution.fitness_score:,}")
        return

    input_files = discover_input_files(args.input_dir)
    if not input_files:
        print(f"No .txt files found in {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for input_path in input_files:
        relative_path = os.path.relpath(input_path, args.input_dir)
        output_path = os.path.join(args.output_dir, relative_path)
        result = solve_instance(input_path, output_path, config, args)
        results.append((relative_path, result.solution.fitness_score))

    print_summary(results)
    if args.validate:
        validate_directory(args.input_dir, args.output_dir)


def discover_input_files(input_dir):
    input_files = []
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()
        for name in sorted(files):
            if name.endswith(".txt"):
                input_files.append(os.path.join(root, name))
    return input_files


def default_output_path(input_path, input_dir, output_dir):
    input_abs = os.path.abspath(input_path)
    input_dir_abs = os.path.abspath(input_dir)
    try:
        common_path = os.path.commonpath([input_abs, input_dir_abs])
    except ValueError:
        common_path = ""

    if os.path.normcase(common_path) == os.path.normcase(input_dir_abs):
        return os.path.join(output_dir, os.path.relpath(input_abs, input_dir_abs))
    return os.path.join(output_dir, os.path.basename(input_path))


def solve_instance(input_path, output_path, config, args):
    started_at = time.time()
    verbose = not args.quiet
    if verbose:
        print(f"Computing {input_path}")
        print(f"Time limit: {config['hybrid']['time_limit']:.1f}s")

    parser = Parser(input_path)
    instance = parser.parse()
    seed_solutions = build_seed_solutions(
        instance,
        config,
        args,
        verbose,
    )

    solver = HybridSolver(
        instance=instance,
        seed_solutions=seed_solutions,
        time_limit_sec=config["hybrid"]["time_limit"],
        ga_ratio=config["hybrid"]["ga_ratio"],
        ga_params=config["ga"],
        improvement_params=config["improvement"],
        seed=args.seed,
        verbose=verbose,
        improvement_enabled=config["hybrid"]["improvement_enabled"],
        hybrid_mode=config["hybrid"]["hybrid_mode"],
        ils_every_generations=hybrid_value(
            config["hybrid"],
            "ils_every_generations",
            "embedded_improvement_interval",
        ),
        ils_candidates=hybrid_value(
            config["hybrid"],
            "ils_candidates",
            "embedded_improvement_count",
        ),
        ils_time_limit=hybrid_value(
            config["hybrid"],
            "ils_time_limit",
            "embedded_improvement_time",
        ),
        ils_reserve_time=hybrid_value(
            config["hybrid"],
            "ils_reserve_time",
            "embedded_improvement_min_remaining",
        ),
        ils_workers=hybrid_value(
            config["hybrid"],
            "ils_workers",
            "embedded_improvement_workers",
        ),
        ils_merge_policy=hybrid_value(
            config["hybrid"],
            "ils_merge_policy",
            "embedded_improvement_insert_policy",
        ),
    )
    result = solver.solve(
        instance_name=os.path.basename(input_path),
        log_dir=args.log_dir,
    )
    result.solution.export(output_path)

    elapsed = time.time() - started_at
    if verbose:
        print(f"Output written to: {output_path}")
        print(f"Elapsed time: {elapsed:.1f}s")
    if args.validate:
        validate_single(input_path, output_path)

    return result


def build_seed_solutions(instance, config, args, verbose):
    hybrid = config["hybrid"]
    improvement = config["improvement"]
    seed_solutions = []

    for seed_path in args.seed_solution:
        seed = load_solution_seed(seed_path, instance)
        seed_solutions.append(seed)
        if verbose:
            print(f"Loaded seed solution: {seed_path} | score {seed.fitness_score:,}")

    if not args.seed_only:
        generated_solutions = InitialSolution.generate_initial_solutions(
            instance,
            max_time=hybrid["init_max_time"],
            alphas=improvement.get("alphas"),
            beta=improvement.get("weighted_beta", 0.12),
            grasp_rcl=improvement.get("grasp_rcl", 0.05),
            grasp_max_time=hybrid["grasp_max_time"],
            verbose=verbose,
            refine_count=hybrid["refine_initial_count"],
        )
        seed_solutions.extend(generated_solutions)

    if not seed_solutions:
        raise ValueError("--seed-only requires at least one --seed-solution file")

    return unique_solutions(seed_solutions)


def load_solution_seed(output_path, instance):
    with open(output_path, "r", encoding="utf-8") as file:
        first_line = file.readline().strip()
        if not first_line:
            raise ValueError(f"Empty seed solution file: {output_path}")
        library_count = int(first_line)
        signed = []
        for _ in range(library_count):
            header = file.readline().strip()
            books_line = file.readline()
            if not header:
                break
            lib_id = int(header.split()[0])
            signed.append(lib_id)
            if books_line == "":
                break

    seen = set(signed)
    order = signed + [lib_id for lib_id in range(instance.num_libs) if lib_id not in seen]
    return Solution.from_order(order, instance)


def unique_solutions(solutions):
    unique = []
    seen = set()
    for solution in sorted(solutions, key=lambda item: item.fitness_score, reverse=True):
        signature = tuple(solution.signed_libraries)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(solution)
    return unique


def apply_cli_overrides(config, args):
    hybrid = config["hybrid"]
    ga = config["ga"]
    improvement = config["improvement"]

    override_if_set(hybrid, "time_limit", args.time_limit)
    override_if_set(hybrid, "hybrid_mode", args.hybrid_mode)
    override_if_set(hybrid, "ga_ratio", args.ga_ratio)
    override_if_set(hybrid, "init_max_time", args.init_max_time)
    override_if_set(hybrid, "grasp_max_time", args.grasp_max_time)
    override_if_set(hybrid, "refine_initial_count", args.refine_initial_count)
    override_if_set(
        hybrid,
        "ils_every_generations",
        args.ils_every_generations,
    )
    override_if_set(
        hybrid,
        "ils_candidates",
        args.ils_candidates,
    )
    override_if_set(
        hybrid,
        "ils_time_limit",
        args.ils_time_limit,
    )
    override_if_set(
        hybrid,
        "ils_reserve_time",
        args.ils_reserve_time,
    )
    override_if_set(
        hybrid,
        "ils_workers",
        args.ils_workers,
    )
    override_if_set(
        hybrid,
        "ils_merge_policy",
        args.ils_merge_policy,
    )
    if args.no_improvement:
        hybrid["improvement_enabled"] = False

    for cli_name, key in {
        "population_size": "population_size",
        "generations": "generations",
        "mutation_steps": "mutation_steps",
        "mutation_prob": "mutation_prob",
        "crossover_rate": "crossover_rate",
        "immigrant_frac": "immigrant_frac",
        "local_refine_steps": "local_refine_steps",
        "elite_refine_interval": "elite_refine_interval",
        "elite_refine_count": "elite_refine_count",
        "selection_tournament_size": "selection_tournament_size",
    }.items():
        override_if_set(ga, key, getattr(args, cli_name))

    for cli_name, key in {
        "variant": "variant",
        "accept_worse_prob": "accept_worse_prob",
        "restart_threshold": "restart_threshold",
        "perturb_strength_base": "perturb_strength_base",
        "perturb_strength_growth": "perturb_strength_growth",
        "local_no_improve_limit": "local_no_improve_limit",
        "ls_order_weight": "ls_order_weight",
        "ls_insert_weight": "ls_insert_weight",
        "ls_strategic_weight": "ls_strategic_weight",
        "perturb_replace_bias": "perturb_replace_bias",
        "restart_fresh_probability": "restart_fresh_probability",
    }.items():
        override_if_set(improvement, key, getattr(args, cli_name))


def override_if_set(target, key, value):
    if value is not None:
        target[key] = value


def hybrid_value(hybrid, new_key, old_key):
    if new_key in hybrid:
        return hybrid[new_key]
    return hybrid[old_key]


def validate_single(input_path, output_path):
    from validator.validator import validate_solution

    result = validate_solution(input_path, output_path, isConsoleApplication=True)
    print(f"Validation: {result}")


def validate_directory(input_dir, output_dir):
    from validator.multiple_validator import validate_all_solutions

    validate_all_solutions(input_dir=input_dir, output_dir=output_dir)


def print_summary(results):
    print("\nSummary")
    print("-" * 50)
    for name, score in results:
        print(f"{name:<40} {score:>15,}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
