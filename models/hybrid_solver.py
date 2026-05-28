import csv
import json
import os
import random
import traceback
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Dict, Sequence

from models.genetic_solver import GeneticSolver
from models.instance_data import InstanceData
from models.solution import Solution
from models.solver import Solver as ImprovementSolver


_WORKER_INSTANCE = None


def _init_embedded_worker(data):
    global _WORKER_INSTANCE
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER_INSTANCE = data


def _embedded_improvement_worker(payload):
    solution, params, time_slice, seed, generation, rank = payload
    started_at = time.time()
    input_score = solution.fitness_score
    try:
        solver = ImprovementSolver(seed=seed, verbose=False)
        candidate = solver.iterated_local_search(
            _WORKER_INSTANCE,
            instance_name=None,
            time_limit=time_slice,
            seed_solution=solution,
            seed_label=f"ga_gen_{generation}_elite_{rank + 1}",
            **params,
        )
        return rank, candidate, input_score, time.time() - started_at, None
    except Exception:
        return (
            rank,
            None,
            input_score,
            time.time() - started_at,
            traceback.format_exc(),
        )


GA_PARAM_KEYS = {
    "population_size",
    "generations",
    "mutation_prob",
    "crossover_rate",
    "immigrant_frac",
    "steady_state_ratio",
    "mutation_steps",
    "mutation_accept_worse_prob",
    "steady_state_offspring_factor",
    "local_refine_steps",
    "elite_refine_interval",
    "elite_refine_count",
    "initial_mutation_ratio",
    "mutation_screen_attempts",
    "mutation_exact_checks",
    "local_screen_attempts",
    "local_exact_checks",
    "operator_weights",
    "selection_weights",
    "selection_tournament_size",
}


IMPROVEMENT_PARAM_KEYS = {
    "pool_size",
    "init_max_time",
    "init_budget_ratio",
    "restart_init_budget_ratio",
    "restart_threshold",
    "perturb_strength_base",
    "perturb_strength_growth",
    "accept_worse_prob",
    "sa_final_temperature_ratio",
    "alpha_values",
    "weighted_beta",
    "grasp_rcl",
    "grasp_max_time",
    "local_no_improve_limit",
    "ls_order_weight",
    "ls_insert_weight",
    "ls_strategic_weight",
    "operator_weights",
    "operators",
    "enable_initial_local_search",
    "perturb_replace_bias",
    "restart_fresh_probability",
    "variant",
}


@dataclass
class HybridResult:
    solution: Solution
    initial_solution: Solution
    genetic_solution: Solution
    improved_solution: Solution
    metrics: Dict


class HybridSolver:
    def __init__(
        self,
        instance: InstanceData,
        seed_solutions: Sequence[Solution],
        time_limit_sec=600.0,
        ga_ratio=0.65,
        ga_params=None,
        improvement_params=None,
        seed=None,
        verbose=True,
        improvement_enabled=True,
        hybrid_mode="ga_with_parallel_ils",
        embedded_improvement_interval=5,
        embedded_improvement_count=1,
        embedded_improvement_time=1.0,
        embedded_improvement_min_remaining=2.0,
        embedded_improvement_workers=1,
        embedded_improvement_insert_policy="all",
        ils_every_generations=None,
        ils_candidates=None,
        ils_time_limit=None,
        ils_reserve_time=None,
        ils_workers=None,
        ils_merge_policy=None,
    ):
        if ils_every_generations is not None:
            embedded_improvement_interval = ils_every_generations
        if ils_candidates is not None:
            embedded_improvement_count = ils_candidates
        if ils_time_limit is not None:
            embedded_improvement_time = ils_time_limit
        if ils_reserve_time is not None:
            embedded_improvement_min_remaining = ils_reserve_time
        if ils_workers is not None:
            embedded_improvement_workers = ils_workers
        if ils_merge_policy is not None:
            embedded_improvement_insert_policy = ils_merge_policy

        self.instance = instance
        self.seed_solutions = self._prepare_seed_solutions(seed_solutions)
        if not self.seed_solutions:
            raise ValueError("HybridSolver requires at least one seed solution")
        self.time_limit_sec = max(0.1, float(time_limit_sec))
        self.ga_ratio = min(1.0, max(0.0, float(ga_ratio)))
        self.ga_params = dict(ga_params or {})
        self.improvement_params = dict(improvement_params or {})
        self.seed = seed
        self.verbose = verbose
        self.improvement_enabled = improvement_enabled
        self.hybrid_mode = self._normalize_hybrid_mode(hybrid_mode)
        self.embedded_improvement_interval = max(0, int(embedded_improvement_interval))
        self.embedded_improvement_count = max(1, int(embedded_improvement_count))
        self.embedded_improvement_time = max(0.05, float(embedded_improvement_time))
        self.embedded_improvement_min_remaining = max(
            0.0,
            float(embedded_improvement_min_remaining),
        )
        self.embedded_improvement_workers = max(1, int(embedded_improvement_workers))
        if self.hybrid_mode == "ga_with_ils":
            self.embedded_improvement_workers = 1
        self.embedded_improvement_insert_policy = (
            "best" if embedded_improvement_insert_policy == "best" else "all"
        )
        self.embedded_improvement_metrics = self._new_embedded_metrics()
        self._embedded_executor = None
        self._embedded_executor_workers = 0
        self.metrics = {}

    def solve(self, instance_name=None, log_dir=None):
        if self.seed is not None:
            random.seed(self.seed)

        started_at = time.time()
        deadline = started_at + self.time_limit_sec
        initial_solution = self.seed_solutions[0].clone()
        embedded_mode = (
            self.hybrid_mode in {"ga_with_parallel_ils", "ga_with_ils"}
            and self.improvement_enabled
        )
        genetic_budget = (
            self.time_limit_sec
            if embedded_mode
            else max(0.1, self.time_limit_sec * self.ga_ratio)
        )
        genetic_budget = min(genetic_budget, max(0.1, deadline - time.time()))

        if self.verbose:
            print("\nHybrid Search")
            print("-" * 50)
            print(f"Initial best seed: {initial_solution.fitness_score:,}")
            print(f"GA seed pool: {len(self.seed_solutions)} solution(s)")
            if embedded_mode:
                label = (
                    "GA with parallel ILS"
                    if self.hybrid_mode == "ga_with_parallel_ils"
                    else "GA with inline ILS"
                )
                print(
                    f"Mode: {label} | GA budget: {genetic_budget:.1f}s | "
                    f"ILS every {self.embedded_improvement_interval} generations | "
                    f"workers: {self.embedded_improvement_workers}"
                )
            else:
                print(
                    f"Genetic phase budget: {genetic_budget:.1f}s | "
                    f"Improvement budget target: {max(0.0, self.time_limit_sec - genetic_budget):.1f}s"
                )

        genetic_solver = self._build_genetic_solver(
            initial_solution,
            genetic_budget,
            embedded_mode=embedded_mode,
        )
        genetic_started_at = time.time()
        try:
            genetic_solution = genetic_solver.solve()
        finally:
            self._shutdown_embedded_executor()
        genetic_elapsed = time.time() - genetic_started_at
        best_solution = genetic_solution.clone()
        self._write_genetic_logs(log_dir, instance_name, genetic_solver)

        if self.verbose:
            print("\nGA phase result")
            print("-" * 50)
            print(f"Start score: {initial_solution.fitness_score:,}")
            print(f"GA best: {genetic_solution.fitness_score:,}")
            print(
                f"GA improvement: "
                f"{self._format_gain(genetic_solution.fitness_score, initial_solution.fitness_score)}"
            )
            print(f"GA elapsed: {genetic_elapsed:.1f}s")
            if embedded_mode:
                self._print_embedded_result()

        improved_solution = None
        improvement_elapsed = 0.0
        remaining = max(0.0, deadline - time.time())
        if not embedded_mode and self.improvement_enabled and remaining >= 0.2:
            if self.verbose:
                print("\nILS phase")
                print("-" * 50)
                print(
                    f"Starting from genetic best: {genetic_solution.fitness_score:,} | "
                    f"budget: {remaining:.1f}s"
                )

            improvement_started_at = time.time()
            improved_solution = self._run_improvement_phase(
                genetic_solution,
                remaining,
                instance_name=instance_name,
                log_dir=log_dir,
            )
            improvement_elapsed = time.time() - improvement_started_at
            if improved_solution.fitness_score > best_solution.fitness_score:
                best_solution = improved_solution.clone()
            if self.verbose:
                print("\nILS phase result")
                print("-" * 50)
                print(f"GA best: {genetic_solution.fitness_score:,}")
                print(f"ILS best: {improved_solution.fitness_score:,}")
                print(
                    f"ILS improvement: "
                    f"{self._format_gain(improved_solution.fitness_score, genetic_solution.fitness_score)}"
                )
                print(f"ILS elapsed: {improvement_elapsed:.1f}s")
        elif self.verbose and not embedded_mode:
            print("\nILS phase skipped: no remaining time")

        elapsed = time.time() - started_at
        self.metrics = {
            "initial_score": int(initial_solution.fitness_score),
            "seed_pool_size": len(self.seed_solutions),
            "genetic_score": int(genetic_solution.fitness_score),
            "improved_score": int(improved_solution.fitness_score)
            if improved_solution is not None
            else None,
            "final_score": int(best_solution.fitness_score),
            "elapsed_sec": round(elapsed, 6),
            "time_limit_sec": self.time_limit_sec,
            "ga_ratio": self.ga_ratio,
            "hybrid_mode": self.hybrid_mode,
            "phase_elapsed_sec": {
                "ga": round(genetic_elapsed, 6),
                "ils": round(improvement_elapsed, 6),
            },
            "phase_gains": {
                "ga_vs_initial": self._gain_dict(
                    genetic_solution.fitness_score,
                    initial_solution.fitness_score,
                ),
                "ils_vs_ga": self._gain_dict(
                    improved_solution.fitness_score if improved_solution is not None else genetic_solution.fitness_score,
                    genetic_solution.fitness_score,
                ),
                "final_vs_initial": self._gain_dict(
                    best_solution.fitness_score,
                    initial_solution.fitness_score,
                ),
            },
            "genetic": genetic_solver.get_run_metrics(),
            "embedded_ils": self.embedded_improvement_metrics,
        }
        self._write_metrics(log_dir, instance_name)

        if self.verbose:
            print("\nHybrid result")
            print("-" * 50)
            print(f"Initial: {initial_solution.fitness_score:,}")
            print(
                f"GA: {genetic_solution.fitness_score:,} "
                f"({self._format_gain(genetic_solution.fitness_score, initial_solution.fitness_score)})"
            )
            if improved_solution is not None:
                print(
                    f"ILS: {improved_solution.fitness_score:,} "
                    f"({self._format_gain(improved_solution.fitness_score, genetic_solution.fitness_score)})"
                )
            elif embedded_mode:
                best_gain = self.embedded_improvement_metrics["best_gain"]
                print(
                    f"GA-linked ILS: {self.embedded_improvement_metrics['improved']} accepted improvements, "
                    f"best call {self._format_raw_gain(best_gain)}"
                )
            print(
                f"Final: {best_solution.fitness_score:,} "
                f"({self._format_gain(best_solution.fitness_score, initial_solution.fitness_score)})"
            )

        return HybridResult(
            solution=best_solution,
            initial_solution=initial_solution,
            genetic_solution=genetic_solution,
            improved_solution=improved_solution,
            metrics=self.metrics,
        )

    def _build_genetic_solver(self, initial_solution, genetic_budget, embedded_mode=False):
        params = {
            key: value
            for key, value in self.ga_params.items()
            if key in GA_PARAM_KEYS
        }
        if embedded_mode:
            params.update(
                {
                    "embedded_improvement_callback": self._run_embedded_improvement,
                    "embedded_improvement_interval": self.embedded_improvement_interval,
                    "embedded_improvement_count": self.embedded_improvement_count,
                    "embedded_improvement_min_remaining": self.embedded_improvement_min_remaining,
                    "embedded_improvement_insert_policy": self.embedded_improvement_insert_policy,
                }
            )
            if (
                self.embedded_improvement_workers > 1
                and self.embedded_improvement_count > 1
            ):
                params["embedded_improvement_callback"] = None
                params["embedded_improvement_batch_callback"] = (
                    self._run_embedded_improvement_batch
                )
        return GeneticSolver(
            initial_solution=initial_solution.clone(),
            instance=self.instance,
            seed_solutions=self.seed_solutions,
            time_limit_sec=genetic_budget,
            verbose=self.verbose,
            **params,
        )

    def _embedded_params(self, time_slice):
        params = dict(self.improvement_params)
        if "alphas" in params and "alpha_values" not in params:
            params["alpha_values"] = params.pop("alphas")
        params = {
            key: value
            for key, value in params.items()
            if key in IMPROVEMENT_PARAM_KEYS
        }
        params.setdefault("init_max_time", min(0.5, max(0.1, time_slice)))
        params["restart_fresh_probability"] = 0.0
        return params

    def _run_embedded_improvement(self, solution, remaining_time, generation, rank):
        time_slice = min(
            self.embedded_improvement_time,
            max(0.0, remaining_time - self.embedded_improvement_min_remaining),
        )
        if time_slice <= 0.0:
            return None

        params = self._embedded_params(time_slice)

        input_score = solution.fitness_score
        started_at = time.time()
        solver = ImprovementSolver(seed=self.seed, verbose=False)
        candidate = solver.iterated_local_search(
            self.instance,
            instance_name=None,
            time_limit=time_slice,
            seed_solution=solution,
            seed_label=f"ga_gen_{generation}_elite_{rank + 1}",
            **params,
        )
        elapsed = time.time() - started_at
        gain = candidate.fitness_score - input_score

        self.embedded_improvement_metrics["attempts"] += 1
        self.embedded_improvement_metrics["elapsed_sec"] = round(
            self.embedded_improvement_metrics["elapsed_sec"] + elapsed,
            6,
        )
        if gain > 0:
            self.embedded_improvement_metrics["improved"] += 1
            self.embedded_improvement_metrics["total_positive_gain"] += int(gain)
            if gain > self.embedded_improvement_metrics["best_gain"]:
                self.embedded_improvement_metrics["best_gain"] = int(gain)
                self.embedded_improvement_metrics["best_input_score"] = int(input_score)
                self.embedded_improvement_metrics["best_output_score"] = int(candidate.fitness_score)

        return candidate

    def _run_embedded_improvement_batch(self, solutions, remaining_time, generation):
        time_slice = min(
            self.embedded_improvement_time,
            max(0.0, remaining_time - self.embedded_improvement_min_remaining),
        )
        if time_slice <= 0.0 or not solutions:
            return [None for _solution in solutions]

        params = self._embedded_params(time_slice)
        max_workers = min(self.embedded_improvement_workers, len(solutions))
        if max_workers <= 1:
            return self._run_embedded_improvement_batch_sequential(
                solutions,
                time_slice,
                generation,
            )

        results = [None for _solution in solutions]
        payloads = []
        for rank, solution in enumerate(solutions):
            worker_seed = (
                None
                if self.seed is None
                else int(self.seed) + generation * 1000 + rank
            )
            payloads.append(
                (
                    solution,
                    params,
                    time_slice,
                    worker_seed,
                    generation,
                    rank,
                )
            )

        try:
            executor = self._embedded_pool(max_workers)
            futures = {
                executor.submit(_embedded_improvement_worker, payload): payload[-1]
                for payload in payloads
            }
            for future in as_completed(futures):
                rank = futures[future]
                try:
                    result_rank, candidate, input_score, elapsed, error = future.result()
                except BrokenProcessPool as exc:
                    self._record_embedded_failure()
                    self._shutdown_embedded_executor()
                    self.embedded_improvement_workers = 1
                    if self.verbose:
                        print(
                            "ILS worker process pool broke; "
                            f"falling back to sequential calls ({type(exc).__name__}: {exc})"
                        )
                    return self._run_embedded_improvement_batch_sequential(
                        solutions,
                        time_slice,
                        generation,
                    )
                except Exception as exc:
                    self._record_embedded_failure()
                    if self.verbose:
                        print(
                            f"ILS worker {rank + 1} failed: "
                            f"{type(exc).__name__}: {exc!r}"
                        )
                    continue
                if error:
                    self._record_embedded_failure()
                    if self.verbose:
                        print(
                            f"ILS worker {rank + 1} failed:\n"
                            f"{error.strip()}"
                        )
                    continue
                self._record_embedded_result(candidate, input_score, elapsed)
                results[result_rank] = candidate
        except BrokenProcessPool as exc:
            self._record_embedded_failure()
            self._shutdown_embedded_executor()
            self.embedded_improvement_workers = 1
            if self.verbose:
                print(
                    "ILS worker process pool broke before completion; "
                    f"falling back to sequential calls ({type(exc).__name__}: {exc})"
                )
            return self._run_embedded_improvement_batch_sequential(
                solutions,
                time_slice,
                generation,
            )

        return results

    def _run_embedded_improvement_batch_sequential(self, solutions, time_slice, generation):
        results = []
        for rank, solution in enumerate(solutions):
            candidate = self._run_embedded_improvement(
                solution,
                time_slice + self.embedded_improvement_min_remaining,
                generation,
                rank,
            )
            results.append(candidate)
        return results

    def _embedded_pool(self, max_workers):
        if (
            self._embedded_executor is None
            or self._embedded_executor_workers != max_workers
        ):
            self._shutdown_embedded_executor()
            self._embedded_executor = ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_embedded_worker,
                initargs=(self.instance,),
            )
            self._embedded_executor_workers = max_workers
        return self._embedded_executor

    def _shutdown_embedded_executor(self):
        if self._embedded_executor is not None:
            self._embedded_executor.shutdown(wait=True, cancel_futures=True)
            self._embedded_executor = None
            self._embedded_executor_workers = 0

    def _record_embedded_result(self, candidate, input_score, elapsed):
        if candidate is None:
            self._record_embedded_failure()
            return
        gain = candidate.fitness_score - input_score
        self.embedded_improvement_metrics["attempts"] += 1
        self.embedded_improvement_metrics["elapsed_sec"] = round(
            self.embedded_improvement_metrics["elapsed_sec"] + elapsed,
            6,
        )
        if gain > 0:
            self.embedded_improvement_metrics["improved"] += 1
            self.embedded_improvement_metrics["total_positive_gain"] += int(gain)
            if gain > self.embedded_improvement_metrics["best_gain"]:
                self.embedded_improvement_metrics["best_gain"] = int(gain)
                self.embedded_improvement_metrics["best_input_score"] = int(input_score)
                self.embedded_improvement_metrics["best_output_score"] = int(candidate.fitness_score)

    def _record_embedded_failure(self):
        self.embedded_improvement_metrics["failures"] += 1

    def _run_improvement_phase(self, seed_solution, time_limit, instance_name=None, log_dir=None):
        params = dict(self.improvement_params)
        if "alphas" in params and "alpha_values" not in params:
            params["alpha_values"] = params.pop("alphas")
        params = {
            key: value
            for key, value in params.items()
            if key in IMPROVEMENT_PARAM_KEYS
        }
        params.setdefault("init_max_time", min(1.0, max(0.1, time_limit)))

        log_csv = None
        operator_stats_csv = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            base = self._safe_instance_name(instance_name)
            log_csv = os.path.join(log_dir, f"{base}.improvement.csv")
            operator_stats_csv = os.path.join(log_dir, f"{base}.operators.csv")

        solver = ImprovementSolver(seed=self.seed, verbose=self.verbose)
        return solver.iterated_local_search(
            self.instance,
            instance_name=instance_name,
            time_limit=time_limit,
            seed_solution=seed_solution,
            seed_label="genetic_best",
            log_csv=log_csv,
            operator_stats_csv=operator_stats_csv,
            **params,
        )

    def _write_genetic_logs(self, log_dir, instance_name, genetic_solver):
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        base = self._safe_instance_name(instance_name)
        path = os.path.join(log_dir, f"{base}.genetic.csv")
        rows = genetic_solver.convergence
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    def _write_metrics(self, log_dir, instance_name):
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        base = self._safe_instance_name(instance_name)
        path = os.path.join(log_dir, f"{base}.metrics.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.metrics, file, indent=2)

    @staticmethod
    def _prepare_seed_solutions(seed_solutions):
        seeds = []
        seen = set()
        for solution in seed_solutions or []:
            if solution is None:
                continue
            signature = tuple(solution.signed_libraries)
            if signature in seen:
                continue
            seen.add(signature)
            seeds.append(solution)
        return sorted(seeds, key=lambda item: item.fitness_score, reverse=True)

    @staticmethod
    def _normalize_hybrid_mode(mode):
        mode = (mode or "ga_with_parallel_ils").strip().lower().replace("-", "_")
        aliases = {
            "embedded_parallel": "ga_with_parallel_ils",
            "parallel_ils": "ga_with_parallel_ils",
            "ga_parallel_ils": "ga_with_parallel_ils",
            "ga_with_parallel_ils": "ga_with_parallel_ils",
            "sequential": "ga_with_ils",
            "embedded_sequential": "ga_with_ils",
            "inline_ils": "ga_with_ils",
            "single_ils": "ga_with_ils",
            "ga_with_ils": "ga_with_ils",
            "after_gens": "ga_then_ils",
            "after_generations": "ga_then_ils",
            "ga_then_improvement": "ga_then_ils",
            "ga_then_local_search": "ga_then_ils",
            "ga_then_ils": "ga_then_ils",
        }
        if mode in aliases:
            return aliases[mode]
        raise ValueError(
            "hybrid_mode must be one of: ga_with_parallel_ils, ga_with_ils, ga_then_ils"
        )

    @staticmethod
    def _safe_instance_name(instance_name):
        if not instance_name:
            return "instance"
        base = os.path.splitext(os.path.basename(instance_name))[0]
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)

    @staticmethod
    def _new_embedded_metrics():
        return {
            "attempts": 0,
            "improved": 0,
            "failures": 0,
            "total_positive_gain": 0,
            "best_gain": 0,
            "best_input_score": None,
            "best_output_score": None,
            "elapsed_sec": 0.0,
        }

    def _print_embedded_result(self):
        metrics = self.embedded_improvement_metrics
        print("\nGA-linked ILS result")
        print("-" * 50)
        print(f"Attempts: {metrics['attempts']}")
        print(f"Accepted improvements: {metrics['improved']}")
        print(f"Total positive ILS gain: {self._format_raw_gain(metrics['total_positive_gain'])}")
        print(f"Best single ILS gain: {self._format_raw_gain(metrics['best_gain'])}")
        print(f"GA-linked ILS elapsed: {metrics['elapsed_sec']:.1f}s")

    @staticmethod
    def _gain_dict(after, before):
        gain = int(after) - int(before)
        percent = (gain / before * 100.0) if before else 0.0
        return {
            "absolute": gain,
            "percent": round(percent, 6),
        }

    @staticmethod
    def _format_gain(after, before):
        gain = int(after) - int(before)
        percent = (gain / before * 100.0) if before else 0.0
        sign = "+" if gain >= 0 else ""
        return f"{sign}{gain:,} ({sign}{percent:.3f}%)"

    @staticmethod
    def _format_raw_gain(gain):
        gain = int(gain)
        sign = "+" if gain >= 0 else ""
        return f"{sign}{gain:,}"
