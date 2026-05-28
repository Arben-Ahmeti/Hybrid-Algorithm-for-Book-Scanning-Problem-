import random
import time
from typing import Dict, Sequence, Tuple

from models.instance_data import InstanceData
from models.selection_strategies import SelectionStrategies
from models.solution import Solution
from models.tweaks import Tweaks


class GeneticSolver:
    STRUCTURED_OPERATOR_LABELS = {"coverage_exchange", "paired_choice_flip"}
    ORDER_OPERATOR_WEIGHTS = [
        ("swap_signed_with_unsigned", 2.4),
        ("sampled_best_exchange", 2.2),
        ("diversity_swap", 1.8),
        ("critical_path_insert", 1.6),
        ("block_reinsert", 1.4),
        ("move_signed", 1.3),
        ("reverse_segment", 1.0),
        ("remove_library", 0.9),
        ("swap_signed", 0.8),
        ("insert_library", 0.8),
        ("swap_neighbor_libraries", 0.5),
        ("coverage_exchange", 0.2),
        ("paired_choice_flip", 0.1),
    ]

    def __init__(
        self,
        initial_solution: Solution,
        instance: InstanceData,
        population_size=100,
        generations=500,
        mutation_prob=0.39,
        crossover_rate=0.33,
        immigrant_frac=0.06,
        steady_state_ratio=0.25,
        time_limit_sec=10 * 60,
        mutation_steps=5,
        seed_solutions: Sequence[Solution] = None,
        mutation_accept_worse_prob=0.08,
        steady_state_offspring_factor=0.55,
        local_refine_steps=4,
        elite_refine_interval=5,
        elite_refine_count=2,
        initial_mutation_ratio=0.45,
        mutation_screen_attempts=0,
        mutation_exact_checks=2,
        local_screen_attempts=0,
        local_exact_checks=4,
        operator_weights: Dict[str, float] = None,
        selection_weights: Dict[str, float] = None,
        selection_tournament_size=10,
        embedded_improvement_callback=None,
        embedded_improvement_batch_callback=None,
        embedded_improvement_interval=0,
        embedded_improvement_count=1,
        embedded_improvement_min_remaining=1.0,
        embedded_improvement_insert_policy="all",
        verbose=True,
    ):
        self.initial_solution = initial_solution
        self.instance = instance
        self.population_size = max(2, int(population_size))
        self.generations = max(1, int(generations))
        self.population = []
        self.seed_solutions = self._prepare_seed_solutions(seed_solutions)
        self.mutation_prob = self._clamp_probability(mutation_prob)
        self.crossover_rate = self._clamp_probability(crossover_rate)
        self.immigrant_frac = self._clamp_probability(immigrant_frac)
        self.mutation_steps = max(1, int(mutation_steps))
        self.time_limit_sec = time_limit_sec
        self.steady_state_ratio = self._clamp_probability(steady_state_ratio)
        self.mutation_accept_worse_prob = self._clamp_probability(
            mutation_accept_worse_prob
        )
        self.steady_state_offspring_factor = max(0.05, steady_state_offspring_factor)
        self.local_refine_steps = max(0, int(local_refine_steps))
        self.elite_refine_interval = max(0, int(elite_refine_interval))
        self.elite_refine_count = max(0, int(elite_refine_count))
        self.initial_mutation_ratio = self._clamp_probability(initial_mutation_ratio)
        self.mutation_screen_attempts = max(0, int(mutation_screen_attempts))
        self.mutation_exact_checks = max(1, int(mutation_exact_checks))
        self.local_screen_attempts = max(0, int(local_screen_attempts))
        self.local_exact_checks = max(1, int(local_exact_checks))
        self.operator_weights = self._build_operator_weights(operator_weights)
        self.structured_refine_active = any(
            label in self.STRUCTURED_OPERATOR_LABELS
            for label, _weight in self.operator_weights
        )
        self.selection_weights = selection_weights
        self.selection_tournament_size = max(1, int(selection_tournament_size))
        self.embedded_improvement_callback = embedded_improvement_callback
        self.embedded_improvement_batch_callback = embedded_improvement_batch_callback
        self.embedded_improvement_interval = max(0, int(embedded_improvement_interval))
        self.embedded_improvement_count = max(1, int(embedded_improvement_count))
        self.embedded_improvement_min_remaining = max(
            0.0,
            float(embedded_improvement_min_remaining),
        )
        self.embedded_improvement_insert_policy = (
            "best" if embedded_improvement_insert_policy == "best" else "all"
        )
        self.verbose = verbose
        self.steady_gen_start = int(self.generations * (1 - self.steady_state_ratio))
        self.steady_time_start = self.time_limit_sec * (1 - self.steady_state_ratio)
        self.operator_stats = self._new_operator_stats()
        self.crossover_stats = {
            "attempts": 0,
            "applied": 0,
            "skipped": 0,
            "children_built": 0,
            "children_accepted": 0,
            "proxy_worse": 0,
            "exact_fallbacks": 0,
            "parent_fallbacks": 0,
        }
        self.embedded_improvement_stats = {
            "attempts": 0,
            "improved": 0,
            "best_updates": 0,
            "elapsed_sec": 0.0,
        }
        self.convergence = []
        self.generations_completed = 0
        self.elapsed_sec = 0.0
        self.stop_reason = "generation_limit"
        self.best_fitness = initial_solution.fitness_score

    def solve(self):
        population = self.initialize_population(self.initial_solution)

        start_time = time.time()
        deadline = start_time + self.time_limit_sec
        best_fitness = None
        plateau_counter = 0
        base_immigrant_frac = self.immigrant_frac

        for generation in range(self.generations):
            elapsed = time.time() - start_time
            if elapsed >= self.time_limit_sec:
                self.stop_reason = "time_limit"
                if self.verbose:
                    print(f"Stopping at gen {generation} due to time limit ({elapsed:.1f}s)")
                break

            population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
            best_solution = population[0]

            if self.structured_refine_active and self.local_refine_steps > 0:
                population = self._refine_structured_elites(population, deadline)
                best_solution = population[0]

            if (
                self.local_refine_steps > 0
                and self.elite_refine_interval > 0
                and generation > 0
                and generation % self.elite_refine_interval == 0
            ):
                population = self._refine_elites(population, deadline)
                best_solution = max(population, key=lambda x: x.fitness_score)
                if best_solution.fitness_score > population[0].fitness_score:
                    population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
                    best_solution = population[0]

            if self._should_run_embedded_improvement(generation, deadline):
                previous_best = best_solution.fitness_score
                population = self._apply_embedded_improvement(population, generation, deadline)
                population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
                best_solution = population[0]
                if self.verbose and best_solution.fitness_score > previous_best:
                    print(
                        f"GA-linked ILS improved GA best at gen {generation}: "
                        f"{previous_best} -> {best_solution.fitness_score}"
                    )

            elapsed = time.time() - start_time
            use_steady = (
                generation >= self.steady_gen_start
                or elapsed >= self.steady_time_start
            )
            improved = best_fitness is None or best_solution.fitness_score > best_fitness
            self._record_generation(
                generation,
                elapsed,
                population,
                "steady" if use_steady else "generative",
                improved,
            )
            if self.verbose:
                print(f"Gen {generation}: Best fitness = {best_solution.fitness_score}")

            if best_fitness is None or best_solution.fitness_score > best_fitness:
                best_fitness = best_solution.fitness_score
                plateau_counter = 0
                self.immigrant_frac = base_immigrant_frac
            else:
                plateau_counter += 1
                if plateau_counter > 5:
                    self.immigrant_frac = min(0.35, self.immigrant_frac * 1.5)

            if use_steady:
                new_population = self.create_offspring_steady_state(population, deadline)
            else:
                new_population = self.create_offspring_generative(population, deadline)

            num_immigrants = int(self.immigrant_frac * self.population_size)
            if num_immigrants > 0 and time.time() < deadline:
                immigrants = self.create_immigrants(num_immigrants, deadline)
                if immigrants:
                    new_population = sorted(new_population, key=lambda x: x.fitness_score, reverse=True)
                    new_population = new_population[:-len(immigrants)] + immigrants

            new_population = sorted(new_population, key=lambda x: x.fitness_score, reverse=True)
            if best_solution.fitness_score > new_population[-1].fitness_score:
                new_population[-1] = self._copy_solution(best_solution)

            population = sorted(
                new_population,
                key=lambda x: x.fitness_score,
                reverse=True,
            )[:self.population_size]
            self.generations_completed = generation + 1

        self.elapsed_sec = time.time() - start_time
        result = max(population, key=lambda x: x.fitness_score)
        self.best_fitness = result.fitness_score
        return result

    def create_offspring_generative(self, population, deadline=None):
        new_population = [self._copy_solution(population[0])]

        while len(new_population) < self.population_size:
            if deadline is not None and time.time() >= deadline:
                break
            parent1, parent2 = self._select_parents(population)
            offspring1, offspring2 = self._make_children(parent1, parent2, deadline)
            new_population.extend([offspring1, offspring2])

        while len(new_population) < self.population_size:
            new_population.append(self._copy_solution(random.choice(population)))
        return new_population[:self.population_size]

    def create_offspring_steady_state(self, population, deadline=None):
        new_population = [self._copy_solution(individual) for individual in population]
        offspring_budget = max(
            2,
            int(self.population_size * self.steady_state_offspring_factor),
        )

        for _ in range(offspring_budget // 2):
            if deadline is not None and time.time() >= deadline:
                break
            parent1, parent2 = self._select_parents(new_population)
            offspring1, offspring2 = self._make_children(parent1, parent2, deadline)
            combined = new_population + [offspring1, offspring2]
            new_population = sorted(
                combined,
                key=lambda x: x.fitness_score,
                reverse=True,
            )[:self.population_size]

        return new_population

    def initialize_population(self, initial_solution, tweak_ratio: float = None):
        tweak_ratio = self.initial_mutation_ratio if tweak_ratio is None else tweak_ratio
        seeds = self._prepare_seed_solutions([initial_solution] + self.seed_solutions)
        population = []
        seen = set()

        for seed in seeds:
            if len(population) >= self.population_size:
                break
            self._append_unique(population, self._copy_solution(seed), seen)

        mutated_budget = min(
            int(self.population_size * tweak_ratio),
            self.population_size - len(population),
        )
        for _ in range(mutated_budget):
            base = random.choice(seeds)
            candidate = self.mutate_solution(
                base,
                iterations=1,
                allow_structured=False,
            )
            self._append_unique(population, candidate, seen)

        while len(population) < self.population_size:
            if random.random() < 0.65:
                population.append(self._copy_solution(random.choice(seeds)))
            else:
                population.append(
                    self.mutate_solution(
                        random.choice(seeds),
                        iterations=1,
                        allow_structured=False,
                    )
                )

        return sorted(population, key=lambda x: x.fitness_score, reverse=True)

    def create_immigrants(self, count, deadline=None):
        seeds = self._prepare_seed_solutions([self.initial_solution] + self.seed_solutions)
        immigrants = []
        for _ in range(count):
            if deadline is not None and time.time() >= deadline:
                break
            base = random.choice(seeds)
            immigrants.append(
                self.mutate_solution(
                    base,
                    iterations=random.randint(1, max(1, min(4, self.mutation_steps + 1))),
                    allow_structured=False,
                )
            )
        return immigrants

    def _select_parents(self, population):
        selection_method = SelectionStrategies.choose_selection_method(
            self.selection_weights,
            self.selection_tournament_size,
        )
        parent1 = selection_method(population)
        parent2 = selection_method(population)
        if len(population) > 1:
            for _ in range(3):
                if parent2 is not parent1:
                    break
                parent2 = selection_method(population)
            if parent2 is parent1:
                alternatives = [individual for individual in population if individual is not parent1]
                if alternatives:
                    parent2 = random.choice(alternatives)
        return parent1, parent2

    def _make_children(self, parent1, parent2, deadline=None):
        self.crossover_stats["attempts"] += 1
        if random.random() < self.crossover_rate:
            self.crossover_stats["applied"] += 1
            offspring1, offspring2 = self.crossover(parent1, parent2)
        else:
            self.crossover_stats["skipped"] += 1
            offspring1 = self._copy_solution(parent1)
            offspring2 = self._copy_solution(parent2)

        if deadline is not None and time.time() >= deadline:
            return offspring1, offspring2

        if random.random() < self.mutation_prob:
            offspring1 = self.mutate_solution(
                offspring1,
                iterations=self.mutation_steps,
                allow_structured=False,
            )
        if random.random() < self.mutation_prob:
            offspring2 = self.mutate_solution(
                offspring2,
                iterations=self.mutation_steps,
                allow_structured=False,
            )

        return offspring1, offspring2

    def mutate_solution(self, solution, iterations=1, allow_structured=True):
        current = self._copy_solution(solution)
        best = self._copy_solution(current)

        for _ in range(max(1, iterations)):
            candidate, operator_label = self._best_screened_neighbor(
                current,
                allow_structured=allow_structured,
            )
            if candidate is None:
                continue

            if candidate.fitness_score > best.fitness_score:
                best = self._copy_solution(candidate)
                if operator_label is not None:
                    self.operator_stats[operator_label]["best_updates"] += 1

            if (
                candidate.fitness_score >= current.fitness_score
                or random.random() < self.mutation_accept_worse_prob
            ):
                current = candidate
                if operator_label is not None:
                    self.operator_stats[operator_label]["accepted"] += 1

        if best.fitness_score >= solution.fitness_score:
            return best
        return current

    def _best_screened_neighbor(
        self,
        solution,
        attempts=None,
        exact_checks=2,
        allow_structured=True,
    ):
        attempts = attempts or self.mutation_screen_attempts or max(8, self.mutation_steps * 2)
        exact_checks = max(1, exact_checks or self.mutation_exact_checks)
        base_order = solution.ordered_libraries()
        base_proxy = self._screen_score(base_order)
        scored_orders = []
        seen = set()

        for _ in range(attempts):
            operator_label, order = self._candidate_order(
                solution,
                allow_structured=allow_structured,
            )
            if order is None:
                continue
            is_structured = operator_label in self.STRUCTURED_OPERATOR_LABELS
            key = tuple(order[: min(len(order), 80)])
            if key in seen:
                continue
            seen.add(key)

            proxy_score = self._screen_score(order)
            if (
                is_structured or
                proxy_score >= base_proxy
                or len(scored_orders) < exact_checks
                or random.random() < 0.08
            ):
                self.operator_stats[operator_label]["screened"] += 1
                scored_orders.append(
                    (1 if is_structured else 0, proxy_score, operator_label, order)
                )

        if not scored_orders:
            return None, None

        scored_orders.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_candidate = None
        best_label = None
        for _priority, _proxy_score, operator_label, order in scored_orders[:exact_checks]:
            self.operator_stats[operator_label]["exact"] += 1
            candidate = Solution.from_order(order, self.instance)
            if candidate.fitness_score > solution.fitness_score:
                self.operator_stats[operator_label]["improved"] += 1
            if (
                best_candidate is None
                or candidate.fitness_score > best_candidate.fitness_score
            ):
                best_candidate = candidate
                best_label = operator_label

        return best_candidate, best_label

    def _refine_elites(self, population, deadline):
        refined = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        count = min(self.elite_refine_count, len(refined))

        for idx in range(count):
            if time.time() >= deadline:
                break
            current = refined[idx]
            for _ in range(max(1, self.local_refine_steps)):
                if time.time() >= deadline:
                    break
                candidate, _operator_label = self._best_screened_neighbor(
                    current,
                    attempts=self.local_screen_attempts or max(12, self.mutation_steps * 5),
                    exact_checks=self.local_exact_checks,
                )
                if candidate is None or candidate.fitness_score <= current.fitness_score:
                    break
                current = candidate
            refined[idx] = current

        return sorted(refined, key=lambda x: x.fitness_score, reverse=True)

    def _refine_structured_elites(self, population, deadline):
        refined = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        if not refined:
            return refined

        current = refined[0]
        structured_pool = [
            (label, weight)
            for label, weight in self.operator_weights
            if label in self.STRUCTURED_OPERATOR_LABELS
        ]
        structured_pool.sort(key=lambda item: item[1], reverse=True)

        for label, _weight in structured_pool:
            if time.time() >= deadline:
                break
            self.operator_stats[label]["attempts"] += 1
            if label == "coverage_exchange":
                order = Tweaks._order_coverage_exchange(current, self.instance)
            elif label == "paired_choice_flip":
                order = Tweaks._order_paired_choice_flip(current, self.instance)
            else:
                order = None

            if order is None:
                continue

            self.operator_stats[label]["produced"] += 1
            self.operator_stats[label]["exact"] += 1
            candidate = Solution.from_order(order, self.instance)
            if candidate.fitness_score > current.fitness_score:
                self.operator_stats[label]["improved"] += 1
                self.operator_stats[label]["accepted"] += 1
                self.operator_stats[label]["best_updates"] += 1
                current = candidate

        refined[0] = current
        return sorted(refined, key=lambda x: x.fitness_score, reverse=True)

    def _should_run_embedded_improvement(self, generation, deadline):
        if (
            self.embedded_improvement_callback is None
            and self.embedded_improvement_batch_callback is None
        ):
            return False
        if self.embedded_improvement_interval <= 0:
            return False
        if generation <= 0 or generation % self.embedded_improvement_interval != 0:
            return False
        return (deadline - time.time()) > self.embedded_improvement_min_remaining

    def _apply_embedded_improvement(self, population, generation, deadline):
        improved_population = list(population)
        elite_count = min(self.embedded_improvement_count, len(improved_population))

        if self.embedded_improvement_batch_callback is not None:
            remaining = deadline - time.time()
            if remaining <= self.embedded_improvement_min_remaining:
                return improved_population
            positions = list(range(elite_count))
            elites = [improved_population[idx].shallow_copy() for idx in positions]
            self.embedded_improvement_stats["attempts"] += len(elites)
            started_at = time.time()
            candidates = self.embedded_improvement_batch_callback(
                elites,
                remaining,
                generation,
            )
            self.embedded_improvement_stats["elapsed_sec"] += time.time() - started_at
            return self._merge_embedded_candidates(
                improved_population,
                positions,
                candidates,
            )

        positions = []
        candidates = []
        for idx in range(elite_count):
            remaining = deadline - time.time()
            if remaining <= self.embedded_improvement_min_remaining:
                break

            current = improved_population[idx]
            self.embedded_improvement_stats["attempts"] += 1
            started_at = time.time()
            candidate = self.embedded_improvement_callback(
                current.shallow_copy(),
                remaining,
                generation,
                idx,
            )
            self.embedded_improvement_stats["elapsed_sec"] += time.time() - started_at
            positions.append(idx)
            candidates.append(candidate)

        return self._merge_embedded_candidates(
            improved_population,
            positions,
            candidates,
        )

    def _merge_embedded_candidates(self, population, positions, candidates):
        improvements = []
        for idx, candidate in zip(positions, candidates or []):
            if candidate is None:
                continue
            current = population[idx]
            if candidate.fitness_score > current.fitness_score:
                improvements.append((idx, current, candidate))

        if not improvements:
            return population

        if self.embedded_improvement_insert_policy == "best":
            _idx, _current, candidate = max(
                improvements,
                key=lambda item: item[2].fitness_score,
            )
            worst_idx = min(
                range(len(population)),
                key=lambda item_idx: population[item_idx].fitness_score,
            )
            if candidate.fitness_score > population[worst_idx].fitness_score:
                if candidate.fitness_score > population[0].fitness_score:
                    self.embedded_improvement_stats["best_updates"] += 1
                self.embedded_improvement_stats["improved"] += 1
                population[worst_idx] = candidate
            return population

        for idx, _current, candidate in improvements:
            if candidate.fitness_score > population[0].fitness_score:
                self.embedded_improvement_stats["best_updates"] += 1
            self.embedded_improvement_stats["improved"] += 1
            population[idx] = candidate

        return population

    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        order1 = self._normalise_order(parent1.ordered_libraries())
        order2 = self._normalise_order(parent2.ordered_libraries())

        if len(order1) < 2 or len(order2) < 2:
            return self._copy_solution(parent1), self._copy_solution(parent2)

        child1_order = self._order_crossover(order1, order2)
        child2_order = self._order_crossover(order2, order1)

        child1 = self._build_child_if_promising(child1_order, parent1, parent2)
        child2 = self._build_child_if_promising(child2_order, parent2, parent1)
        return child1, child2

    def _build_child_if_promising(self, order, primary_parent, secondary_parent):
        parent_proxy = min(
            self._screen_score(primary_parent.ordered_libraries()),
            self._screen_score(secondary_parent.ordered_libraries()),
        )
        child_proxy = self._screen_score(order)

        if child_proxy < parent_proxy:
            self.crossover_stats["proxy_worse"] += 1

        child = Solution.from_order(order, self.instance)
        self.crossover_stats["children_built"] += 1
        if (
            child.fitness_score < primary_parent.fitness_score
            and random.random() > self.mutation_accept_worse_prob
        ):
            self.crossover_stats["exact_fallbacks"] += 1
            self.crossover_stats["parent_fallbacks"] += 1
            return self._copy_solution(primary_parent)
        self.crossover_stats["children_accepted"] += 1
        return child

    def _candidate_order(self, solution, allow_structured=True):
        operator_pool = self.operator_weights
        if not allow_structured:
            regular_pool = [
                item
                for item in self.operator_weights
                if item[0] not in self.STRUCTURED_OPERATOR_LABELS
            ]
            if regular_pool:
                operator_pool = regular_pool
        labels, weights = zip(*operator_pool)
        label = random.choices(labels, weights=weights, k=1)[0]
        self.operator_stats[label]["attempts"] += 1

        if label == "swap_signed":
            order = self._order_swap_signed(solution)
        elif label == "swap_signed_with_unsigned":
            order = self._order_swap_signed_with_unsigned(solution)
        elif label == "move_signed":
            order = self._order_move_signed(solution)
        elif label == "swap_neighbor_libraries":
            order = self._order_swap_neighbor_libraries(solution)
        elif label == "insert_library":
            order = self._order_insert_library(solution)
        elif label == "critical_path_insert":
            order = self._order_critical_path_insert(solution)
        elif label == "remove_library":
            order = self._order_remove_library(solution)
        elif label == "reverse_segment":
            order = self._order_reverse_segment(solution)
        elif label == "block_reinsert":
            order = self._order_block_reinsert(solution)
        elif label == "diversity_swap":
            order = self._order_diversity_swap(solution)
        elif label == "sampled_best_exchange":
            order = self._order_sampled_best_exchange(solution)
        elif label == "coverage_exchange":
            order = Tweaks._order_coverage_exchange(solution, self.instance)
        elif label == "paired_choice_flip":
            order = Tweaks._order_paired_choice_flip(solution, self.instance)
        else:
            order = None

        if order is not None:
            self.operator_stats[label]["produced"] += 1
        return label, order

    def _order_swap_signed(self, solution):
        signed_count = len(solution.signed_libraries)
        if signed_count < 2:
            return None
        order = solution.ordered_libraries()
        i, j = random.sample(range(signed_count), 2)
        order[i], order[j] = order[j], order[i]
        return order

    def _order_swap_signed_with_unsigned(self, solution):
        signed_count = len(solution.signed_libraries)
        if signed_count == 0 or not solution.unsigned_libraries:
            return None
        order = solution.ordered_libraries()
        signed_idx = random.randrange(signed_count)
        unsigned_frontier = min(
            len(solution.unsigned_libraries),
            max(16, 4 * self._frontier_size(signed_count)),
        )
        unsigned_idx = signed_count + random.randrange(unsigned_frontier)
        order[signed_idx], order[unsigned_idx] = order[unsigned_idx], order[signed_idx]
        return order

    def _order_move_signed(self, solution):
        signed_count = len(solution.signed_libraries)
        if signed_count < 2:
            return None
        order = solution.ordered_libraries()
        i, j = random.sample(range(signed_count), 2)
        lib_id = order.pop(i)
        order.insert(j, lib_id)
        return order

    def _order_swap_neighbor_libraries(self, solution):
        signed_count = len(solution.signed_libraries)
        if signed_count < 2:
            return None
        order = solution.ordered_libraries()
        pos = random.randrange(signed_count - 1)
        order[pos], order[pos + 1] = order[pos + 1], order[pos]
        return order

    def _order_insert_library(self, solution):
        signed_count = len(solution.signed_libraries)
        if not solution.unsigned_libraries:
            return None
        order = solution.ordered_libraries()
        unsigned_frontier = min(
            len(solution.unsigned_libraries),
            max(16, 4 * self._frontier_size(signed_count)),
        )
        source = signed_count + random.randrange(unsigned_frontier)
        lib_id = order.pop(source)
        target = random.randrange(max(1, signed_count + 1))
        order.insert(target, lib_id)
        return order

    def _order_critical_path_insert(self, solution):
        signed_count = len(solution.signed_libraries)
        unsigned = solution.unsigned_libraries
        if not unsigned:
            return None

        candidates = sorted(
            unsigned,
            key=lambda lib_id: self._unsigned_value(solution, lib_id),
            reverse=True,
        )[: min(16, len(unsigned))]
        candidates = [lib_id for lib_id in candidates if self._unsigned_value(solution, lib_id) > 0]
        if not candidates:
            return None

        best_lib = random.choice(candidates[: max(1, len(candidates) // 2)])
        order = solution.ordered_libraries()
        order.remove(best_lib)
        early_limit = max(1, min(signed_count, max(2, signed_count // 3)))
        order.insert(random.randrange(early_limit), best_lib)
        return order

    def _order_remove_library(self, solution):
        signed_count = len(solution.signed_libraries)
        if signed_count == 0:
            return None
        order = solution.ordered_libraries()
        weak_positions = self._weak_signed_positions(solution)
        idx = random.choice(weak_positions) if weak_positions else random.randrange(signed_count)
        lib_id = order.pop(idx)
        order.append(lib_id)
        return order

    def _order_reverse_segment(self, solution):
        signed_count = len(solution.signed_libraries)
        if signed_count < 3:
            return None
        order = solution.ordered_libraries()
        max_segment = min(signed_count, max(3, int(signed_count ** 0.5) + 1))
        segment = random.randint(2, max_segment)
        start = random.randrange(0, signed_count - segment + 1)
        end = start + segment
        order[start:end] = reversed(order[start:end])
        return order

    def _order_block_reinsert(self, solution):
        signed_count = len(solution.signed_libraries)
        if signed_count < 3:
            return None
        order = solution.ordered_libraries()
        block_size = min(max(2, signed_count // 10), 8, signed_count)
        start = random.randrange(0, signed_count - block_size + 1)
        block = order[start:start + block_size]
        del order[start:start + block_size]
        insert_at = random.randrange(0, max(1, signed_count - block_size + 1))
        order[insert_at:insert_at] = block
        return order

    def _order_diversity_swap(self, solution):
        signed_count = len(solution.signed_libraries)
        if signed_count == 0 or not solution.unsigned_libraries:
            return None
        weak_positions = self._weak_signed_positions(solution)
        strong_unsigned = self._rank_unsigned_candidates(solution, sample_limit=32)
        if not weak_positions or not strong_unsigned:
            return None

        weak_idx = random.choice(weak_positions)
        replacement = random.choice(strong_unsigned[: max(1, min(6, len(strong_unsigned)))])
        order = solution.ordered_libraries()
        replacement_idx = order.index(replacement)
        order[weak_idx], order[replacement_idx] = order[replacement_idx], order[weak_idx]
        return order

    def _order_sampled_best_exchange(self, solution, trials=24):
        signed_count = len(solution.signed_libraries)
        if signed_count == 0 or not solution.unsigned_libraries:
            return None

        order = solution.ordered_libraries()
        weak_positions = self._weak_signed_positions(solution, limit=6)
        strong_unsigned = self._rank_unsigned_candidates(solution, sample_limit=48)[:24]
        if not weak_positions or not strong_unsigned:
            return None

        base_score = self._screen_score(order)
        best_score = base_score
        best_order = None
        attempts = 0

        for signed_pos in weak_positions:
            for replacement in strong_unsigned:
                if attempts >= trials:
                    break
                attempts += 1
                candidate_order = order.copy()
                replacement_pos = candidate_order.index(replacement)
                candidate_order[signed_pos], candidate_order[replacement_pos] = (
                    candidate_order[replacement_pos],
                    candidate_order[signed_pos],
                )
                score = self._screen_score(candidate_order)
                if score > best_score:
                    best_score = score
                    best_order = candidate_order
            if attempts >= trials:
                break

        return best_order

    def _weak_signed_positions(self, solution, limit=5):
        scored = []
        for pos, lib_id in enumerate(solution.signed_libraries):
            books = solution.scanned_books_per_library.get(lib_id, [])
            score = sum(self.instance.scores[book_id] for book_id in books)
            signup = max(1, self.instance.lib_signup_days[lib_id])
            scored.append((score / signup, score, len(books), pos))

        scored.sort()
        return [pos for _eff, _score, _count, pos in scored[: max(1, min(limit, len(scored)))]]

    def _rank_unsigned_candidates(self, solution, sample_limit=32):
        unsigned = solution.unsigned_libraries
        if not unsigned:
            return []
        sample_size = min(len(unsigned), max(sample_limit, int(len(unsigned) ** 0.5) * 4))
        sampled = random.sample(unsigned, sample_size) if len(unsigned) > sample_size else unsigned
        ranked = [
            (self._unsigned_value(solution, lib_id), lib_id)
            for lib_id in sampled
        ]
        ranked = [(value, lib_id) for value, lib_id in ranked if value > 0]
        ranked.sort(reverse=True)
        return [lib_id for _value, lib_id in ranked]

    def _unsigned_value(self, solution, lib_id):
        signup = self.instance.lib_signup_days[lib_id]
        if signup >= self.instance.num_days:
            return 0.0

        limit = min(
            self.instance.lib_num_books[lib_id],
            max(12, self.instance.lib_books_per_day[lib_id] * 6),
        )
        value = 0.0
        novel = 0
        scanned = solution.scanned_books
        for book_id in self.instance.lib_book_ids[lib_id][:limit]:
            if book_id in scanned:
                continue
            value += self.instance.scores[book_id] / max(1, self.instance.book_freq[book_id])
            novel += 1

        if novel == 0:
            return 0.0

        fill_ratio = novel / max(1, limit)
        throughput_bonus = 1.0 + min(3.0, self.instance.lib_books_per_day[lib_id] / 10.0)
        return value * (0.5 + fill_ratio) * throughput_bonus / max(1, signup)

    def _screen_score(self, order):
        screen = getattr(self.instance, "screen_evaluate_sequential", None)
        if callable(screen):
            return screen(order)
        return self.instance.fast_evaluate(order)

    def _normalise_order(self, order):
        seen = set()
        normalised = []
        for lib_id in order:
            if lib_id in seen or lib_id < 0 or lib_id >= self.instance.num_libs:
                continue
            seen.add(lib_id)
            normalised.append(lib_id)
        for lib_id in range(self.instance.num_libs):
            if lib_id not in seen:
                normalised.append(lib_id)
        return normalised

    @staticmethod
    def _order_crossover(primary_order, secondary_order):
        size = len(primary_order)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        inherited = primary_order[start:end + 1]
        child[start:end + 1] = inherited
        used = set(inherited)

        fill_values = [lib_id for lib_id in secondary_order if lib_id not in used]
        fill_index = 0
        child_index = (end + 1) % size
        while fill_index < len(fill_values):
            if child[child_index] is None:
                child[child_index] = fill_values[fill_index]
                fill_index += 1
            child_index = (child_index + 1) % size

        return child

    @staticmethod
    def _frontier_size(signed_count):
        return min(signed_count, max(12, 2 * int(max(1, signed_count) ** 0.5)))

    def _record_generation(self, generation, elapsed, population, mode, improved):
        scores = [individual.fitness_score for individual in population]
        self.convergence.append(
            {
                "generation": generation,
                "elapsed_sec": round(elapsed, 6),
                "best_fitness": int(scores[0]),
                "avg_fitness": round(sum(scores) / len(scores), 3),
                "worst_fitness": int(scores[-1]),
                "mode": mode,
                "improved": int(bool(improved)),
                "immigrant_frac": round(self.immigrant_frac, 6),
            }
        )

    def get_run_metrics(self):
        return {
            "best_fitness": int(self.best_fitness),
            "generations_completed": self.generations_completed,
            "stop_reason": self.stop_reason,
            "elapsed_sec": round(self.elapsed_sec, 6),
            "parameters": {
                "population_size": self.population_size,
                "generations": self.generations,
                "mutation_prob": self.mutation_prob,
                "crossover_rate": self.crossover_rate,
                "immigrant_frac": self.immigrant_frac,
                "steady_state_ratio": self.steady_state_ratio,
                "mutation_steps": self.mutation_steps,
                "mutation_accept_worse_prob": self.mutation_accept_worse_prob,
                "steady_state_offspring_factor": self.steady_state_offspring_factor,
                "local_refine_steps": self.local_refine_steps,
                "elite_refine_interval": self.elite_refine_interval,
                "elite_refine_count": self.elite_refine_count,
                "initial_mutation_ratio": self.initial_mutation_ratio,
                "mutation_screen_attempts": self.mutation_screen_attempts,
                "mutation_exact_checks": self.mutation_exact_checks,
                "local_screen_attempts": self.local_screen_attempts,
                "local_exact_checks": self.local_exact_checks,
                "operator_weights": dict(self.operator_weights),
                "selection_weights": self.selection_weights,
                "selection_tournament_size": self.selection_tournament_size,
                "embedded_improvement_interval": self.embedded_improvement_interval,
                "embedded_improvement_count": self.embedded_improvement_count,
                "embedded_improvement_min_remaining": self.embedded_improvement_min_remaining,
                "embedded_improvement_insert_policy": self.embedded_improvement_insert_policy,
            },
            "operator_stats": self.operator_stats,
            "crossover_stats": self.crossover_stats,
            "embedded_improvement_stats": {
                **self.embedded_improvement_stats,
                "elapsed_sec": round(
                    self.embedded_improvement_stats["elapsed_sec"],
                    6,
                ),
            },
            "convergence_rows": len(self.convergence),
        }

    def _new_operator_stats(self):
        return {
            label: {
                "attempts": 0,
                "produced": 0,
                "screened": 0,
                "exact": 0,
                "improved": 0,
                "accepted": 0,
                "best_updates": 0,
            }
            for label, _weight in self.ORDER_OPERATOR_WEIGHTS
        }

    def _build_operator_weights(self, operator_weights):
        provided = operator_weights or {}
        raw_weights = {}
        for label, default_weight in self.ORDER_OPERATOR_WEIGHTS:
            if label in self.STRUCTURED_OPERATOR_LABELS:
                weight = default_weight
            else:
                weight = float(provided.get(label, default_weight))
            raw_weights[label] = max(0.0, weight)

        adjusted_weights = Tweaks.instance_adjusted_weights(raw_weights, self.instance)
        weights = [
            (label, adjusted_weights.get(label, 0.0))
            for label, _default_weight in self.ORDER_OPERATOR_WEIGHTS
            if adjusted_weights.get(label, 0.0) > 0
        ]
        if weights:
            return weights

        fallback = Tweaks.instance_adjusted_weights(
            dict(self.ORDER_OPERATOR_WEIGHTS),
            self.instance,
        )
        return [
            (label, fallback.get(label, default_weight))
            for label, default_weight in self.ORDER_OPERATOR_WEIGHTS
            if fallback.get(label, default_weight) > 0
        ]

    @staticmethod
    def _clamp_probability(value):
        return min(1.0, max(0.0, float(value)))

    @staticmethod
    def _copy_solution(solution):
        return solution.shallow_copy(copy_scan_data=False)

    def _prepare_seed_solutions(self, seed_solutions):
        seeds = []
        seen = set()
        for seed in seed_solutions or []:
            if seed is None:
                continue
            signature = tuple(seed.signed_libraries)
            if signature in seen:
                continue
            seen.add(signature)
            seeds.append(seed)
        if not seeds:
            seeds = [self.initial_solution]
        return sorted(seeds, key=lambda x: x.fitness_score, reverse=True)

    @staticmethod
    def _append_unique(population, solution, seen):
        signature = tuple(solution.signed_libraries)
        if signature in seen:
            return False
        seen.add(signature)
        population.append(solution)
        return True
