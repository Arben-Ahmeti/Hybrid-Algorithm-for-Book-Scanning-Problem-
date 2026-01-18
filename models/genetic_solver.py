import random
import time
from typing import Tuple

from models.selection_strategies import SelectionStrategies
from models.tweaks import Tweaks
from models.solution import Solution
from models.instance_data import InstanceData
from models.ils_solver import ILS_Solver
import copy
from concurrent.futures import ProcessPoolExecutor
def ils_worker(args):
    """Worker function for parallel ILS execution"""
    data, solution, worker_id = args
    solver = ILS_Solver()
    original_fitness = solution.fitness_score
    print(f"[Worker {worker_id}] Original fitness: {original_fitness}")
    result = solver.iterated_local_search(
        data,
        solution,
        time_limit=10,
        max_iterations=1000
    )
    new_fitness = result.fitness_score
    print(f"[Worker {worker_id}] New fitness: {new_fitness}")
    return (original_fitness, new_fitness, result, worker_id)
class GeneticSolver:
    def __init__(self,
                 initial_solution: Solution,
                 instance: InstanceData,
                 population_size=100,
                 generations=500,
                 mutation_prob=0.39,
                 crossover_rate=0.33,
                 immigrant_frac=0.06,
                 steady_state_ratio=0.25,
                 time_limit_sec=10 * 60,
                 tweak_steps=5
                 ):
        self.initial_solution = initial_solution
        self.instance = instance
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.mutation_prob = mutation_prob
        self.crossover_rate = crossover_rate
        self.immigrant_frac = immigrant_frac
        self.tweak_steps = tweak_steps
        self.time_limit_sec = time_limit_sec
        self.steady_state_ratio = steady_state_ratio
        self.steady_gen_start = int(self.generations * (1 - steady_state_ratio))
        self.steady_time_start = self.time_limit_sec * (1 - steady_state_ratio)

    def solve(self):
        # Initialize population with slight variations of initial solution
        population = self.initialize_population(self.initial_solution)

        start_time = time.time()

        best_fitness = None
        plateau_counter = 0
        base_immigrant_frac = self.immigrant_frac

        #-bb
        data = copy.deepcopy(self.instance)

        for generation in range(self.generations):
            elapsed = time.time() - start_time
            if elapsed >= self.time_limit_sec:
                # print(f"Stopping at gen {generation} due to time limit ({elapsed:.1f}s)")
                break

            # Evaluate population
            population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
            best_solution = population[0]
            # print(f"Gen {generation}: Best fitness = {best_solution.fitness_score}")

            # Plateau tracking
            if best_fitness is None or best_solution.fitness_score > best_fitness:
                best_fitness = best_solution.fitness_score
                plateau_counter = 0
                self.immigrant_frac = base_immigrant_frac  # Reset if improvement
            else:
                plateau_counter += 1
                # Increase immigrant_frac after N stagnant generations (e.g., 10)
                if plateau_counter > 5:
                    self.immigrant_frac = min(1.0, self.immigrant_frac * 1.5)

            # decide whether to use generational or steady-state:
            use_steady = (
                    generation >= self.steady_gen_start
                    or elapsed >= self.steady_time_start
            )
            if not use_steady:
                new_population = self.create_offspring_generative(population)
            else:
                new_population = self.create_offspring_steady_state(population)

            num_immigrants = int(self.immigrant_frac * self.population_size)
            if num_immigrants > 0:
                immigrants = self.initialize_population(self.initial_solution)[:num_immigrants]

                # Remove worst individuals to make room for immigrants
                new_population = sorted(new_population, key=lambda x: x.fitness_score, reverse=True)
                new_population = new_population[:-num_immigrants] + immigrants

            # Ensure best solution is not lost
            if best_solution.fitness_score > min(new_population, key=lambda x: x.fitness_score).fitness_score:
                new_population[-1] = best_solution
            # ==========

            tasks = [
                (data, copy.deepcopy(best_solution), 0),  # solver1 task
                (data, copy.deepcopy(new_population[0]), 1),  # solver2 task
                (data, copy.deepcopy(new_population[1]), 2),  # solver3 task
                (data, copy.deepcopy(new_population[2]), 3),  # solver4 task
                (data, copy.deepcopy(new_population[3]), 4),  # solver5 task
                (data, copy.deepcopy(new_population[4]), 5)  # solver6 task
            ]

            # Execute all three in parallel
            with ProcessPoolExecutor(max_workers=6) as executor:
                results = list(executor.map(ils_worker, tasks))

            # Process results and find the best
            best_improvement = -float('inf')
            best_solution_found = None

            for original_fitness, new_fitness, solution, worker_id in results:
                if original_fitness < new_fitness:
                    print(f"Yeahh, u gjet - Worker {worker_id}")

                    if new_fitness > best_improvement:
                        best_improvement = new_fitness
                        best_solution_found = solution

            # Update if improvement found
            if best_solution_found is not None:
                best_solution = best_solution_found
                new_population[-1] = best_solution_found

          #===============
            # Update population
            population = new_population[:self.population_size]

        return max(population, key=lambda x: x.fitness_score)

    def create_offspring_generative(self, population):
        new_population = []
        while len(new_population) < self.population_size:
            selection_method = SelectionStrategies.choose_selection_method()
            parent1 = selection_method(population)
            parent2 = selection_method(population)

            offspring1, offspring2 = self.crossover(parent1, parent2)

            if random.random() < self.mutation_prob:
                offspring1 = Tweaks.tweak_with_iterations(offspring1, self.instance, iterations=self.tweak_steps)
            if random.random() < self.mutation_prob:
                offspring2 = Tweaks.tweak_with_iterations(offspring2, self.instance, iterations=self.tweak_steps)

            new_population.extend([offspring1.shallow_copy(), offspring2.shallow_copy()])

        return new_population

    def create_offspring_steady_state(self, population):
        new_population = [population[0]]

        while len(new_population) < self.population_size:
            # Selection
            selection_method = SelectionStrategies.choose_selection_method()
            parent1 = selection_method(population)
            parent2 = selection_method(population)

            offspring1, offspring2 = self.crossover(parent1, parent2)

            if random.random() < self.mutation_prob:
                offspring1 = Tweaks.tweak_with_iterations(offspring1, self.instance, iterations=self.tweak_steps)
            if random.random() < self.mutation_prob:
                offspring2 = Tweaks.tweak_with_iterations(offspring2, self.instance, iterations=self.tweak_steps)

            # Combine the population with offspring and select the best ones
            combined = population + [offspring1.shallow_copy(), offspring2.shallow_copy()]
            new_population = sorted(combined, key=lambda x: x.fitness_score, reverse=True)[:self.population_size]

        return new_population

    def initialize_population(self, initial_solution, tweak_ratio: float = 0.5):
        population = [initial_solution.shallow_copy()]

        num_tweaked = int(self.population_size * tweak_ratio)
        num_clones = self.population_size - num_tweaked - 1

        # Add tweaked solutions
        for _ in range(num_tweaked):
            tweaked = Tweaks.tweak_with_iterations(
                initial_solution,
                self.instance,
                iterations=random.randint(1, self.tweak_steps)
            )
            population.append(tweaked.shallow_copy())

        # Add direct shallow clones
        for _ in range(num_clones):
            population.append(initial_solution.shallow_copy())

        return population


    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:

        def create_offspring(p1_signed, p2_signed):
            size = len(p1_signed)

            # Convert to sets for O(1) lookups
            p1_set = set(p1_signed)
            p2_set = set(p2_signed)

            # Create quick lookup for used libraries
            used_libs = set()

            # Initialize offspring with None
            offspring_signed = [None] * size

            # 1. Select random positions from parent1 (faster sampling)
            copy_indices = random.sample(range(size), k=size // 2)
            for idx in copy_indices:
                lib = p1_signed[idx]
                offspring_signed[idx] = lib
                used_libs.add(lib)

            # 2. Prepare parent2 libraries not yet used
            available_p2 = [lib for lib in p2_signed if lib not in used_libs]
            p2_ptr = 0

            # 3. Fill remaining positions
            remaining_indices = [i for i, lib in enumerate(offspring_signed) if lib is None]

            for idx in remaining_indices:
                if p2_ptr < len(available_p2):
                    offspring_signed[idx] = available_p2[p2_ptr]
                    p2_ptr += 1
                else:
                    # Fallback to unused libraries from parent1
                    remaining_p1 = [lib for lib in p1_signed if lib not in used_libs]
                    if remaining_p1:
                        offspring_signed[idx] = random.choice(remaining_p1)
                        used_libs.add(offspring_signed[idx])
                    else:
                        # If all libraries are used (shouldn't happen with valid parents)
                        unused = list(p1_set - used_libs)
                        if unused:
                            offspring_signed[idx] = random.choice(unused)
                        else:
                            raise ValueError("Crossover failed - no available libraries")

            return offspring_signed

        # Create base signed orders
        try:
            offspring1_signed = create_offspring(parent1.signed_libraries, parent2.signed_libraries)
            offspring2_signed = create_offspring(parent2.signed_libraries, parent1.signed_libraries)

            # Create complete solutions
            def build_solution(signed_libs):
                scanned_books = set()
                scanned_per_lib = {}
                used_libs = []

                current_day = 0
                for lib in signed_libs:
                    lib_data = self.instance.libs[lib]
                    if current_day + lib_data.signup_days > self.instance.num_days:
                        continue

                    current_day += lib_data.signup_days
                    remaining_days = self.instance.num_days - current_day
                    max_books = remaining_days * lib_data.books_per_day

                    available_books = [b.id for b in lib_data.books if b.id not in scanned_books]
                    available_books.sort(key=lambda x: self.instance.scores[x], reverse=True)
                    selected = available_books[:max_books]

                    if selected:
                        scanned_books.update(selected)
                        scanned_per_lib[lib] = selected
                        used_libs.append(lib)

                built = Solution(
                    signed_libs=used_libs,
                    unsigned_libs=list(set(range(self.instance.num_libs)) - set(used_libs)),
                    scanned_books_per_library=scanned_per_lib,
                    scanned_books=scanned_books
                )

                built.calculate_fitness_score(self.instance.scores)
                return built

            return (build_solution(offspring1_signed),
                    build_solution(offspring2_signed))


        except ValueError as e:
            # Fallback to parents if crossover fails
            # print(f"Crossover failed: {e}, returning parents")
            return parent1, parent2
