import random


def _crossover_hyper(a: dict, b: dict) -> dict:
    child = {}
    for k in a:
        child[k] = a[k] if random.random() < 0.5 else b[k]
    return child


class MetaGeneticOptimizer:
    """
    Used to decide hyperparam `mutation_prob, crossover_rate, immigrant_frac` values for an instance.
    Run it on multiple instances to get the best average value distributions

    Usage:
    meta_opt = MetaGeneticOptimizer(GeneticSolver, instance, initial_solution)
    best_hyper = meta_opt.optimize()
    print("Best hyperparameters:", best_hyper)
    # run full solver with best hyper
    solver = GeneticSolver(initial_solution, instance)
    solver.mutation_prob = best_hyper['mutation_prob']
    solver.crossover_rate = best_hyper['crossover_rate']
    solver.immigrant_frac = best_hyper['immigrant_frac']
    solution = solver.solve()

    """

    def __init__(
        self,
        base_solver_cls,
        instance,
        initial_solution,
        meta_pop_size: int = 5,
        meta_generations: int = 5,
        inner_generations: int = 20,
        inner_pop_size: int = 50,
    ):
        self.base_solver_cls = base_solver_cls
        self.instance = instance
        self.initial_solution = initial_solution
        self.meta_pop_size = meta_pop_size
        self.meta_generations = meta_generations
        # hyperparam bounds
        self.bounds = {
            'mutation_prob': (0.1, 1.0),
            'crossover_rate': (0.0, 1.0),
            'immigrant_frac': (0.0, 0.3),
        }
        self.inner_generations = inner_generations
        self.inner_pop_size = inner_pop_size

    def _random_hyper(self) -> dict:
        return {
            'mutation_prob': random.uniform(*self.bounds['mutation_prob']),
            'crossover_rate': random.uniform(*self.bounds['crossover_rate']),
            'immigrant_frac': random.uniform(*self.bounds['immigrant_frac']),
        }

    def _evaluate(self, hyper: dict) -> float:
        # run a short GA with these hyperparams and return best fitness
        solver = self.base_solver_cls(
            initial_solution=self.initial_solution,
            instance=self.instance,
            population_size=self.inner_pop_size,
            generations=self.inner_generations
        )
        # inject hyperparams
        solver.mutation_prob = hyper['mutation_prob']
        solver.crossover_rate = hyper['crossover_rate']
        solver.immigrant_frac = hyper['immigrant_frac']
        best = solver.solve()
        return best.fitness_score

    def optimize(self) -> dict:
        # initialize meta-population
        meta_pop = [self._random_hyper() for _ in range(self.meta_pop_size)]
        meta_scores = [None] * self.meta_pop_size

        for mg in range(self.meta_generations):
            # evaluate all meta-individuals
            for i, hyper in enumerate(meta_pop):
                meta_scores[i] = self._evaluate(hyper)
            # sort by performance
            sorted_idx = sorted(range(len(meta_scores)), key=lambda i: meta_scores[i], reverse=True)
            meta_pop = [meta_pop[i] for i in sorted_idx]
            meta_scores = [meta_scores[i] for i in sorted_idx]
            # print progress
            print(f"Meta Gen {mg}: best inner fitness = {meta_scores[0]}")
            # keep top half, generate offspring for bottom half
            survivors = meta_pop[: self.meta_pop_size // 2]
            offspring = []
            while len(offspring) + len(survivors) < self.meta_pop_size:
                # tournament select
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                child = _crossover_hyper(parent1, parent2)
                child = self._mutate_hyper(child)
                offspring.append(child)
            meta_pop = survivors + offspring

        # return best hyper
        return meta_pop[0]

    def _mutate_hyper(self, hyper: dict) -> dict:
        # gaussian perturb
        for k, (low, high) in self.bounds.items():
            if random.random() < 0.3:
                sigma = (high - low) * 0.1
                hyper[k] += random.gauss(0, sigma)
                hyper[k] = min(max(hyper[k], low), high)
        return hyper