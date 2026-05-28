import random


class SelectionStrategies:
    WEIGHTS = {
        'tournament': 3.0,
        'roulette': 2.0,
        'rank': 1.0
    }

    @staticmethod
    def get_selection_methods(selection_weights=None):
        selection_weights = selection_weights or {}
        return [
            (
                SelectionStrategies.tournament_selection,
                max(0.0, selection_weights.get('tournament', SelectionStrategies.WEIGHTS['tournament'])),
            ),
            (
                SelectionStrategies.roulette_wheel_selection,
                max(0.0, selection_weights.get('roulette', SelectionStrategies.WEIGHTS['roulette'])),
            ),
            (
                SelectionStrategies.rank_selection,
                max(0.0, selection_weights.get('rank', SelectionStrategies.WEIGHTS['rank'])),
            ),
        ]

    @staticmethod
    def choose_selection_method(selection_weights=None):
        methods, weights = zip(*SelectionStrategies.get_selection_methods(selection_weights))
        if sum(weights) <= 0.0:
            weights = (
                SelectionStrategies.WEIGHTS['tournament'],
                SelectionStrategies.WEIGHTS['roulette'],
                SelectionStrategies.WEIGHTS['rank'],
            )
        return random.choices(methods, weights=weights, k=1)[0]

    @staticmethod
    def tournament_selection(population, k=10):
        """Select one individual using tournament selection."""
        k = min(k, len(population))
        tournament = random.sample(population, k)
        return max(tournament, key=lambda ind: ind.fitness_score)

    @staticmethod
    def roulette_wheel_selection(population):
        """Select one individual using roulette wheel selection."""
        total_fitness = sum(ind.fitness_score for ind in population)
        if total_fitness == 0:
            return random.choice(population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind in population:
            current += ind.fitness_score
            if current > pick:
                return ind
        return population[-1]  # fallback

    @staticmethod
    def rank_selection(population):
        """Select one individual using rank-based selection."""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness_score)
        ranks = list(range(1, len(population) + 1))
        total_rank = sum(ranks)
        pick = random.uniform(0, total_rank)
        current = 0
        for ind, rank in zip(sorted_pop, ranks):
            current += rank
            if current > pick:
                return ind
        return sorted_pop[-1]  # fallback
