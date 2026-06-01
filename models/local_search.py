import random
import time

from models.solution import Solution
from models.tweaks import Tweaks


class LocalSearch:
    @staticmethod
    def _weighted_methods(tweak_weights=None):
        methods = Tweaks.get_tweak_methods()
        labels = [label for label, _ in methods]
        probs = [
            (tweak_weights or Tweaks.DEFAULT_WEIGHTS).get(label, 0.0)
            for label in labels
        ]
        if sum(probs) <= 0:
            raise ValueError("At least one local-search operator weight must be positive")
        return methods, probs

    @staticmethod
    def _polish_profile(data):
        total_occurrences = sum(data.lib_num_books)
        dense_small_library_instance = (
            data.num_libs <= 250 and data.num_days <= 1000
        )
        if (
            not dense_small_library_instance
            and (
            data.num_libs >= 15000
            or total_occurrences >= 700000
            or data.num_days >= 50000
            )
        ):
            return {
                "passes": 1,
                "prefix_limit": 18,
                "adjacent_checks": 48,
                "total_checks": 84,
                "move_span": 2,
                "boundary_signed": 5,
                "boundary_unsigned": 8,
                "boundary_checks": 30,
            }
        if (
            not dense_small_library_instance
            and (
            data.num_libs >= 3000
            or total_occurrences >= 300000
            or data.num_days >= 10000
            )
        ):
            return {
                "passes": 2,
                "prefix_limit": 24,
                "adjacent_checks": 72,
                "total_checks": 120,
                "move_span": 3,
                "boundary_signed": 6,
                "boundary_unsigned": 10,
                "boundary_checks": 48,
            }
        if (
            data.num_libs >= 500
            or total_occurrences >= 100000
            or data.num_days >= 1000
        ):
            return {
                "passes": 2,
                "prefix_limit": 32,
                "adjacent_checks": 108,
                "total_checks": 180,
                "move_span": 3,
                "boundary_signed": 8,
                "boundary_unsigned": 12,
                "boundary_checks": 72,
            }
        return {
            "passes": 2,
            "prefix_limit": 40,
            "adjacent_checks": 132,
            "total_checks": 220,
            "move_span": 4,
            "boundary_signed": 10,
            "boundary_unsigned": 14,
            "boundary_checks": 96,
        }

    @staticmethod
    def local_search(
        solution,
        data,
        time_limit=60.0,
        max_iterations=1000,
        no_improve_limit=250,
        tweak_weights=None,
    ):
        start_time = time.time()
        current_solution = solution.clone()
        best_solution = current_solution.clone()
        current_proxy_score = data.screen_evaluate_sequential(
            current_solution.ordered_libraries()
        )
        iterations = 0
        stagnant = 0
        methods, probs = LocalSearch._weighted_methods(tweak_weights)

        while (time.time() - start_time < time_limit) and (iterations < max_iterations):
            label, tweak_method = random.choices(methods, weights=probs, k=1)[0]
            new_solution = None

            if label in Tweaks.FAST_ORDER_OPERATORS:
                order = Tweaks.build_candidate_order(label, current_solution, data)
                if order is not None:
                    score = data.screen_evaluate_sequential(order)
                    if score > current_proxy_score:
                        new_solution = Solution.from_order(order, data)
            else:
                new_solution = tweak_method(current_solution, data)

            if new_solution is not None:
                current_solution = new_solution
                current_proxy_score = data.screen_evaluate_sequential(
                    current_solution.ordered_libraries()
                )
                stagnant = 0
                if current_solution.fitness_score > best_solution.fitness_score:
                    best_solution = current_solution.clone()
            else:
                stagnant += 1
                if stagnant >= no_improve_limit:
                    break
            iterations += 1

        deadline = start_time + time_limit
        if time.time() < deadline:
            best_solution = LocalSearch._polish_solution(best_solution, data, deadline)

        return best_solution

    @staticmethod
    def _polish_solution(solution, data, deadline):
        current = solution
        profile = LocalSearch._polish_profile(data)
        passes = 0
        while passes < profile["passes"] and time.time() < deadline:
            passes += 1
            signed_count = len(current.signed_libraries)
            prefix_limit = min(signed_count, profile["prefix_limit"])
            if prefix_limit < 2:
                break

            base_order = current.ordered_libraries()
            best_candidate = current
            checks = 0

            for i in range(prefix_limit - 1):
                if time.time() >= deadline or checks >= profile["adjacent_checks"]:
                    break
                order = base_order.copy()
                order[i], order[i + 1] = order[i + 1], order[i]
                checks += 1
                if data.screen_evaluate(order) > best_candidate.fitness_score:
                    candidate = Solution.from_order(order, data)
                    if candidate.fitness_score > best_candidate.fitness_score:
                        best_candidate = candidate

            move_span = profile["move_span"]
            for i in range(prefix_limit):
                if time.time() >= deadline or checks >= profile["total_checks"]:
                    break
                left = max(0, i - move_span)
                right = min(prefix_limit - 1, i + move_span)
                for j in range(left, right + 1):
                    if i == j or time.time() >= deadline or checks >= profile["total_checks"]:
                        continue
                    order = base_order.copy()
                    lib_id = order.pop(i)
                    order.insert(j, lib_id)
                    checks += 1
                    if data.screen_evaluate(order) > best_candidate.fitness_score:
                        candidate = Solution.from_order(order, data)
                        if candidate.fitness_score > best_candidate.fitness_score:
                            best_candidate = candidate

            # Intensify the boundary between signed and unsigned libraries,
            # where small changes often determine the final feasible prefix.
            signed_tail = min(signed_count, profile["boundary_signed"])
            unsigned_limit = min(len(base_order) - signed_count, profile["boundary_unsigned"])
            boundary_checks = 0
            if signed_tail > 0 and unsigned_limit > 0 and time.time() < deadline:
                signed_start = signed_count - signed_tail
                unsigned_end = signed_count + unsigned_limit
                for i in range(signed_start, signed_count):
                    if time.time() >= deadline or boundary_checks >= profile["boundary_checks"]:
                        break
                    for j in range(signed_count, unsigned_end):
                        if time.time() >= deadline or boundary_checks >= profile["boundary_checks"]:
                            break
                        order = base_order.copy()
                        order[i], order[j] = order[j], order[i]
                        boundary_checks += 1
                        if data.screen_evaluate(order) > best_candidate.fitness_score:
                            candidate = Solution.from_order(order, data)
                            if candidate.fitness_score > best_candidate.fitness_score:
                                best_candidate = candidate

            if best_candidate.fitness_score > current.fitness_score:
                current = best_candidate
            else:
                break

        return current
