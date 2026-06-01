import random
import time

from models.solution import Solution
from models.tweaks import Tweaks


class LocalSearch:
    # The local search has two scientifically distinct stages:
    # 1) proxy-guided exploration, where many cheap neighbors are sampled;
    # 2) exact refinement, where remaining time is spent with the full decoder.
    # The exact phase always receives at least this tail of the local-search
    # budget; if proxy search stagnates earlier, it receives more.
    EXACT_REFINEMENT_MIN_SHARE = 0.15

    @staticmethod
    def _unsigned_frontier_size(signed_count, unsigned_count):
        return min(unsigned_count, max(64, 4 * LocalSearch._frontier_size(signed_count)))

    @staticmethod
    def _screen_score(data, order):
        return data.screen_evaluate_sequential(order)

    @staticmethod
    def _weighted_methods(tweak_weights=None, data=None):
        methods = Tweaks.get_tweak_methods()
        weights = tweak_weights or Tweaks.DEFAULT_WEIGHTS
        if data is not None:
            weights = Tweaks.instance_adjusted_weights(weights, data)
        labels = [label for label, _ in methods]
        probs = [weights.get(label, 0.0) for label in labels]
        if sum(probs) <= 0:
            raise ValueError("At least one local-search operator weight must be positive")
        return methods, probs

    @staticmethod
    def _stats_entry(operator_stats, label):
        if operator_stats is None:
            return None
        return operator_stats.setdefault(
            label,
            {
                "attempts": 0,
                "proxy_improved": 0,
                "exact_checked": 0,
                "accepted": 0,
                "rejected": 0,
            },
        )

    @staticmethod
    def local_search(
        solution,
        data,
        time_limit=60.0,
        no_improve_limit=250,
        tweak_weights=None,
        operator_stats=None,
        use_proxy=True,
    ):
        start_time = time.time()
        deadline = start_time + time_limit
        exploration_deadline = start_time + (
            time_limit * (1.0 - LocalSearch.EXACT_REFINEMENT_MIN_SHARE)
        )
        current_solution = solution.clone()
        best_solution = current_solution.clone()
        current_proxy_score = (
            LocalSearch._screen_score(data, current_solution.ordered_libraries())
            if use_proxy
            else 0
        )
        stagnant = 0
        methods, probs = LocalSearch._weighted_methods(tweak_weights, data)

        while time.time() < exploration_deadline:
            label, tweak_method = random.choices(methods, weights=probs, k=1)[0]
            stats = LocalSearch._stats_entry(operator_stats, label)
            if stats is not None:
                stats["attempts"] += 1

            new_solution = None
            new_proxy_score = None
            if label in Tweaks.FAST_ORDER_OPERATORS:
                order = Tweaks.build_candidate_order(label, current_solution, data)
                if order is not None:
                    if use_proxy:
                        new_solution, new_proxy_score = LocalSearch._try_proxy_move(
                            order,
                            data,
                            current_solution,
                            current_proxy_score,
                            stats,
                        )
                    else:
                        new_solution = LocalSearch._try_exact_move(
                            order,
                            data,
                            current_solution,
                            stats,
                        )
            else:
                candidate = tweak_method(current_solution, data)
                if (
                    candidate is not None
                    and candidate.fitness_score > current_solution.fitness_score
                ):
                    new_solution = candidate

            if new_solution is not None:
                current_solution = new_solution
                if use_proxy:
                    current_proxy_score = (
                        new_proxy_score
                        if new_proxy_score is not None
                        else LocalSearch._screen_score(
                            data,
                            current_solution.ordered_libraries(),
                        )
                    )
                stagnant = 0
                if stats is not None:
                    stats["accepted"] += 1
                if current_solution.fitness_score > best_solution.fitness_score:
                    best_solution = current_solution.clone()
            else:
                if stats is not None:
                    stats["rejected"] += 1
                stagnant += 1
                if stagnant >= no_improve_limit:
                    break

        if time.time() < deadline:
            best_solution = LocalSearch._exact_refinement(best_solution, data, deadline)

        return best_solution

    @staticmethod
    def _try_exact_move(order, data, current_solution, stats):
        if stats is not None:
            stats["exact_checked"] += 1
        exact_score = data.fast_evaluate(order)
        if exact_score <= current_solution.fitness_score:
            return None
        candidate = Solution.from_order(order, data)
        if candidate.fitness_score <= current_solution.fitness_score:
            return None
        return candidate

    @staticmethod
    def _try_proxy_move(order, data, current_solution, current_proxy_score, stats):
        proxy_score = LocalSearch._screen_score(data, order)
        proxy_improved = proxy_score > current_proxy_score
        if proxy_score < current_proxy_score:
            return None, None

        if stats is not None:
            if proxy_improved:
                stats["proxy_improved"] += 1
            stats["exact_checked"] += 1

        exact_score = data.fast_evaluate(order)
        if exact_score <= current_solution.fitness_score:
            return None, None

        candidate = Solution.from_order(order, data)
        if candidate.fitness_score <= current_solution.fitness_score:
            return None, None
        return candidate, proxy_score

    @staticmethod
    def _exact_refinement(solution, data, deadline):
        """
        Unified exact refinement with the full decoder.

        This phase spends the remaining budget on exact-scored moves only:
        sampled order-neighborhood moves for broader improvement, followed by
        deterministic prefix and boundary checks for final cleanup.
        """
        current = solution
        while time.time() < deadline:
            start_score = current.fitness_score
            current = LocalSearch._sampled_exact_pass(
                current,
                data,
                deadline,
            )
            if time.time() < deadline:
                current = LocalSearch._deterministic_exact_polish(
                    current,
                    data,
                    deadline,
                )
            if current.fitness_score <= start_score:
                break
        return current

    @staticmethod
    def _sampled_exact_pass(solution, data, deadline):
        base_order = solution.ordered_libraries()
        signed_count = len(solution.signed_libraries)
        if signed_count == 0 or len(base_order) < 2:
            return solution

        best_order = None
        best_score = solution.fitness_score
        sample_count = max(16, 2 * LocalSearch._frontier_size(signed_count))
        for _ in range(sample_count):
            if time.time() >= deadline:
                break
            order = LocalSearch._sample_order_neighbor(base_order, signed_count)
            if order is None:
                continue
            score = data.fast_evaluate(order)
            if score > best_score:
                best_score = score
                best_order = order

        if best_order is None:
            return solution
        candidate = Solution.from_order(best_order, data)
        if candidate.fitness_score <= solution.fitness_score:
            return solution
        return candidate

    @staticmethod
    def _sample_order_neighbor(base_order, signed_count):
        order_len = len(base_order)
        unsigned_count = order_len - signed_count
        if signed_count <= 0:
            return None

        order = base_order.copy()
        move = random.random()

        if move < 0.40 and unsigned_count > 0:
            signed_idx = random.randrange(signed_count)
            unsigned_frontier = LocalSearch._unsigned_frontier_size(
                signed_count,
                unsigned_count,
            )
            unsigned_idx = signed_count + random.randrange(unsigned_frontier)
            order[signed_idx], order[unsigned_idx] = order[unsigned_idx], order[signed_idx]
            return order

        if move < 0.55 and signed_count >= 2:
            i, j = random.sample(range(signed_count), 2)
            order[i], order[j] = order[j], order[i]
            return order

        if move < 0.70 and signed_count >= 2:
            i, j = random.sample(range(signed_count), 2)
            lib_id = order.pop(i)
            order.insert(j, lib_id)
            return order

        if move < 0.82 and unsigned_count > 0:
            unsigned_frontier = LocalSearch._unsigned_frontier_size(
                signed_count,
                unsigned_count,
            )
            unsigned_idx = signed_count + random.randrange(unsigned_frontier)
            lib_id = order.pop(unsigned_idx)
            order.insert(random.randrange(signed_count + 1), lib_id)
            return order

        if move < 0.92 and signed_count >= 3:
            max_segment = min(signed_count, max(2, int(signed_count ** 0.5)))
            segment = random.randint(2, max_segment)
            start = random.randrange(0, signed_count - segment + 1)
            end = start + segment
            order[start:end] = reversed(order[start:end])
            return order

        if signed_count > 0:
            idx = random.randrange(signed_count)
            lib_id = order.pop(idx)
            order.append(lib_id)
            return order

        return None

    @staticmethod
    def _deterministic_exact_polish(solution, data, deadline):
        current = solution
        while time.time() < deadline:
            signed_count = len(current.signed_libraries)
            if signed_count < 2:
                break

            base_order = current.ordered_libraries()
            front = LocalSearch._frontier_size(signed_count)
            best_candidate = current

            best_candidate = LocalSearch._polish_adjacent_prefix(
                best_candidate,
                base_order,
                data,
                front,
                deadline,
            )
            if time.time() < deadline:
                best_candidate = LocalSearch._polish_prefix_moves(
                    best_candidate,
                    base_order,
                    data,
                    front,
                    deadline,
                )
            if time.time() < deadline:
                best_candidate = LocalSearch._polish_boundary(
                    best_candidate,
                    base_order,
                    data,
                    deadline,
                )

            if best_candidate.fitness_score <= current.fitness_score:
                break
            current = best_candidate
        return current

    @staticmethod
    def _frontier_size(signed_count):
        return min(signed_count, max(12, 2 * int(signed_count ** 0.5)))

    @staticmethod
    def _polish_adjacent_prefix(current, base_order, data, front, deadline):
        best_candidate = current
        for i in range(front - 1):
            if time.time() >= deadline:
                break
            order = base_order.copy()
            order[i], order[i + 1] = order[i + 1], order[i]
            if data.fast_evaluate(order) > best_candidate.fitness_score:
                candidate = Solution.from_order(order, data)
                if candidate.fitness_score > best_candidate.fitness_score:
                    best_candidate = candidate
        return best_candidate

    @staticmethod
    def _polish_prefix_moves(current, base_order, data, front, deadline):
        best_candidate = current
        radius = max(2, int(front ** 0.5))
        for i in range(front):
            if time.time() >= deadline:
                break
            left = max(0, i - radius)
            right = min(front - 1, i + radius)
            for j in range(left, right + 1):
                if i == j or time.time() >= deadline:
                    continue
                order = base_order.copy()
                lib_id = order.pop(i)
                order.insert(j, lib_id)
                if data.fast_evaluate(order) > best_candidate.fitness_score:
                    candidate = Solution.from_order(order, data)
                    if candidate.fitness_score > best_candidate.fitness_score:
                        best_candidate = candidate
        return best_candidate

    @staticmethod
    def _polish_boundary(current, base_order, data, deadline):
        best_candidate = current
        signed_count = len(current.signed_libraries)
        unsigned_count = len(base_order) - signed_count
        boundary = min(signed_count, unsigned_count, max(8, int(signed_count ** 0.5)))
        if boundary <= 0:
            return best_candidate

        signed_start = signed_count - boundary
        unsigned_end = signed_count + boundary
        for i in range(signed_start, signed_count):
            if time.time() >= deadline:
                break
            for j in range(signed_count, unsigned_end):
                if time.time() >= deadline:
                    break
                order = base_order.copy()
                order[i], order[j] = order[j], order[i]
                if data.fast_evaluate(order) > best_candidate.fitness_score:
                    candidate = Solution.from_order(order, data)
                    if candidate.fitness_score > best_candidate.fitness_score:
                        best_candidate = candidate
        return best_candidate
