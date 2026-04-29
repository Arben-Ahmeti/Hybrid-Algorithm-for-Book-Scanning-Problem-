import random

from models.solution import Solution


class Tweaks:
    FAST_ORDER_OPERATORS = {
        "remove_library",
        "move_signed",
        "swap_signed",
        "block_reinsert",
        "swap_signed_with_unsigned",
        "reverse_segment",
        "insert_library",
        "swap_neighbor_libraries",
        "critical_path_insert",
        "diversity_swap",
    }

    DEFAULT_WEIGHTS = {
        # Weights calibrated from operator-isolation experiments on a
        # five-instance suite together with the `e_so_many_books` instance.
        "remove_library": 2.8,
        "move_signed": 2.4,
        "swap_signed": 2.2,
        "block_reinsert": 1.9,
        "swap_signed_with_unsigned": 1.6,
        "reverse_segment": 1.3,
        "insert_library": 1.1,
        "swap_neighbor_libraries": 0.8,
        "critical_path_insert": 0.7,
        "diversity_swap": 0.6,
    }
    GROUPS = {
        "order": {
            "swap_signed",
            "move_signed",
            "swap_neighbor_libraries",
            "reverse_segment",
            "block_reinsert",
        },
        "insert": {
            "swap_signed_with_unsigned",
            "insert_library",
            "critical_path_insert",
            "remove_library",
        },
        "strategic": {
            "diversity_swap",
        },
    }

    @staticmethod
    def operator_labels():
        return [label for label, _ in Tweaks.get_tweak_methods()]

    @staticmethod
    def get_tweak_methods():
        return [
            ("remove_library", Tweaks.tweak_solution_remove_library),
            ("move_signed", Tweaks.tweak_solution_move_signed),
            ("swap_signed", Tweaks.tweak_solution_swap_signed),
            ("block_reinsert", Tweaks.tweak_solution_block_reinsert),
            ("swap_signed_with_unsigned", Tweaks.tweak_solution_swap_signed_with_unsigned),
            ("reverse_segment", Tweaks.tweak_solution_reverse_segment),
            ("insert_library", Tweaks.tweak_solution_insert_library),
            ("swap_neighbor_libraries", Tweaks.tweak_solution_swap_neighbor_libraries),
            ("critical_path_insert", Tweaks.tweak_solution_critical_path_insert),
            ("diversity_swap", Tweaks.tweak_solution_diversity_swap),
        ]

    @staticmethod
    def choose_tweak_method(weights=None):
        weights = weights or Tweaks.DEFAULT_WEIGHTS
        methods = Tweaks.get_tweak_methods()
        labels = [label for label, _ in methods]
        funcs = [func for _, func in methods]
        probs = [weights.get(label, 0.0) for label in labels]
        if sum(probs) <= 0:
            raise ValueError("At least one local-search operator weight must be positive")
        return random.choices(funcs, weights=probs, k=1)[0]

    @staticmethod
    def grouped_weights(order_scale=1.0, insert_scale=1.0, strategic_scale=1.0):
        scales = {
            "order": order_scale,
            "insert": insert_scale,
            "strategic": strategic_scale,
        }
        weights = {}
        for label, base in Tweaks.DEFAULT_WEIGHTS.items():
            scale = 1.0
            for group_name, members in Tweaks.GROUPS.items():
                if label in members:
                    scale = scales[group_name]
                    break
            weights[label] = max(0.0, base * scale)
        return weights

    @staticmethod
    def build_weights(
        order_scale=1.0,
        insert_scale=1.0,
        strategic_scale=1.0,
        overrides=None,
        enabled=None,
    ):
        weights = Tweaks.grouped_weights(
            order_scale=order_scale,
            insert_scale=insert_scale,
            strategic_scale=strategic_scale,
        )
        if overrides:
            for label, value in overrides.items():
                if label in weights and value is not None:
                    weights[label] = max(0.0, float(value))
        if enabled is not None:
            enabled_set = set(enabled)
            for label in list(weights.keys()):
                if label not in enabled_set:
                    weights[label] = 0.0
        return weights

    @staticmethod
    def _clone_order(solution):
        return solution.signed_libraries + solution.unsigned_libraries

    @staticmethod
    def _rebuilt(order, data):
        return Solution.from_order(order, data)

    @staticmethod
    def build_candidate_order(label, solution, data):
        if label == "swap_signed":
            return Tweaks._order_swap_signed(solution)
        if label == "swap_signed_with_unsigned":
            return Tweaks._order_swap_signed_with_unsigned(solution)
        if label == "move_signed":
            return Tweaks._order_move_signed(solution)
        if label == "swap_neighbor_libraries":
            return Tweaks._order_swap_neighbor_libraries(solution)
        if label == "insert_library":
            return Tweaks._order_insert_library(solution)
        if label == "critical_path_insert":
            return Tweaks._order_critical_path_insert(solution, data)
        if label == "remove_library":
            return Tweaks._order_remove_library(solution)
        if label == "reverse_segment":
            return Tweaks._order_reverse_segment(solution)
        if label == "block_reinsert":
            return Tweaks._order_block_reinsert(solution)
        if label == "diversity_swap":
            return Tweaks._order_diversity_swap(solution, data)
        return None

    @staticmethod
    def _order_swap_signed(solution):
        if len(solution.signed_libraries) < 2:
            return None
        order = Tweaks._clone_order(solution)
        i, j = random.sample(range(len(solution.signed_libraries)), 2)
        order[i], order[j] = order[j], order[i]
        return order

    @staticmethod
    def _order_swap_signed_with_unsigned(solution, bias_type=None, bias_ratio=2 / 3):
        signed_count = len(solution.signed_libraries)
        if signed_count == 0 or not solution.unsigned_libraries:
            return None

        order = Tweaks._clone_order(solution)
        if bias_type == "favor_first_half" and signed_count > 1 and random.random() < bias_ratio:
            signed_idx = random.randint(0, max(0, signed_count // 2 - 1))
        elif bias_type == "favor_second_half" and signed_count > 1 and random.random() < bias_ratio:
            signed_idx = random.randint(signed_count // 2, signed_count - 1)
        else:
            signed_idx = random.randint(0, signed_count - 1)

        unsigned_idx = random.randint(signed_count, len(order) - 1)
        order[signed_idx], order[unsigned_idx] = order[unsigned_idx], order[signed_idx]
        return order

    @staticmethod
    def _order_move_signed(solution):
        if len(solution.signed_libraries) < 2:
            return None
        order = Tweaks._clone_order(solution)
        signed_count = len(solution.signed_libraries)
        i = random.randint(0, signed_count - 1)
        j = random.randint(0, signed_count - 1)
        if i == j:
            return None
        lib_id = order.pop(i)
        order.insert(j, lib_id)
        return order

    @staticmethod
    def _order_swap_neighbor_libraries(solution):
        if len(solution.signed_libraries) < 2:
            return None
        order = Tweaks._clone_order(solution)
        pos = random.randint(0, len(solution.signed_libraries) - 2)
        order[pos], order[pos + 1] = order[pos + 1], order[pos]
        return order

    @staticmethod
    def _order_insert_library(solution):
        signed_count = len(solution.signed_libraries)
        if not solution.unsigned_libraries:
            return None
        order = Tweaks._clone_order(solution)
        unsigned_idx = random.randint(signed_count, len(order) - 1)
        lib_id = order.pop(unsigned_idx)
        insert_pos = random.randint(0, signed_count)
        order.insert(insert_pos, lib_id)
        return order

    @staticmethod
    def _order_critical_path_insert(solution, data):
        signed_count = len(solution.signed_libraries)
        unsigned = solution.unsigned_libraries
        if not unsigned:
            return None

        candidate_count = min(12, len(unsigned))
        fastest = sorted(unsigned, key=lambda lid: data.lib_signup_days[lid])[:candidate_count]
        if not fastest:
            return None

        best_lib = None
        best_value = None
        for lib_id in fastest:
            signup = data.lib_signup_days[lib_id]
            if signup >= data.num_days:
                continue
            limit = min(data.lib_num_books[lib_id], max(12, data.lib_books_per_day[lib_id] * 6))
            value = 0.0
            for book_id in data.lib_book_ids[lib_id][:limit]:
                value += data.scores[book_id] / max(1, data.book_freq[book_id])
            value /= max(1, signup)
            if best_value is None or value > best_value:
                best_value = value
                best_lib = lib_id

        if best_lib is None:
            return None

        order = Tweaks._clone_order(solution)
        order.remove(best_lib)
        early_limit = max(1, min(signed_count, max(1, signed_count // 3)))
        insert_pos = random.randint(0, early_limit)
        order.insert(insert_pos, best_lib)
        return order

    @staticmethod
    def _order_remove_library(solution):
        if len(solution.signed_libraries) == 0:
            return None
        order = Tweaks._clone_order(solution)
        signed_count = len(solution.signed_libraries)
        idx = random.randint(0, signed_count - 1)
        lib_id = order.pop(idx)
        order.append(lib_id)
        return order

    @staticmethod
    def _order_reverse_segment(solution):
        if len(solution.signed_libraries) < 3:
            return None
        order = Tweaks._clone_order(solution)
        signed_count = len(solution.signed_libraries)
        i, j = sorted(random.sample(range(signed_count), 2))
        order[i : j + 1] = reversed(order[i : j + 1])
        return order

    @staticmethod
    def _order_block_reinsert(solution):
        signed_count = len(solution.signed_libraries)
        if signed_count < 3:
            return None

        order = Tweaks._clone_order(solution)
        block_size = min(max(2, signed_count // 10), 8)
        start = random.randint(0, max(0, signed_count - block_size))
        block = order[start : start + block_size]
        del order[start : start + block_size]

        insert_at = random.randint(0, max(0, len(solution.signed_libraries) - block_size))
        order[insert_at:insert_at] = block
        return order

    @staticmethod
    def _order_diversity_swap(solution, data):
        signed = solution.signed_libraries
        unsigned = solution.unsigned_libraries
        if not signed or not unsigned:
            return None

        contributions = solution.library_contributions(data)
        if not contributions:
            return None

        worst = min(
            contributions,
            key=lambda item: (item["score"], item["score_per_signup"], item["book_count"]),
        )
        scanned = solution.scanned_books
        pool_size = min(24, len(unsigned))
        candidate_pool = random.sample(unsigned, pool_size) if len(unsigned) > pool_size else unsigned

        best_candidate = None
        best_value = None
        for lib_id in candidate_pool:
            signup = data.lib_signup_days[lib_id]
            if signup >= data.num_days:
                continue
            limit = min(data.lib_num_books[lib_id], max(12, data.lib_books_per_day[lib_id] * 6))
            value = 0.0
            novel = 0
            for book_id in data.lib_book_ids[lib_id][:limit]:
                if book_id in scanned:
                    continue
                value += data.scores[book_id] / max(1, data.book_freq[book_id])
                novel += 1
            if novel == 0:
                continue
            value /= max(1, signup)
            if best_value is None or value > best_value:
                best_value = value
                best_candidate = lib_id

        if best_candidate is None:
            return None

        order = Tweaks._clone_order(solution)
        i = worst["position"]
        j = order.index(best_candidate)
        order[i], order[j] = order[j], order[i]
        return order

    @staticmethod
    def tweak_solution_swap_signed(solution, data):
        order = Tweaks._order_swap_signed(solution)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_swap_signed_with_unsigned(solution, data, bias_type=None, bias_ratio=2 / 3):
        order = Tweaks._order_swap_signed_with_unsigned(
            solution,
            bias_type=bias_type,
            bias_ratio=bias_ratio,
        )
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_move_signed(solution, data):
        order = Tweaks._order_move_signed(solution)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_swap_neighbor_libraries(solution, data):
        order = Tweaks._order_swap_neighbor_libraries(solution)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_insert_library(solution, data):
        order = Tweaks._order_insert_library(solution)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_critical_path_insert(solution, data):
        order = Tweaks._order_critical_path_insert(solution, data)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_remove_library(solution, data):
        order = Tweaks._order_remove_library(solution)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_reverse_segment(solution, data):
        order = Tweaks._order_reverse_segment(solution)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_diversity_swap(solution, data):
        order = Tweaks._order_diversity_swap(solution, data)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_block_reinsert(solution, data):
        order = Tweaks._order_block_reinsert(solution)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)
