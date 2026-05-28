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
        "sampled_best_exchange",
    }

    DEFAULT_WEIGHTS = {
        # Base weights calibrated from operator-isolation experiments.
        # Structured-instance operators are conditional: they receive weight in
        # the full solver, but `instance_adjusted_weights` activates them only
        # when the input structure matches their assumptions.
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
        "sampled_best_exchange": 0.45,
        "coverage_exchange": 0.2,
        "paired_choice_flip": 0.1,
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
            "sampled_best_exchange",
            "coverage_exchange",
            "paired_choice_flip",
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
            ("sampled_best_exchange", Tweaks.tweak_solution_sampled_best_exchange),
            ("coverage_exchange", Tweaks.tweak_solution_coverage_exchange),
            ("paired_choice_flip", Tweaks.tweak_solution_paired_choice_flip),
        ]

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
    def instance_adjusted_weights(weights, data):
        adjusted = dict(weights or Tweaks.DEFAULT_WEIGHTS)
        coverage_weight = adjusted.get("coverage_exchange", 0.0)
        if not Tweaks._is_uniform_coverage_instance(data):
            adjusted["coverage_exchange"] = 0.0
            adjusted["paired_choice_flip"] = 0.0
            return adjusted

        paired_model = Tweaks._paired_choice_model(data)
        if paired_model is not None and adjusted.get("paired_choice_flip", 0.0) > 0.0:
            paired_target_share = 0.40
            coverage_target_share = 0.40
            other_total = sum(
                weight
                for label, weight in adjusted.items()
                if label not in {"coverage_exchange", "paired_choice_flip"}
            )
            if other_total > 0.0:
                remaining_share = 1.0 - paired_target_share - coverage_target_share
                adjusted["paired_choice_flip"] = max(
                    adjusted["paired_choice_flip"],
                    other_total * paired_target_share / remaining_share,
                )
                if coverage_weight > 0.0:
                    adjusted["coverage_exchange"] = max(
                        coverage_weight,
                        other_total * coverage_target_share / remaining_share,
                    )
            return adjusted

        adjusted["paired_choice_flip"] = 0.0
        if coverage_weight > 0.0:
            target_share = 0.55
            other_total = sum(
                weight
                for label, weight in adjusted.items()
                if label != "coverage_exchange"
            )
            if other_total <= 0.0:
                return adjusted
            adjusted["coverage_exchange"] = max(
                coverage_weight,
                other_total * target_share / (1.0 - target_share),
            )
        return adjusted

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
        if label == "sampled_best_exchange":
            return Tweaks._order_sampled_best_exchange(solution, data)
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
    def _unsigned_exchange_value(solution, data, lib_id):
        signup = data.lib_signup_days[lib_id]
        if signup >= data.num_days:
            return 0.0

        limit = min(data.lib_num_books[lib_id], max(12, data.lib_books_per_day[lib_id] * 6))
        if limit <= 0:
            return 0.0

        value = 0.0
        novel = 0
        scanned = solution.scanned_books
        for book_id in data.lib_book_ids[lib_id][:limit]:
            if book_id in scanned:
                continue
            value += data.scores[book_id] / max(1, data.book_freq[book_id])
            novel += 1

        if novel == 0:
            return 0.0

        fill_ratio = novel / max(1, limit)
        throughput_bonus = 1.0 + min(3.0, data.lib_books_per_day[lib_id] / 10.0)
        return value * (0.5 + fill_ratio) * throughput_bonus / max(1, signup)

    @staticmethod
    def _order_sampled_best_exchange(
        solution,
        data,
        signed_sample=6,
        unsigned_sample=24,
        trials=24,
    ):
        signed_count = len(solution.signed_libraries)
        unsigned = solution.unsigned_libraries
        if signed_count == 0 or not unsigned:
            return None

        order = Tweaks._clone_order(solution)
        contributions = solution.library_contributions(data)
        if contributions:
            weak_positions = [
                item["position"]
                for item in sorted(
                    contributions,
                    key=lambda item: (
                        item["score_per_signup"],
                        item["score"],
                        item["book_count"],
                    ),
                )[: min(signed_sample, len(contributions))]
            ]
        else:
            weak_positions = random.sample(
                range(signed_count),
                min(signed_sample, signed_count),
            )

        pool_size = min(len(unsigned), max(unsigned_sample, trials * 2))
        candidate_pool = random.sample(unsigned, pool_size) if len(unsigned) > pool_size else list(unsigned)
        ranked_unsigned = []
        for lib_id in candidate_pool:
            value = Tweaks._unsigned_exchange_value(solution, data, lib_id)
            if value > 0.0:
                ranked_unsigned.append((value, lib_id))
        if not ranked_unsigned:
            return None

        ranked_unsigned.sort(reverse=True)
        candidates = [lib_id for _, lib_id in ranked_unsigned[:unsigned_sample]]
        candidate_positions = {lib_id: order.index(lib_id) for lib_id in candidates}

        base_score = data.screen_evaluate_sequential(order)
        best_score = base_score
        best_order = None
        attempts = 0

        for signed_pos in weak_positions:
            for replacement_id in candidates:
                if attempts >= trials:
                    break
                attempts += 1
                replacement_pos = candidate_positions[replacement_id]
                candidate_order = order.copy()
                candidate_order[signed_pos], candidate_order[replacement_pos] = (
                    candidate_order[replacement_pos],
                    candidate_order[signed_pos],
                )
                score = data.screen_evaluate_sequential(candidate_order)
                if score > best_score:
                    best_score = score
                    best_order = candidate_order
            if attempts >= trials:
                break

        return best_order

    @staticmethod
    def _is_uniform_coverage_instance(data):
        cached = getattr(data, "_uniform_coverage_operator_cache", None)
        if cached is not None:
            return cached

        if data.num_books == 0 or data.num_libs == 0:
            result = False
        else:
            first_score = data.scores[0]
            first_signup = data.lib_signup_days[0]
            first_rate = data.lib_books_per_day[0]
            result = (
                first_score > 0
                and first_signup > 0
                and first_rate > 0
                and all(score == first_score for score in data.scores)
                and all(signup == first_signup for signup in data.lib_signup_days)
                and all(rate == first_rate for rate in data.lib_books_per_day)
            )
        setattr(data, "_uniform_coverage_operator_cache", result)
        return result

    @staticmethod
    def _paired_choice_model(data):
        cached = getattr(data, "_paired_choice_operator_cache", None)
        if cached is not None:
            return cached

        model = None
        if Tweaks._is_uniform_coverage_instance(data) and data.num_libs % 2 == 0:
            pair_count = data.num_libs // 2
            feasible_count = (data.num_days - 1) // data.lib_signup_days[0]
            if feasible_count >= pair_count:
                cvars = []
                cvals = []
                occ = [[] for _ in range(pair_count)]
                neighbors = [set() for _ in range(pair_count)]
                always_covered = 0
                represented_books = 0
                valid = True

                for book_id, libs in enumerate(data.book_libs):
                    if not libs:
                        continue
                    if len(libs) not in (2, 3):
                        valid = False
                        break

                    by_pair = {}
                    duplicate_value = False
                    for lib_id in libs:
                        pair_id = lib_id // 2
                        value = lib_id & 1
                        values = by_pair.setdefault(pair_id, set())
                        if value in values:
                            duplicate_value = True
                            break
                        values.add(value)
                    if duplicate_value:
                        valid = False
                        break

                    if any(len(values) == 2 for values in by_pair.values()):
                        if len(libs) == 2 and len(by_pair) == 1:
                            always_covered += 1
                            represented_books += 1
                            continue
                        valid = False
                        break

                    if len(by_pair) != len(libs):
                        valid = False
                        break

                    clause = len(cvars)
                    vars_ = tuple(lib_id // 2 for lib_id in libs)
                    vals = tuple(lib_id & 1 for lib_id in libs)
                    cvars.append(vars_)
                    cvals.append(vals)
                    represented_books += 1
                    for var, val in zip(vars_, vals):
                        occ[var].append((clause, val))
                    for var in vars_:
                        neighbors[var].update(vars_)

                if valid and represented_books == data.num_books and cvars:
                    always_ratio = always_covered / max(1, data.num_books)
                    avg_occ = sum(len(items) for items in occ) / max(1, pair_count)
                    if always_ratio >= 0.05 and avg_occ >= 1.0:
                        model = {
                            "pair_count": pair_count,
                            "cvars": cvars,
                            "cvals": cvals,
                            "occ": occ,
                            "neighbors": [list(items) for items in neighbors],
                            "always_covered": always_covered,
                        }

        setattr(data, "_paired_choice_operator_cache", model)
        return model

    @staticmethod
    def _coverage_score(data, cover_count):
        return sum(
            data.scores[book_id]
            for book_id, count in enumerate(cover_count)
            if count > 0
        )

    @staticmethod
    def _coverage_selected_prefix(solution):
        signed_count = len(solution.signed_libraries)
        if signed_count < 2:
            return []
        return solution.signed_libraries[: signed_count - 1]

    @staticmethod
    def _coverage_counts(data, selected):
        selected_flag = [False] * data.num_libs
        cover_count = [0] * data.num_books
        for lib_id in selected:
            selected_flag[lib_id] = True
            for book_id in data.lib_book_ids[lib_id]:
                cover_count[book_id] += 1
        return selected_flag, cover_count

    @staticmethod
    def _coverage_uncovered_value(data, lib_id, cover_count):
        value = 0
        for book_id in data.lib_book_ids[lib_id]:
            if cover_count[book_id] == 0:
                value += data.scores[book_id]
        return value

    @staticmethod
    def _coverage_unique_losses(data, selected, cover_count):
        losses = []
        for lib_id in selected:
            loss = 0
            for book_id in data.lib_book_ids[lib_id]:
                if cover_count[book_id] == 1:
                    loss += data.scores[book_id]
            losses.append((loss, lib_id))
        losses.sort()
        return losses

    @staticmethod
    def _coverage_order(data, selected, selected_flag, cover_count):
        order = sorted(
            selected,
            key=lambda lib_id: (
                data.lib_num_books[lib_id],
                data.lib_books_per_day[lib_id],
                -data.lib_signup_days[lib_id],
                lib_id,
            ),
            reverse=True,
        )
        unsigned = [
            lib_id
            for lib_id in range(data.num_libs)
            if not selected_flag[lib_id]
        ]
        unsigned.sort(
            key=lambda lib_id: (
                Tweaks._coverage_uncovered_value(data, lib_id, cover_count),
                data.lib_num_books[lib_id],
                data.lib_books_per_day[lib_id],
                -data.lib_signup_days[lib_id],
                lib_id,
            ),
            reverse=True,
        )
        order.extend(unsigned)
        return order

    @staticmethod
    def _paired_assignment(solution, data, model):
        selected = set(solution.signed_libraries)
        assignment = bytearray(model["pair_count"])
        for pair_id in range(model["pair_count"]):
            lib0 = 2 * pair_id
            lib1 = lib0 + 1
            has0 = lib0 in selected
            has1 = lib1 in selected
            if has1 and not has0:
                assignment[pair_id] = 1
            elif has0 and not has1:
                assignment[pair_id] = 0
            else:
                score0 = len(solution.scanned_books_per_library.get(lib0, []))
                score1 = len(solution.scanned_books_per_library.get(lib1, []))
                if score1 > score0:
                    assignment[pair_id] = 1
                elif score0 > score1:
                    assignment[pair_id] = 0
                else:
                    assignment[pair_id] = (
                        1 if data.lib_num_books[lib1] > data.lib_num_books[lib0] else 0
                    )
        return assignment

    @staticmethod
    def _paired_state_score(model, assignment):
        sat = bytearray(len(model["cvars"]))
        score = 0
        for clause, vars_ in enumerate(model["cvars"]):
            vals = model["cvals"][clause]
            count = 0
            for var, val in zip(vars_, vals):
                if assignment[var] == val:
                    count += 1
            sat[clause] = count
            if count:
                score += 1
        return sat, score

    @staticmethod
    def _paired_delta(model, assignment, sat, var):
        change = 0
        current = assignment[var]
        for clause, val in model["occ"][var]:
            count = sat[clause]
            if current == val:
                if count == 1:
                    change -= 1
            elif count == 0:
                change += 1
        return change

    @staticmethod
    def _paired_flip(model, assignment, sat, var):
        old = assignment[var]
        for clause, val in model["occ"][var]:
            before = sat[clause]
            after = before - 1 if old == val else before + 1
            sat[clause] = after
        assignment[var] = 1 - old

    @staticmethod
    def _paired_local_search(model, assignment, sat, max_flips):
        values = []
        positions = [-1] * model["pair_count"]

        def add(var):
            if positions[var] < 0:
                positions[var] = len(values)
                values.append(var)

        def remove(var):
            idx = positions[var]
            if idx < 0:
                return
            last = values.pop()
            if idx < len(values):
                values[idx] = last
                positions[last] = idx
            positions[var] = -1

        for var in range(model["pair_count"]):
            if Tweaks._paired_delta(model, assignment, sat, var) > 0:
                add(var)

        flips = 0
        while values and flips < max_flips:
            idx = random.randrange(len(values))
            var = values[idx]
            remove(var)
            if Tweaks._paired_delta(model, assignment, sat, var) <= 0:
                continue
            Tweaks._paired_flip(model, assignment, sat, var)
            flips += 1
            for other in model["neighbors"][var]:
                if Tweaks._paired_delta(model, assignment, sat, other) > 0:
                    add(other)
                else:
                    remove(other)
        return flips

    @staticmethod
    def _paired_order(data, assignment, model):
        selected = [
            2 * pair_id + int(assignment[pair_id])
            for pair_id in range(model["pair_count"])
        ]
        selected_flag = [False] * data.num_libs
        for lib_id in selected:
            selected_flag[lib_id] = True
        cover_count = [0] * data.num_books
        for lib_id in selected:
            for book_id in data.lib_book_ids[lib_id]:
                cover_count[book_id] += 1
        return Tweaks._coverage_order(data, selected, selected_flag, cover_count)

    @staticmethod
    def _order_paired_choice_flip(solution, data):
        model = Tweaks._paired_choice_model(data)
        if model is None:
            return None

        assignment = Tweaks._paired_assignment(solution, data, model)
        sat, current_score = Tweaks._paired_state_score(model, assignment)
        cached_best = model.get("best_assignment")
        if cached_best is not None:
            cached_assignment = bytearray(cached_best)
            cached_sat, cached_score = Tweaks._paired_state_score(model, cached_assignment)
            if cached_score >= current_score:
                assignment = cached_assignment
                sat = cached_sat
                current_score = cached_score

        pair_count = model["pair_count"]
        if pair_count >= 10000:
            first_pass_flips = 6000
            attempts = 24
            max_flips = 12000
        elif pair_count >= 1000:
            first_pass_flips = 3500
            attempts = 14
            max_flips = 7000
        else:
            first_pass_flips = max(50, min(2000, pair_count // 3))
            attempts = 6
            max_flips = max(80, min(4000, pair_count // 2))

        Tweaks._paired_local_search(
            model,
            assignment,
            sat,
            max_flips=first_pass_flips,
        )
        sat, current_score = Tweaks._paired_state_score(model, assignment)
        base_score = current_score
        best_assignment = bytearray(assignment)
        best_score = current_score

        for attempt in range(attempts):
            if cached_best is not None and attempt % 2 == 1:
                candidate = bytearray(cached_best)
            else:
                candidate = bytearray(assignment)
            candidate_sat, _candidate_score = Tweaks._paired_state_score(model, candidate)
            if attempt < attempts // 3:
                strength = random.randint(4, max(8, min(128, pair_count // 120 + 8)))
            elif attempt < 2 * attempts // 3:
                strength = random.randint(24, max(32, min(384, pair_count // 45 + 24)))
            else:
                strength = random.randint(96, max(128, min(900, pair_count // 18 + 96)))
            for _ in range(strength):
                var = random.randrange(pair_count)
                Tweaks._paired_flip(model, candidate, candidate_sat, var)
            Tweaks._paired_local_search(
                model,
                candidate,
                candidate_sat,
                max_flips=max_flips,
            )
            _, candidate_score = Tweaks._paired_state_score(model, candidate)
            if candidate_score >= current_score:
                assignment = candidate
                current_score = candidate_score
            if candidate_score > best_score:
                best_score = candidate_score
                best_assignment = candidate

        cached_best_score = model.get("best_score", -1)
        if best_score > cached_best_score:
            model["best_assignment"] = bytearray(best_assignment)
            model["best_score"] = best_score

        if best_score <= base_score:
            return None
        return Tweaks._paired_order(data, best_assignment, model)

    @staticmethod
    def _order_coverage_exchange(solution, data):
        if not Tweaks._is_uniform_coverage_instance(data):
            return None

        selected = Tweaks._coverage_selected_prefix(solution)
        selected_count = len(selected)
        if selected_count < 2 or selected_count >= data.num_libs:
            return None

        selected_flag, cover_count = Tweaks._coverage_counts(data, selected)
        base_coverage_score = Tweaks._coverage_score(data, cover_count)
        losses = Tweaks._coverage_unique_losses(data, selected, cover_count)

        max_remove = min(selected_count, max(2, min(64, selected_count // 300 + 2)))
        min_remove = min(max_remove, 1 if selected_count < 8 else 4)
        remove_count = random.randint(min_remove, max_remove)
        weak_pool_size = min(selected_count, max(remove_count * 3, remove_count))
        weak_pool = [lib_id for _loss, lib_id in losses[:weak_pool_size]]
        removed = random.sample(weak_pool, min(remove_count, len(weak_pool)))

        candidate_selected = list(selected)
        candidate_flag = selected_flag[:]
        candidate_cover = cover_count[:]
        for lib_id in removed:
            if not candidate_flag[lib_id]:
                continue
            candidate_flag[lib_id] = False
            candidate_selected.remove(lib_id)
            for book_id in data.lib_book_ids[lib_id]:
                candidate_cover[book_id] -= 1

        unsigned = [
            lib_id
            for lib_id in range(data.num_libs)
            if not candidate_flag[lib_id]
        ]
        sample_size = min(len(unsigned), max(64, remove_count * 80))
        sampled = random.sample(unsigned, sample_size) if len(unsigned) > sample_size else unsigned
        ranked_additions = []
        for lib_id in sampled:
            gain = Tweaks._coverage_uncovered_value(data, lib_id, candidate_cover)
            if gain > 0:
                ranked_additions.append((gain, data.lib_num_books[lib_id], lib_id))
        ranked_additions.sort(reverse=True)

        for _gain, _size, lib_id in ranked_additions:
            if len(candidate_selected) >= selected_count:
                break
            if candidate_flag[lib_id]:
                continue
            candidate_flag[lib_id] = True
            candidate_selected.append(lib_id)
            for book_id in data.lib_book_ids[lib_id]:
                candidate_cover[book_id] += 1

        if len(candidate_selected) < selected_count:
            for lib_id in unsigned:
                if candidate_flag[lib_id]:
                    continue
                candidate_flag[lib_id] = True
                candidate_selected.append(lib_id)
                for book_id in data.lib_book_ids[lib_id]:
                    candidate_cover[book_id] += 1
                if len(candidate_selected) >= selected_count:
                    break

        from models.coverage_search import improve_coverage_one_swap

        candidate_selected, candidate_flag, candidate_cover = improve_coverage_one_swap(
            data,
            candidate_selected,
            candidate_flag,
            candidate_cover,
            max_passes=10,
        )
        if Tweaks._coverage_score(data, candidate_cover) <= base_coverage_score:
            return None

        return Tweaks._coverage_order(
            data,
            candidate_selected,
            candidate_flag,
            candidate_cover,
        )

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
    def tweak_solution_sampled_best_exchange(solution, data):
        order = Tweaks._order_sampled_best_exchange(solution, data)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_coverage_exchange(solution, data):
        order = Tweaks._order_coverage_exchange(solution, data)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_paired_choice_flip(solution, data):
        order = Tweaks._order_paired_choice_flip(solution, data)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)

    @staticmethod
    def tweak_solution_block_reinsert(solution, data):
        order = Tweaks._order_block_reinsert(solution)
        if order is None:
            return solution
        return Tweaks._rebuilt(order, data)
