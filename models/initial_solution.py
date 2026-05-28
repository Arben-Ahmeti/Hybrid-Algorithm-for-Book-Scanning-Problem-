import heapq
import random
import time

from models.solution import Solution


class InitialSolution:
    @staticmethod
    def generate_initial_solution(
        data,
        max_time=120.0,
        alphas=None,
        beta=0.12,
        grasp_rcl=0.05,
        grasp_max_time=5.0,
        verbose=True,
        refine_count=8,
    ):
        if hasattr(data, "to_flat_arrays"):
            from models.evaluation import warmup_jit

            warmup_jit(data.to_flat_arrays())
        start_time = time.time()
        alphas = alphas or [0.5, 1.0, 1.5, 2.0]
        screened_candidates = []

        if verbose:
            print("\nGenerating Initial Solutions...")

        constructors = InitialSolution._build_constructor_plan(
            data=data,
            alphas=alphas,
            beta=beta,
        )

        for label, method, kwargs in constructors:
            if time.time() - start_time >= max_time:
                break
            if verbose:
                print(f"  Running {label}...")
            order = method(data, **kwargs)
            approx_score = data.fast_evaluate(order)
            screened_candidates.append((approx_score, label, order))
            if verbose:
                print(f"    Approx score: {approx_score:,}")

        elapsed = time.time() - start_time
        remaining_for_grasp = max(0.0, max_time - elapsed)
        grasp_time = min(max(0.0, grasp_max_time), remaining_for_grasp)
        if grasp_time > 0.0:
            grasp_label = f"GRASP rcl={grasp_rcl:.2f} time={grasp_time:.1f}s"
            if verbose:
                print(f"  Running {grasp_label}...")
            grasp_order, approx_score = InitialSolution._build_best_grasp_order(
                data,
                rcl_ratio=grasp_rcl,
                max_time=grasp_time,
            )
            screened_candidates.append((approx_score, grasp_label, grasp_order))
            if verbose:
                print(f"    Approx score: {approx_score:,}")

        if not screened_candidates:
            fallback_order = InitialSolution._build_order_sorted(data)
            screened_candidates.append((data.fast_evaluate(fallback_order), "Fallback Sorted", fallback_order))

        screened_candidates.sort(key=lambda item: item[0], reverse=True)
        refine_count = min(max(1, refine_count), len(screened_candidates))
        candidates = []
        for _, label, order in screened_candidates[:refine_count]:
            solution = Solution.from_order(order, data)
            candidates.append((solution.fitness_score, label, solution))
            if verbose:
                print(f"    Exact score: {solution.fitness_score:,} ({label})")

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_label, best_solution = candidates[0]

        if verbose:
            print(f"Best Initial Score: {best_score:,} ({best_label})")

        return best_solution, candidates

    @staticmethod
    def generate_initial_solutions(
        data,
        max_time=120.0,
        alphas=None,
        beta=0.12,
        grasp_rcl=0.05,
        grasp_max_time=5.0,
        verbose=True,
        refine_count=8,
    ):
        _best_solution, candidates = InitialSolution.generate_initial_solution(
            data,
            max_time=max_time,
            alphas=alphas,
            beta=beta,
            grasp_rcl=grasp_rcl,
            grasp_max_time=grasp_max_time,
            verbose=verbose,
            refine_count=refine_count,
        )
        return [solution for _score, _label, solution in candidates]

    @staticmethod
    def generate_adaptive_restart_solution(data, alphas=None):
        alphas = alphas or [0.5, 1.0, 1.5, 2.0]
        alpha = random.choice(alphas)
        noise = random.uniform(0.03, 0.10)
        if random.random() < 0.25:
            rarity_weighted = random.random() < 0.5
            beta = random.choice([10.0, 20.0, 30.0])
            gamma = random.choice([1.5, 2.0, 2.5])
            order = InitialSolution._build_order_fill_ratio_heap(
                data,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                rarity_weighted=rarity_weighted,
                noise=noise,
            )
            mode = "fill_rare" if rarity_weighted else "fill_raw"
            return (
                Solution.from_order(order, data),
                f"fresh:{mode}/a={alpha}/b={beta:.0f}/g={gamma:.1f}/n={noise:.2f}",
            )

        potential_mode = random.choice(["top_raw", "top_rare", "cap_raw", "cap_rare"])
        order = InitialSolution._build_order_adaptive_heap(
            data,
            alpha=alpha,
            potential_mode=potential_mode,
            noise=noise,
        )
        return (
            Solution.from_order(order, data),
            f"fresh:{potential_mode}/a={alpha}/n={noise:.2f}",
        )

    @staticmethod
    def _build_order_sorted(data):
        return sorted(
            range(data.num_libs),
            key=lambda lib_id: (
                data.lib_signup_days[lib_id],
                -sum(data.scores[book_id] for book_id in data.lib_book_ids[lib_id]),
            ),
        )

    @staticmethod
    def _build_order_greedy_heap(
        data,
        alpha=1.0,
        capacity_aware=False,
        rarity_weighted=False,
        noise=0.0,
        beta=0.0,
    ):
        heap = []
        used_books = set()
        remaining = set(range(data.num_libs))
        order = []
        day = 0

        for lib_id in remaining:
            score = InitialSolution._library_value(
                data,
                lib_id,
                day,
                used_books,
                alpha=alpha,
                capacity_aware=capacity_aware,
                rarity_weighted=rarity_weighted,
                noise=noise,
                position_penalty_beta=beta,
                used_count=len(order),
            )
            heapq.heappush(heap, (-score, lib_id, day))

        while heap and day < data.num_days:
            neg_score, lib_id, scored_day = heapq.heappop(heap)
            if lib_id not in remaining:
                continue

            if scored_day != day:
                refreshed = InitialSolution._library_value(
                    data,
                    lib_id,
                    day,
                    used_books,
                    alpha=alpha,
                    capacity_aware=capacity_aware,
                    rarity_weighted=rarity_weighted,
                    noise=noise,
                    position_penalty_beta=beta,
                    used_count=len(order),
                )
                heapq.heappush(heap, (-refreshed, lib_id, day))
                continue

            if -neg_score <= 0:
                break

            order.append(lib_id)
            remaining.remove(lib_id)
            day += data.libs[lib_id].signup_days
            InitialSolution._mark_greedy_books(data, lib_id, day, used_books)

        order.extend(sorted(remaining))
        return order

    @staticmethod
    def _build_order_static_priority(data, alpha=1.0, potential_mode="cap_raw"):
        potential = data.potential_array(potential_mode)
        ordered = sorted(
            range(data.num_libs),
            key=lambda lib_id: (
                potential[lib_id] / (max(1, data.lib_signup_days[lib_id]) ** alpha),
                data.lib_books_per_day[lib_id],
                -data.lib_signup_days[lib_id],
            ),
            reverse=True,
        )
        return ordered

    @staticmethod
    def _build_order_adaptive_heap(data, alpha=1.0, potential_mode="top_raw", noise=0.0):
        heap = []
        potential = data.potential_array(potential_mode)
        rarity_weighted = potential_mode.endswith("rare")

        for lib_id in range(data.num_libs):
            signup = data.lib_signup_days[lib_id]
            base = potential[lib_id] / (max(1, signup) ** alpha)
            if base <= 0:
                continue
            if noise > 0.0:
                base *= 1.0 + random.uniform(-noise, noise)
            heapq.heappush(heap, (-base, lib_id))

        order = []
        used_books = set()
        used_libs = set()
        day = 0

        while heap and day < data.num_days:
            neg_eff, lib_id = heapq.heappop(heap)
            if lib_id in used_libs:
                continue

            signup = data.lib_signup_days[lib_id]
            if day + signup >= data.num_days:
                continue

            while True:
                remaining_days = data.num_days - (day + signup)
                capacity = remaining_days * data.lib_books_per_day[lib_id]
                if capacity <= 0:
                    lib_id = -1
                    break

                gain = InitialSolution._library_gain(
                    data,
                    lib_id,
                    used_books,
                    capacity,
                    rarity_weighted=rarity_weighted,
                )
                if gain <= 0:
                    lib_id = -1
                    break

                real_eff = gain / (max(1, signup) ** alpha)
                if not heap or real_eff >= -heap[0][0]:
                    break

                heapq.heappush(heap, (-real_eff, lib_id))
                neg_eff, lib_id = heapq.heappop(heap)
                if lib_id in used_libs:
                    lib_id = -1
                    break
                signup = data.lib_signup_days[lib_id]
                if day + signup >= data.num_days:
                    lib_id = -1
                    break

            if lib_id == -1:
                continue

            order.append(lib_id)
            used_libs.add(lib_id)
            day += signup
            InitialSolution._mark_greedy_books(data, lib_id, day, used_books)

        order.extend(lib_id for lib_id in range(data.num_libs) if lib_id not in used_libs)
        return order

    @staticmethod
    def _build_order_fill_ratio_heap(
        data,
        alpha=1.0,
        beta=20.0,
        gamma=2.0,
        rarity_weighted=False,
        noise=0.0,
    ):
        heap = []
        used_books = set()
        used_libs = set()
        order = []
        day = 0

        for lib_id in range(data.num_libs):
            score = InitialSolution._library_fill_value(
                data,
                lib_id,
                day,
                used_books,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                rarity_weighted=rarity_weighted,
                noise=noise,
            )
            if score > 0:
                heapq.heappush(heap, (-score, lib_id, day, len(order)))

        while heap and day < data.num_days:
            neg_score, lib_id, scored_day, scored_version = heapq.heappop(heap)
            if lib_id in used_libs:
                continue

            if scored_day != day or scored_version != len(order):
                refreshed = InitialSolution._library_fill_value(
                    data,
                    lib_id,
                    day,
                    used_books,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    rarity_weighted=rarity_weighted,
                    noise=noise,
                )
                if refreshed > 0:
                    heapq.heappush(heap, (-refreshed, lib_id, day, len(order)))
                continue

            if -neg_score <= 0:
                break

            order.append(lib_id)
            used_libs.add(lib_id)
            day += data.lib_signup_days[lib_id]
            InitialSolution._mark_greedy_books(data, lib_id, day, used_books)

        order.extend(lib_id for lib_id in range(data.num_libs) if lib_id not in used_libs)
        return order

    @staticmethod
    def _build_order_branch_fill_greedy(data, alpha=1.575, gamma=30.0):
        planned = bytearray(data.num_books)
        selected = bytearray(data.num_libs)
        order = []
        current_day = 0
        denom = [
            (data.lib_signup_days[lib_id] + 1) ** alpha
            for lib_id in range(data.num_libs)
        ]
        heap = []

        for lib_id in range(data.num_libs):
            upper = sum(data.scores[book_id] for book_id in data.lib_book_ids[lib_id])
            if upper > 0:
                heapq.heappush(heap, (-(upper / denom[lib_id]), lib_id))

        def upper_bound(lib_id):
            value = 0
            for book_id in data.lib_book_ids[lib_id]:
                if not planned[book_id]:
                    value += data.scores[book_id]
            return value / denom[lib_id]

        def actual_value(lib_id):
            scan_days = data.num_days - current_day - data.lib_signup_days[lib_id]
            if scan_days <= 0:
                return (-1.0, 0, 0), []

            capacity = scan_days * data.lib_books_per_day[lib_id]
            if capacity <= 0:
                return (-1.0, 0, 0), []

            chosen = []
            raw_score = 0
            for book_id in data.lib_book_ids[lib_id]:
                if planned[book_id]:
                    continue
                chosen.append(book_id)
                raw_score += data.scores[book_id]
                if len(chosen) >= capacity:
                    break

            if not chosen:
                return (-1.0, 0, 0), []

            fill = len(chosen) / max(1, min(capacity, data.lib_num_books[lib_id]))
            key = raw_score * (fill ** gamma) / denom[lib_id]
            return (key, raw_score, -data.lib_signup_days[lib_id]), chosen

        while heap and current_day < data.num_days:
            best_key = None
            best_lib = -1
            best_books = []
            touched = []

            while heap:
                _old_upper, lib_id = heapq.heappop(heap)
                if selected[lib_id]:
                    continue

                upper = upper_bound(lib_id)
                key, books = actual_value(lib_id)
                if key[1] <= 0:
                    selected[lib_id] = 1
                    continue

                touched.append((-upper, lib_id))
                if best_key is None or key > best_key:
                    best_key = key
                    best_lib = lib_id
                    best_books = books

                next_upper = -heap[0][0] if heap else -1.0
                if best_key[0] >= next_upper - 1e-12:
                    break

            for item in touched:
                if item[1] != best_lib and not selected[item[1]]:
                    heapq.heappush(heap, item)

            if best_lib < 0:
                break
            signup = data.lib_signup_days[best_lib]
            if current_day + signup >= data.num_days:
                break

            selected[best_lib] = 1
            current_day += signup
            order.append(best_lib)
            for book_id in best_books:
                planned[book_id] = 1

        order.extend(lib_id for lib_id in range(data.num_libs) if not selected[lib_id])
        return order

    @staticmethod
    def _build_best_grasp_order(data, rcl_ratio=0.05, max_time=5.0):
        start_time = time.time()
        best_order = None
        best_score = -1

        while time.time() - start_time < max_time:
            order = InitialSolution._build_grasp_order(data, rcl_ratio)
            score = data.fast_evaluate(order)
            if score > best_score:
                best_score = score
                best_order = order

        if best_order is None:
            best_order = InitialSolution._build_order_greedy_heap(
                data,
                alpha=1.5,
                capacity_aware=True,
                rarity_weighted=False,
            )
            best_score = data.fast_evaluate(best_order)

        return best_order, best_score

    @staticmethod
    def _build_grasp_order(data, rcl_ratio):
        remaining = InitialSolution._build_order_greedy_heap(
            data,
            alpha=1.2,
            capacity_aware=True,
            rarity_weighted=True,
        )
        order = []
        used_books = set()
        day = 0
        scan_limit = 64
        min_candidates = 12

        while remaining and day < data.num_days:
            scored = []
            scan_count = min(len(remaining), max(scan_limit, min_candidates))
            for lib_id in remaining[:scan_count]:
                value = InitialSolution._library_value(
                    data,
                    lib_id,
                    day,
                    used_books,
                    alpha=1.2,
                    capacity_aware=True,
                    rarity_weighted=True,
                )
                if value > 0:
                    scored.append((value, lib_id))
                    if len(scored) >= min_candidates:
                        break

            if not scored:
                break

            scored.sort(reverse=True)
            rcl_size = max(1, int(len(scored) * rcl_ratio))
            _, chosen_lib = random.choice(scored[:rcl_size])
            order.append(chosen_lib)
            remaining.remove(chosen_lib)
            day += data.libs[chosen_lib].signup_days
            InitialSolution._mark_greedy_books(data, chosen_lib, day, used_books)

        order.extend(remaining)
        return order

    @staticmethod
    def _mark_greedy_books(data, lib_id, day, used_books):
        if day >= data.num_days:
            return
        remaining_days = data.num_days - day
        capacity = remaining_days * data.lib_books_per_day[lib_id]
        if capacity <= 0:
            return
        count = 0
        for book_id in data.lib_book_ids[lib_id]:
            if book_id in used_books:
                continue
            used_books.add(book_id)
            count += 1
            if count >= capacity:
                break

    @staticmethod
    def _library_gain(data, lib_id, used_books, capacity, rarity_weighted=False):
        gain = 0.0
        taken = 0
        score_source = data.effective_scores if rarity_weighted else data.scores
        for book_id in data.lib_book_ids[lib_id]:
            if book_id in used_books:
                continue
            gain += score_source[book_id]
            taken += 1
            if taken >= capacity:
                break
        return gain

    @staticmethod
    def _library_fill_value(
        data,
        lib_id,
        current_day,
        used_books,
        alpha=1.0,
        beta=20.0,
        gamma=2.0,
        rarity_weighted=False,
        noise=0.0,
    ):
        signup_days = data.lib_signup_days[lib_id]
        if current_day + signup_days >= data.num_days:
            return 0.0

        remaining_days = data.num_days - (current_day + signup_days)
        capacity = remaining_days * data.lib_books_per_day[lib_id]
        if capacity <= 0:
            return 0.0

        useful_limit = min(capacity, data.lib_num_books[lib_id])
        if useful_limit <= 0:
            return 0.0

        value = 0.0
        useful_count = 0
        for book_id in data.lib_book_ids[lib_id]:
            if book_id in used_books:
                continue
            if rarity_weighted:
                value += data.scores[book_id] / max(1, data.book_freq[book_id])
            else:
                value += data.scores[book_id]
            useful_count += 1
            if useful_count >= capacity:
                break

        if value <= 0.0 or useful_count <= 0:
            return 0.0

        fill_ratio = useful_count / max(1, useful_limit)
        denom = (signup_days + beta) ** alpha
        score = value * (fill_ratio ** gamma) / max(1.0, denom)
        if noise > 0.0:
            score *= 1.0 + random.uniform(-noise, noise)
        return score

    @staticmethod
    def _library_value(
        data,
        lib_id,
        current_day,
        used_books,
        alpha=1.0,
        capacity_aware=False,
        rarity_weighted=False,
        noise=0.0,
        position_penalty_beta=0.0,
        used_count=0,
    ):
        signup_days = data.lib_signup_days[lib_id]
        if current_day + signup_days >= data.num_days:
            return 0.0

        remaining_days = data.num_days - (current_day + signup_days)
        capacity = remaining_days * data.lib_books_per_day[lib_id]
        if capacity <= 0:
            return 0.0

        value = 0.0
        count = 0
        for book_id in data.lib_book_ids[lib_id]:
            if book_id in used_books:
                continue
            rarity_factor = 1.0
            if rarity_weighted:
                rarity_factor = 1.0 / max(1, data.book_freq[book_id])
            value += data.scores[book_id] * rarity_factor
            count += 1
            if capacity_aware and count >= capacity:
                break

        if value <= 0:
            return 0.0

        denom = max(1.0, float(signup_days) ** alpha)
        denom *= 1.0 + position_penalty_beta * used_count
        score = value / denom
        if capacity_aware:
            num_books = data.lib_num_books[lib_id]
            score *= 1.0 + min(capacity, num_books) / max(1, num_books)
        if noise > 0.0:
            score *= 1.0 + random.uniform(-noise, noise)
        return score

    @staticmethod
    def _build_constructor_plan(data, alphas, beta):
        constructors = [("Sorted", InitialSolution._build_order_sorted, {})]
        for alpha in alphas:
            constructors.append(
                (
                    f"Heap Greedy top/raw alpha={alpha}",
                    InitialSolution._build_order_adaptive_heap,
                    {"alpha": alpha, "potential_mode": "top_raw"},
                )
            )
        weighted_alpha = alphas[len(alphas) // 2] if alphas else 1.0
        constructors.extend(
            [
                (
                    f"Heap Greedy cap/raw alpha={weighted_alpha}",
                    InitialSolution._build_order_adaptive_heap,
                    {"alpha": weighted_alpha, "potential_mode": "cap_raw"},
                ),
                (
                    f"Heap Greedy cap/rare alpha={weighted_alpha}",
                    InitialSolution._build_order_adaptive_heap,
                    {"alpha": weighted_alpha, "potential_mode": "cap_rare"},
                ),
                (
                    f"Fill Greedy raw alpha={weighted_alpha}",
                    InitialSolution._build_order_fill_ratio_heap,
                    {"alpha": weighted_alpha, "rarity_weighted": False},
                ),
                (
                    f"Fill Greedy rare alpha={weighted_alpha}",
                    InitialSolution._build_order_fill_ratio_heap,
                    {"alpha": weighted_alpha, "rarity_weighted": True},
                ),
            ]
        )
        if data.num_libs <= 3000:
            constructors.append(
                (
                    "Branch Fill Greedy alpha=1.575 gamma=30.0",
                    InitialSolution._build_order_branch_fill_greedy,
                    {"alpha": 1.575, "gamma": 30.0},
                )
            )
        constructors.extend(
            [
                (
                    f"Static Greedy cap/raw alpha={weighted_alpha}",
                    InitialSolution._build_order_static_priority,
                    {"alpha": weighted_alpha, "potential_mode": "cap_raw"},
                ),
                (
                    f"Static Greedy cap/rare alpha={weighted_alpha}",
                    InitialSolution._build_order_static_priority,
                    {"alpha": weighted_alpha, "potential_mode": "cap_rare"},
                ),
                (
                    f"Noisy Heap cap/raw alpha={weighted_alpha}",
                    InitialSolution._build_order_adaptive_heap,
                    {"alpha": weighted_alpha, "potential_mode": "cap_raw", "noise": 0.04},
                ),
                (
                    f"Noisy Heap cap/rare alpha={weighted_alpha}",
                    InitialSolution._build_order_adaptive_heap,
                    {"alpha": weighted_alpha, "potential_mode": "cap_rare", "noise": 0.04},
                ),
            ]
        )
        return constructors
