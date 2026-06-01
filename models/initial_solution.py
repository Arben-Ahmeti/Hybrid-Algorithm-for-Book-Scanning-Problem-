import heapq
import random
import threading
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
        noisy_restarts=1,
        verbose=True,
    ):
        if hasattr(data, "to_flat_arrays"):
            from models.evaluation import warmup_jit

            warmup_jit(data.to_flat_arrays())
        start_time = time.time()
        alphas = alphas or [0.5, 1.0, 1.5, 2.0]
        screened_candidates = []

        print("\nOK...........")

        if verbose:
            print("\nGenerating Initial Solutions...")

        constructors = InitialSolution._build_constructor_plan(
            alphas=alphas,
            beta=beta,
        )

        for i, (label, method, method_kwargs) in enumerate(constructors):
            if time.time() - start_time >= max_time:
                print(f"Timeout reached at constructor {i}/{len(constructors)}")
                break

            print(f"Running constructor {i + 1}/{len(constructors)}: {label}")

            try:
                method_start = time.time()
                order = method(data, **method_kwargs)
                method_time = time.time() - method_start
                print(f"  Completed in {method_time:.2f}s")

                if method_time > 10.0:
                    print(f"  WARNING: Method {label} took {method_time:.2f}s")

                # FIX: Actually save the result!
                if order is not None:
                    score = data.fast_evaluate(order)
                    screened_candidates.append((score, label, order))
                    print(f"  Saved candidate with score: {score:,}")

            except Exception as e:
                print(f"  ERROR in {label}: {e}")
                import traceback
                traceback.print_exc()
                continue

        elapsed = time.time() - start_time
        remaining_for_grasp = max(0.0, max_time - elapsed)
        grasp_time = min(max(0.0, grasp_max_time), remaining_for_grasp)

        # if grasp_time > 0.0:
        #     grasp_label = f"GRASP rcl={grasp_rcl:.2f} time={grasp_time:.1f}s"
        #     if verbose:
        #         print(f"  Running {grasp_label}...")
        #     grasp_order, approx_score = InitialSolution._build_best_grasp_order(
        #         data,
        #         rcl_ratio=grasp_rcl,
        #         max_time=grasp_time,
        #     )
        #     screened_candidates.append((approx_score, grasp_label, grasp_order))
        #     if verbose:
        #         print(f"    Approx score: {approx_score:,}")

        print(f"📊 All constructors completed. Screened candidates: {len(screened_candidates)}")

        if not screened_candidates:
            print("⚠️ No screened candidates, creating fallback...")
            fallback_order = InitialSolution._build_order_sorted(data)
            screened_candidates.append((data.fast_evaluate(fallback_order), "Fallback Sorted", fallback_order))
            print("✅ Fallback created")

        print("🔄 Sorting screened candidates...")
        screened_candidates.sort(key=lambda item: item[0], reverse=True)
        print("✅ Candidates sorted")

        refine_count = min(3, len(screened_candidates))
        print(f"🔄 Refining {refine_count} candidates...")

        candidates = []
        for i, (score, label, order) in enumerate(screened_candidates[:refine_count]):
            print(f"  Refining candidate {i + 1}/{refine_count}: {label} (score: {score})")

            # THIS IS LIKELY WHERE IT HANGS:
            # print(f"    Creating solution from order...")
            # solution = Solution.from_order(order, data)
            # print(f"    Solution created, calculating fitness...")
            def create_solution_with_timeout(order, data, timeout=10):
                result = [None]
                error = [None]

                def target():
                    try:
                        print(f"    Calling Solution.from_order() with {len(order)} libs...")
                        result[0] = Solution.from_order(order, data)
                        print(f"    Solution.from_order() returned!")
                    except Exception as e:
                        import traceback
                        error[0] = traceback.format_exc()
                        print(f"    Solution.from_order() EXCEPTION: {e}")

                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                thread.join(timeout=timeout)

                if thread.is_alive():
                    print(f"    TIMEOUT: Solution.from_order() hung after {timeout}s!")
                    return None
                if error[0]:
                    print(f"    ERROR in Solution.from_order(): {error[0]}")
                    return None
                return result[0]

            # Then use it:
            solution = create_solution_with_timeout(order, data, timeout=10)
            if solution is None:
                print(f"    Skipping candidate due to timeout/error")
                continue
            # The hang might be here if fitness calculation is expensive
            fitness = solution.fitness_score
            print(f"    Fitness calculated: {fitness}")

            candidates.append((fitness, label, solution))
            print(f"    Candidate {i + 1} completed: {fitness:,} ({label})")

        print("🔄 Sorting final candidates...")
        candidates.sort(key=lambda item: item[0], reverse=True)
        # ADD THIS GUARD: if all refinements failed, fall back to screened candidates
        if not candidates:
            print("⚠️ All refined candidates failed! Falling back to screened candidates...")
            for score, label, order in screened_candidates[:3]:
                try:
                    # Build a simple fallback solution directly without Solution.from_order()
                    scanned_books = set()
                    scanned_per_lib = {}
                    used_libs = []
                    current_day = 0

                    for lib_id in order:
                        signup = data.lib_signup_days[lib_id]
                        if current_day + signup >= data.num_days:
                            continue
                        current_day += signup
                        remaining_days = data.num_days - current_day
                        max_books = remaining_days * data.lib_books_per_day[lib_id]
                        if max_books <= 0:
                            continue

                        available = [b for b in data.lib_book_ids[lib_id]
                                     if b not in scanned_books][:max_books]
                        if available:
                            scanned_books.update(available)
                            scanned_per_lib[lib_id] = available
                            used_libs.append(lib_id)

                    fallback = Solution(
                        signed_libs=used_libs,
                        unsigned_libs=list(set(range(data.num_libs)) - set(used_libs)),
                        scanned_books_per_library=scanned_per_lib,
                        scanned_books=scanned_books
                    )
                    fallback.calculate_fitness_score(data.scores)
                    candidates.append((fallback.fitness_score, label, fallback))
                    print(f"  ✅ Fallback solution created: {fallback.fitness_score:,} ({label})")
                    break  # One fallback is enough
                except Exception as e:
                    print(f"  ❌ Fallback also failed: {e}")
                    continue

        if not candidates:
            raise RuntimeError("❌ FATAL: Could not generate any valid initial solution!")

        best_score, best_label, best_solution = candidates[0]

        print("\n✅ Initial Solutions Completed!")
        if verbose:
            print(f"Best Initial Score: {best_score:,} ({best_label})")
        return best_solution, candidates


    @staticmethod
    def generate_adaptive_restart_solution(data, alphas=None):
        alphas = alphas or [0.5, 1.0, 1.5, 2.0]
        alpha = random.choice(alphas)
        potential_mode = random.choice(["top_raw", "top_rare", "cap_raw", "cap_rare"])
        noise = random.uniform(0.03, 0.10)
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
    def generate_initial_solution_grasp(data, rcl_ratio=0.05, max_time=5.0):
        order, _ = InitialSolution._build_best_grasp_order(data, rcl_ratio, max_time)
        return Solution.from_order(order, data)

    @staticmethod
    def generate_initial_sorted(data):
        return Solution.from_order(InitialSolution._build_order_sorted(data), data)

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
    def generate_initial_static_greedy(data, alpha=1.0, capacity_aware=False, rarity_weighted=False):
        remaining = list(range(data.num_libs))
        order = []
        used_books = set()
        day = 0

        while remaining and day < data.num_days:
            best_lib = None
            best_value = None
            for lib_id in remaining:
                value = InitialSolution._library_value(
                    data,
                    lib_id,
                    day,
                    used_books,
                    alpha=alpha,
                    capacity_aware=capacity_aware,
                    rarity_weighted=rarity_weighted,
                )
                if best_value is None or value > best_value:
                    best_value = value
                    best_lib = lib_id

            if best_lib is None or best_value <= 0:
                break

            order.append(best_lib)
            remaining.remove(best_lib)
            #day += data.libs[best_lib].signup_days
            day += data.lib_signup_days[best_lib]
            InitialSolution._mark_greedy_books(data, best_lib, day, used_books)

        order.extend(remaining)
        return Solution.from_order(order, data)

    @staticmethod
    def generate_initial_dynamic(data, alpha=1.0):
        remaining = set(range(data.num_libs))
        order = []
        used_books = set()
        day = 0

        while remaining and day < data.num_days:
            best_lib = None
            best_value = None
            for lib_id in remaining:
                value = InitialSolution._library_value(
                    data,
                    lib_id,
                    day,
                    used_books,
                    alpha=alpha,
                    capacity_aware=True,
                    rarity_weighted=True,
                )
                if best_value is None or value > best_value:
                    best_value = value
                    best_lib = lib_id

            if best_lib is None or best_value <= 0:
                break

            order.append(best_lib)
            remaining.remove(best_lib)
            day += data.libs[best_lib].signup_days
            InitialSolution._mark_greedy_books(data, best_lib, day, used_books)

        order.extend(sorted(remaining))
        return Solution.from_order(order, data)

    @staticmethod
    def generate_initial_greedy_heap(
        data,
        alpha=1.0,
        capacity_aware=False,
        rarity_weighted=False,
        noise=0.0,
        beta=0.0,
    ):
        order = InitialSolution._build_order_greedy_heap(
            data,
            alpha=alpha,
            capacity_aware=capacity_aware,
            rarity_weighted=rarity_weighted,
            noise=noise,
            beta=beta,
        )
        return Solution.from_order(order, data)

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
            #day += data.libs[lib_id].signup_days
            day += data.lib_signup_days[lib_id]
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
    def generate_initial_weighted_efficiency(data, alpha=1.0, beta=0.12):
        order = InitialSolution._build_order_weighted_efficiency(data, alpha=alpha, beta=beta)
        return Solution.from_order(order, data)

    @staticmethod
    def _build_order_weighted_efficiency(data, alpha=1.0, beta=0.12):
        return InitialSolution._build_order_greedy_heap(
            data,
            alpha=alpha,
            capacity_aware=True,
            rarity_weighted=False,
            beta=beta,
        )

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
            #day += data.libs[chosen_lib].signup_days
            day += data.lib_signup_days[chosen_lib]
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
    def _build_constructor_plan(alphas, beta):
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
