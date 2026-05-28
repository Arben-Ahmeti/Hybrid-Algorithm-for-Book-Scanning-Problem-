"""Exact JIT-accelerated evaluation for the Book Scanning solver."""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(fastmath=False, cache=False):
        def decorator(func):
            return func
        return decorator


@njit(fastmath=True, cache=True)
def fast_evaluate(order, libs_signup, libs_rate, books_flat, books_offsets,
                  books_lengths, book_scores, total_days):
    """Exact score using the official sequential library scanning rule."""
    scanned = np.zeros(len(book_scores), dtype=np.uint8)
    current_day = 0
    score = np.int64(0)

    for idx in range(len(order)):
        lib_id = order[idx]
        signup = libs_signup[lib_id]
        if current_day + signup >= total_days:
            continue

        current_day += signup
        remaining_days = total_days - current_day
        capacity = remaining_days * libs_rate[lib_id]
        if capacity <= 0:
            continue

        offset = books_offsets[lib_id]
        length = books_lengths[lib_id]
        taken = 0
        for k in range(length):
            if taken >= capacity:
                break
            book_id = books_flat[offset + k]
            if scanned[book_id] == 0:
                scanned[book_id] = 1
                score += book_scores[book_id]
                taken += 1

    return score


@njit(fastmath=True, cache=True)
def fast_evaluate_global(order, libs_signup, libs_rate, lib_num_books,
                         books_by_score, book_libs_flat, book_libs_offsets,
                         book_libs_lengths, book_scores, total_days):
    """Global assignment heuristic for a fixed library order."""
    num_libs = len(libs_signup)
    active = np.zeros(num_libs, dtype=np.uint8)
    remaining_capacity = np.zeros(num_libs, dtype=np.int32)
    remaining_candidates = np.zeros(num_libs, dtype=np.int32)
    positions = np.zeros(num_libs, dtype=np.int32)
    day = 0

    for position in range(len(order)):
        lib_id = order[position]
        signup = libs_signup[lib_id]
        if day + signup >= total_days:
            continue
        day += signup
        remaining_days = total_days - day
        capacity = remaining_days * libs_rate[lib_id]
        if capacity <= 0:
            continue
        active[lib_id] = 1
        remaining_capacity[lib_id] = min(capacity, lib_num_books[lib_id])
        remaining_candidates[lib_id] = lib_num_books[lib_id]
        positions[lib_id] = position

    score = np.int64(0)
    for idx in range(len(books_by_score)):
        book_id = books_by_score[idx]
        best_lib = -1
        best_key1 = -1
        best_key2 = -1
        best_key3 = -2147483647
        offset = book_libs_offsets[book_id]
        length = book_libs_lengths[book_id]

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            if active[lib_id] == 0 or remaining_capacity[lib_id] <= 0:
                continue
            key1 = remaining_candidates[lib_id]
            if remaining_capacity[lib_id] < key1:
                key1 = remaining_capacity[lib_id]
            key2 = remaining_capacity[lib_id]
            key3 = -positions[lib_id]
            if (
                key1 > best_key1
                or (key1 == best_key1 and key2 > best_key2)
                or (key1 == best_key1 and key2 == best_key2 and key3 > best_key3)
            ):
                best_key1 = key1
                best_key2 = key2
                best_key3 = key3
                best_lib = lib_id

        if best_lib == -1:
            continue

        score += book_scores[book_id]
        remaining_capacity[best_lib] -= 1

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            if active[lib_id] == 1:
                remaining_candidates[lib_id] -= 1

    return score


@njit(fastmath=True, cache=True)
def fast_evaluate_sequential(order, libs_signup, libs_rate,
                             books_flat, books_offsets, books_lengths,
                             book_scores, total_days):
    """
    Faster proxy score used for aggressive local search.
    Books are scanned greedily in library order from each library's
    pre-sorted book list.
    """
    scanned = np.zeros(len(book_scores), dtype=np.uint8)
    current_day = 0
    score = np.int64(0)

    for idx in range(len(order)):
        lib_id = order[idx]
        signup = libs_signup[lib_id]
        if current_day + signup >= total_days:
            continue

        current_day += signup
        remaining_days = total_days - current_day
        capacity = remaining_days * libs_rate[lib_id]
        if capacity <= 0:
            continue

        offset = books_offsets[lib_id]
        length = books_lengths[lib_id]
        taken = 0
        for k in range(length):
            if taken >= capacity:
                break
            book_id = books_flat[offset + k]
            if scanned[book_id] == 0:
                scanned[book_id] = 1
                score += book_scores[book_id]
                taken += 1

    return score


@njit(fastmath=True, cache=True)
def fast_evaluate_sequential_workspace(
    order,
    libs_signup,
    libs_rate,
    books_flat,
    books_offsets,
    books_lengths,
    book_scores,
    total_days,
    scanned,
    touched_books,
):
    """
    Sequential score using caller-provided buffers.

    The standard evaluator allocates a fresh scanned-book mask for every call.
    Local search calls this function many times, so this variant records only
    touched books and clears those entries before returning.
    """
    current_day = 0
    score = np.int64(0)
    touched_count = 0

    for idx in range(len(order)):
        lib_id = order[idx]
        signup = libs_signup[lib_id]
        if current_day + signup >= total_days:
            continue

        current_day += signup
        remaining_days = total_days - current_day
        capacity = remaining_days * libs_rate[lib_id]
        if capacity <= 0:
            continue

        offset = books_offsets[lib_id]
        length = books_lengths[lib_id]
        taken = 0
        for k in range(length):
            if taken >= capacity:
                break
            book_id = books_flat[offset + k]
            if scanned[book_id] == 0:
                scanned[book_id] = 1
                touched_books[touched_count] = book_id
                touched_count += 1
                score += book_scores[book_id]
                taken += 1

    for idx in range(touched_count):
        scanned[touched_books[idx]] = 0

    return score


@njit(fastmath=True, cache=True)
def fast_evaluate_global_workspace(
    order,
    libs_signup,
    libs_rate,
    lib_num_books,
    books_by_score,
    book_libs_flat,
    book_libs_offsets,
    book_libs_lengths,
    book_scores,
    total_days,
    active,
    remaining_capacity,
    remaining_candidates,
    positions,
    touched_libs,
):
    """
    Global assignment heuristic using caller-provided buffers.

    Only libraries selected by the current order are reset, avoiding repeated
    O(num_libraries) allocations and clears during local search.
    """
    day = 0
    selected_count = 0

    for position in range(len(order)):
        lib_id = order[position]
        signup = libs_signup[lib_id]
        if day + signup >= total_days:
            continue
        day += signup
        remaining_days = total_days - day
        capacity = remaining_days * libs_rate[lib_id]
        if capacity <= 0:
            continue
        active[lib_id] = 1
        remaining_capacity[lib_id] = min(capacity, lib_num_books[lib_id])
        remaining_candidates[lib_id] = lib_num_books[lib_id]
        positions[lib_id] = position
        touched_libs[selected_count] = lib_id
        selected_count += 1

    score = np.int64(0)
    for idx in range(len(books_by_score)):
        book_id = books_by_score[idx]
        best_lib = -1
        best_key1 = -1
        best_key2 = -1
        best_key3 = -2147483647
        offset = book_libs_offsets[book_id]
        length = book_libs_lengths[book_id]

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            if active[lib_id] == 0 or remaining_capacity[lib_id] <= 0:
                continue
            key1 = remaining_candidates[lib_id]
            if remaining_capacity[lib_id] < key1:
                key1 = remaining_capacity[lib_id]
            key2 = remaining_capacity[lib_id]
            key3 = -positions[lib_id]
            if (
                key1 > best_key1
                or (key1 == best_key1 and key2 > best_key2)
                or (key1 == best_key1 and key2 == best_key2 and key3 > best_key3)
            ):
                best_key1 = key1
                best_key2 = key2
                best_key3 = key3
                best_lib = lib_id

        if best_lib == -1:
            continue

        score += book_scores[book_id]
        remaining_capacity[best_lib] -= 1

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            if active[lib_id] == 1:
                remaining_candidates[lib_id] -= 1

    for idx in range(selected_count):
        lib_id = touched_libs[idx]
        active[lib_id] = 0
        remaining_capacity[lib_id] = 0
        remaining_candidates[lib_id] = 0
        positions[lib_id] = 0

    return score


@njit(fastmath=True, cache=True)
def fast_evaluate_balanced_global_workspace(
    order,
    libs_signup,
    libs_rate,
    lib_num_books,
    books_by_score,
    book_libs_flat,
    book_libs_offsets,
    book_libs_lengths,
    book_scores,
    total_days,
    active,
    remaining_capacity,
    positions,
    touched_libs,
):
    """
    Global assignment using a remaining-capacity ratio tie-break.

    For a fixed library order, high-score books are assigned to an active
    library that can scan them. The key favors libraries with larger
    remaining_capacity / library_size, which is especially useful when many
    libraries overlap heavily but have different capacities.
    """
    day = 0
    selected_count = 0

    for position in range(len(order)):
        lib_id = order[position]
        signup = libs_signup[lib_id]
        if day + signup >= total_days:
            continue
        day += signup
        remaining_days = total_days - day
        capacity = remaining_days * libs_rate[lib_id]
        if capacity <= 0:
            continue
        active[lib_id] = 1
        remaining_capacity[lib_id] = capacity
        positions[lib_id] = position
        touched_libs[selected_count] = lib_id
        selected_count += 1

    score = np.int64(0)
    for idx in range(len(books_by_score)):
        book_id = books_by_score[idx]
        best_lib = -1
        best_ratio = -1.0
        best_capacity = -1
        offset = book_libs_offsets[book_id]
        length = book_libs_lengths[book_id]

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            capacity = remaining_capacity[lib_id]
            if active[lib_id] == 0 or capacity <= 0:
                continue
            size = lib_num_books[lib_id]
            if size <= 0:
                size = 1
            ratio = capacity / size
            if ratio > best_ratio or (ratio == best_ratio and capacity > best_capacity):
                best_ratio = ratio
                best_capacity = capacity
                best_lib = lib_id

        if best_lib == -1:
            continue

        score += book_scores[book_id]
        remaining_capacity[best_lib] -= 1

    for idx in range(selected_count):
        lib_id = touched_libs[idx]
        active[lib_id] = 0
        remaining_capacity[lib_id] = 0
        positions[lib_id] = 0

    return score


@njit(fastmath=True, cache=True)
def fast_evaluate_detailed_global(
    order,
    libs_signup,
    libs_rate,
    lib_num_books,
    books_by_score,
    book_libs_flat,
    book_libs_offsets,
    book_libs_lengths,
    book_scores,
    total_days,
    out_selected,
    out_books,
    out_book_libs,
):
    """Detailed global assignment output for reconstruction."""
    num_libs = len(libs_signup)
    active = np.zeros(num_libs, dtype=np.uint8)
    remaining_capacity = np.zeros(num_libs, dtype=np.int32)
    remaining_candidates = np.zeros(num_libs, dtype=np.int32)
    positions = np.zeros(num_libs, dtype=np.int32)
    day = 0
    score = np.int64(0)
    selected_count = 0
    assigned_count = 0

    for position in range(len(order)):
        lib_id = order[position]
        signup = libs_signup[lib_id]
        if day + signup >= total_days:
            continue
        day += signup
        remaining_days = total_days - day
        capacity = remaining_days * libs_rate[lib_id]
        if capacity <= 0:
            continue
        active[lib_id] = 1
        remaining_capacity[lib_id] = min(capacity, lib_num_books[lib_id])
        remaining_candidates[lib_id] = lib_num_books[lib_id]
        positions[lib_id] = position
        out_selected[selected_count] = lib_id
        selected_count += 1

    for idx in range(len(books_by_score)):
        book_id = books_by_score[idx]
        best_lib = -1
        best_key1 = -1
        best_key2 = -1
        best_key3 = -2147483647
        offset = book_libs_offsets[book_id]
        length = book_libs_lengths[book_id]

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            if active[lib_id] == 0 or remaining_capacity[lib_id] <= 0:
                continue
            key1 = remaining_candidates[lib_id]
            if remaining_capacity[lib_id] < key1:
                key1 = remaining_capacity[lib_id]
            key2 = remaining_capacity[lib_id]
            key3 = -positions[lib_id]
            if (
                key1 > best_key1
                or (key1 == best_key1 and key2 > best_key2)
                or (key1 == best_key1 and key2 == best_key2 and key3 > best_key3)
            ):
                best_key1 = key1
                best_key2 = key2
                best_key3 = key3
                best_lib = lib_id

        if best_lib == -1:
            continue

        score += book_scores[book_id]
        out_books[assigned_count] = book_id
        out_book_libs[assigned_count] = best_lib
        assigned_count += 1
        remaining_capacity[best_lib] -= 1

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            if active[lib_id] == 1:
                remaining_candidates[lib_id] -= 1

    return score, selected_count, assigned_count


@njit(fastmath=True, cache=True)
def fast_evaluate_detailed_balanced_global(
    order,
    libs_signup,
    libs_rate,
    lib_num_books,
    books_by_score,
    book_libs_flat,
    book_libs_offsets,
    book_libs_lengths,
    book_scores,
    total_days,
    out_selected,
    out_books,
    out_book_libs,
):
    """Detailed balanced global assignment output for reconstruction."""
    num_libs = len(libs_signup)
    active = np.zeros(num_libs, dtype=np.uint8)
    remaining_capacity = np.zeros(num_libs, dtype=np.int32)
    day = 0
    score = np.int64(0)
    selected_count = 0
    assigned_count = 0

    for position in range(len(order)):
        lib_id = order[position]
        signup = libs_signup[lib_id]
        if day + signup >= total_days:
            continue
        day += signup
        remaining_days = total_days - day
        capacity = remaining_days * libs_rate[lib_id]
        if capacity <= 0:
            continue
        active[lib_id] = 1
        remaining_capacity[lib_id] = capacity
        out_selected[selected_count] = lib_id
        selected_count += 1

    for idx in range(len(books_by_score)):
        book_id = books_by_score[idx]
        best_lib = -1
        best_ratio = -1.0
        best_capacity = -1
        offset = book_libs_offsets[book_id]
        length = book_libs_lengths[book_id]

        for k in range(length):
            lib_id = book_libs_flat[offset + k]
            capacity = remaining_capacity[lib_id]
            if active[lib_id] == 0 or capacity <= 0:
                continue
            size = lib_num_books[lib_id]
            if size <= 0:
                size = 1
            ratio = capacity / size
            if ratio > best_ratio or (ratio == best_ratio and capacity > best_capacity):
                best_ratio = ratio
                best_capacity = capacity
                best_lib = lib_id

        if best_lib == -1:
            continue

        score += book_scores[book_id]
        out_books[assigned_count] = book_id
        out_book_libs[assigned_count] = best_lib
        assigned_count += 1
        remaining_capacity[best_lib] -= 1

    return score, selected_count, assigned_count


@njit(fastmath=True, cache=True)
def fast_evaluate_detailed(
    order,
    libs_signup,
    libs_rate,
    books_flat,
    books_offsets,
    books_lengths,
    book_scores,
    total_days,
    out_selected,
    out_books,
    out_book_libs,
):
    """Exact detailed evaluation with official sequential assignment output."""
    scanned = np.zeros(len(book_scores), dtype=np.uint8)
    current_day = 0
    score = np.int64(0)
    selected_count = 0
    assigned_count = 0

    for idx in range(len(order)):
        lib_id = order[idx]
        signup = libs_signup[lib_id]
        if current_day + signup >= total_days:
            continue

        current_day += signup
        remaining_days = total_days - current_day
        capacity = remaining_days * libs_rate[lib_id]
        if capacity <= 0:
            continue

        offset = books_offsets[lib_id]
        length = books_lengths[lib_id]
        taken = 0
        start_assigned = assigned_count
        for k in range(length):
            if taken >= capacity:
                break
            book_id = books_flat[offset + k]
            if scanned[book_id] == 0:
                scanned[book_id] = 1
                score += book_scores[book_id]
                out_books[assigned_count] = book_id
                out_book_libs[assigned_count] = lib_id
                assigned_count += 1
                taken += 1

        if assigned_count > start_assigned:
            out_selected[selected_count] = lib_id
            selected_count += 1

    return score, selected_count, assigned_count


def warmup_jit(flat):
    """Trigger Numba compilation before timed runs."""
    if not HAS_NUMBA:
        return
    dummy = np.array([0], dtype=np.int32)
    fast_evaluate(dummy, flat['libs_signup'], flat['libs_rate'],
                  flat['books_flat'], flat['books_offsets'],
                  flat['books_lengths'], flat['book_scores'],
                  flat['total_days'])
    fast_evaluate_global(dummy, flat['libs_signup'], flat['libs_rate'],
                         flat['lib_num_books'], flat['books_by_score'],
                         flat['book_libs_flat'], flat['book_libs_offsets'],
                         flat['book_libs_lengths'], flat['book_scores'],
                         flat['total_days'])
    fast_evaluate_sequential(dummy, flat['libs_signup'], flat['libs_rate'],
                             flat['books_flat'], flat['books_offsets'],
                             flat['books_lengths'], flat['book_scores'],
                             flat['total_days'])
    scanned = np.zeros(max(1, flat['n_books']), dtype=np.uint8)
    touched_books = np.zeros(max(1, flat['n_books']), dtype=np.int32)
    active = np.zeros(max(1, flat['n_libs']), dtype=np.uint8)
    remaining_capacity = np.zeros(max(1, flat['n_libs']), dtype=np.int32)
    remaining_candidates = np.zeros(max(1, flat['n_libs']), dtype=np.int32)
    positions = np.zeros(max(1, flat['n_libs']), dtype=np.int32)
    touched_libs = np.zeros(max(1, flat['n_libs']), dtype=np.int32)
    fast_evaluate_sequential_workspace(
        dummy, flat['libs_signup'], flat['libs_rate'],
        flat['books_flat'], flat['books_offsets'],
        flat['books_lengths'], flat['book_scores'],
        flat['total_days'], scanned, touched_books)
    fast_evaluate_global_workspace(
        dummy, flat['libs_signup'], flat['libs_rate'],
        flat['lib_num_books'], flat['books_by_score'],
        flat['book_libs_flat'], flat['book_libs_offsets'],
        flat['book_libs_lengths'], flat['book_scores'],
        flat['total_days'], active, remaining_capacity,
        remaining_candidates, positions, touched_libs)
    fast_evaluate_balanced_global_workspace(
        dummy, flat['libs_signup'], flat['libs_rate'],
        flat['lib_num_books'], flat['books_by_score'],
        flat['book_libs_flat'], flat['book_libs_offsets'],
        flat['book_libs_lengths'], flat['book_scores'],
        flat['total_days'], active, remaining_capacity,
        positions, touched_libs)
    out_selected = np.zeros(1, dtype=np.int32)
    out_books = np.zeros(max(1, flat['n_books']), dtype=np.int32)
    out_book_libs = np.zeros(max(1, flat['n_books']), dtype=np.int32)
    fast_evaluate_detailed_global(dummy, flat['libs_signup'], flat['libs_rate'],
                                  flat['lib_num_books'], flat['books_by_score'],
                                  flat['book_libs_flat'], flat['book_libs_offsets'],
                                  flat['book_libs_lengths'], flat['book_scores'],
                                  flat['total_days'],
                                  out_selected, out_books, out_book_libs)
    fast_evaluate_detailed_balanced_global(
        dummy, flat['libs_signup'], flat['libs_rate'],
        flat['lib_num_books'], flat['books_by_score'],
        flat['book_libs_flat'], flat['book_libs_offsets'],
        flat['book_libs_lengths'], flat['book_scores'],
        flat['total_days'],
        out_selected, out_books, out_book_libs)
    fast_evaluate_detailed(dummy, flat['libs_signup'], flat['libs_rate'],
                           flat['books_flat'], flat['books_offsets'],
                           flat['books_lengths'], flat['book_scores'],
                           flat['total_days'],
                           out_selected, out_books, out_book_libs)
