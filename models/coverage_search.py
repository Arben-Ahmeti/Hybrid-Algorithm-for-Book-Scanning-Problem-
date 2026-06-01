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


@njit(cache=True)
def coverage_one_swap_core(
    book_scores,
    book_starts,
    flat_books,
    selected_in,
    selected_flag_in,
    cover_count_in,
    max_passes,
):
    book_count = len(cover_count_in)
    lib_count = len(selected_flag_in)
    selected_count = len(selected_in)

    selected = np.empty(lib_count, dtype=np.int64)
    for i in range(selected_count):
        selected[i] = selected_in[i]
    selected_flag = selected_flag_in.copy()
    cover_count = cover_count_in.copy()

    if selected_count == 0:
        return selected[:0].copy(), selected_flag, cover_count

    owner = np.empty(book_count, dtype=np.int64)
    loss = np.empty(lib_count, dtype=np.int64)
    stamp = np.zeros(lib_count, dtype=np.int64)
    overlap = np.zeros(lib_count, dtype=np.int64)
    touched = np.empty(lib_count, dtype=np.int64)
    current_stamp = 0

    for _pass in range(max_passes):
        owner.fill(-1)
        loss.fill(0)

        for selected_pos in range(selected_count):
            lib_id = selected[selected_pos]
            for pos in range(book_starts[lib_id], book_starts[lib_id + 1]):
                book_id = flat_books[pos]
                if cover_count[book_id] == 1:
                    owner[book_id] = lib_id
                    loss[lib_id] += book_scores[book_id]

        min_loss_lib = selected[0]
        min_loss = loss[min_loss_lib]
        for selected_pos in range(1, selected_count):
            lib_id = selected[selected_pos]
            lib_loss = loss[lib_id]
            if lib_loss < min_loss or (lib_loss == min_loss and lib_id < min_loss_lib):
                min_loss = lib_loss
                min_loss_lib = lib_id

        best_delta = 0
        best_add = -1
        best_remove = -1
        for add_idx in range(lib_count):
            if selected_flag[add_idx]:
                continue

            gain = 0
            touched_count = 0
            current_stamp += 1
            for pos in range(book_starts[add_idx], book_starts[add_idx + 1]):
                book_id = flat_books[pos]
                count = cover_count[book_id]
                if count == 0:
                    gain += book_scores[book_id]
                elif count == 1:
                    owned_by = owner[book_id]
                    if owned_by != -1:
                        if stamp[owned_by] != current_stamp:
                            stamp[owned_by] = current_stamp
                            overlap[owned_by] = book_scores[book_id]
                            touched[touched_count] = owned_by
                            touched_count += 1
                        else:
                            overlap[owned_by] += book_scores[book_id]

            if gain == 0 and touched_count == 0:
                continue

            adjusted_loss = min_loss
            remove_idx = min_loss_lib
            for touched_pos in range(touched_count):
                owned_by = touched[touched_pos]
                candidate_loss = loss[owned_by] - overlap[owned_by]
                if candidate_loss < adjusted_loss or (
                    candidate_loss == adjusted_loss and owned_by < remove_idx
                ):
                    adjusted_loss = candidate_loss
                    remove_idx = owned_by

            delta = gain - adjusted_loss
            if delta > best_delta:
                best_delta = delta
                best_add = add_idx
                best_remove = remove_idx

        if best_delta <= 0:
            break

        selected_flag[best_remove] = False
        for selected_pos in range(selected_count):
            if selected[selected_pos] == best_remove:
                selected_count -= 1
                selected[selected_pos] = selected[selected_count]
                break
        for pos in range(book_starts[best_remove], book_starts[best_remove + 1]):
            cover_count[flat_books[pos]] -= 1

        selected_flag[best_add] = True
        selected[selected_count] = best_add
        selected_count += 1
        for pos in range(book_starts[best_add], book_starts[best_add + 1]):
            cover_count[flat_books[pos]] += 1

    return selected[:selected_count].copy(), selected_flag, cover_count


def improve_coverage_one_swap(data, selected, selected_flag, cover_count, max_passes=120):
    total_books = sum(data.lib_num_books)
    book_starts = np.empty(data.num_libs + 1, dtype=np.int64)
    flat_books = np.empty(total_books, dtype=np.int64)
    pos = 0
    book_starts[0] = 0
    for lib_id in range(data.num_libs):
        books = data.lib_book_ids[lib_id]
        end = pos + len(books)
        flat_books[pos:end] = books
        pos = end
        book_starts[lib_id + 1] = pos

    selected_arr = np.asarray(selected, dtype=np.int64)
    selected_flag_arr = np.asarray(selected_flag, dtype=np.bool_)
    cover_count_arr = np.asarray(cover_count, dtype=np.int64)
    score_arr = np.asarray(data.scores, dtype=np.int64)

    out_selected, out_flag, out_cover = coverage_one_swap_core(
        score_arr,
        book_starts,
        flat_books,
        selected_arr,
        selected_flag_arr,
        cover_count_arr,
        max_passes,
    )
    return out_selected.tolist(), out_flag.tolist(), out_cover.tolist()
