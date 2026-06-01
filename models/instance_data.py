import numpy as np


class InstanceData:
    def __init__(self, num_books, num_libs, num_days, scores, libs):
        self.num_books = num_books
        self.num_libs = num_libs
        self.num_days = num_days
        self.scores = scores
        self.libs = libs
        self.book_libs = [[] for _ in range(num_books)]
        self.lib_signup_days = [0] * num_libs
        self.lib_books_per_day = [0] * num_libs
        self.lib_num_books = [0] * num_libs
        self.lib_book_ids = [[] for _ in range(num_libs)]
        for i, lib in enumerate(libs):
            self.lib_signup_days[i] = lib.signup_days
            self.lib_books_per_day[i] = lib.books_per_day
            self.lib_num_books[i] = lib.num_books
            book_ids = [book.id for book in lib.books]
            self.lib_book_ids[i] = book_ids
            for book in lib.books:
                self.book_libs[book.id].append(i)
        self.books_by_score = sorted(
            range(num_books), key=lambda bid: scores[bid], reverse=True)
        self.book_freq = [len(self.book_libs[bid]) for bid in range(num_books)]
        self.effective_scores = [
            scores[bid] * (1.0 + 0.35 / (self.book_freq[bid] ** 0.5 if self.book_freq[bid] > 0 else 1.0))
            for bid in range(num_books)
        ]
        self.top_raw_potential = [0.0] * num_libs
        self.top_rare_potential = [0.0] * num_libs
        self.cap_raw_potential = [0.0] * num_libs
        self.cap_rare_potential = [0.0] * num_libs
        self._build_potentials()

        self._flat = None
        self._rebuild_workspace = None
        self._evaluation_workspace = None
        self.assignment_mode = "portfolio"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_flat"] = None
        state["_rebuild_workspace"] = None
        state["_evaluation_workspace"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _build_potentials(self):
        k_limit = 1000
        for lib_id in range(self.num_libs):
            book_ids = self.lib_book_ids[lib_id]
            top_limit = min(len(book_ids), k_limit)
            if top_limit > 0:
                top_slice = book_ids[:top_limit]
                self.top_raw_potential[lib_id] = float(sum(self.scores[bid] for bid in top_slice))
                self.top_rare_potential[lib_id] = float(sum(self.effective_scores[bid] for bid in top_slice))

            signup = self.lib_signup_days[lib_id]
            rate = self.lib_books_per_day[lib_id]
            cap_limit = min(len(book_ids), max(0, self.num_days - signup) * rate, k_limit)
            if cap_limit > 0:
                cap_slice = book_ids[:cap_limit]
                self.cap_raw_potential[lib_id] = float(sum(self.scores[bid] for bid in cap_slice))
                self.cap_rare_potential[lib_id] = float(sum(self.effective_scores[bid] for bid in cap_slice))

    def potential_array(self, mode):
        if mode == "top_raw":
            return self.top_raw_potential
        if mode == "top_rare":
            return self.top_rare_potential
        if mode == "cap_raw":
            return self.cap_raw_potential
        if mode == "cap_rare":
            return self.cap_rare_potential
        raise ValueError(f"Unknown potential mode: {mode}")

    def to_flat_arrays(self):
        """
        Convert the instance representation to flat NumPy arrays for
        Numba-accelerated evaluation. The result is cached after the first call.
        """
        if self._flat is not None:
            return self._flat

        book_scores = np.array(self.scores, dtype=np.int32)
        libs_signup = np.array(self.lib_signup_days, dtype=np.int32)
        libs_rate = np.array(self.lib_books_per_day, dtype=np.int32)
        lib_num_books = np.array(self.lib_num_books, dtype=np.int32)
        books_by_score = np.array(self.books_by_score, dtype=np.int32)

        flat_books = []
        offsets = []
        lengths = []
        for i in range(self.num_libs):
            offsets.append(len(flat_books))
            bids = self.lib_book_ids[i]
            lengths.append(len(bids))
            flat_books.extend(bids)

        book_libs_flat = []
        book_libs_offsets = []
        book_libs_lengths = []
        for book_id in range(self.num_books):
            book_libs_offsets.append(len(book_libs_flat))
            libs = self.book_libs[book_id]
            book_libs_lengths.append(len(libs))
            book_libs_flat.extend(libs)

        self._flat = {
            'n_books': self.num_books,
            'n_libs': self.num_libs,
            'total_days': self.num_days,
            'book_scores': book_scores,
            'libs_signup': libs_signup,
            'libs_rate': libs_rate,
            'lib_num_books': lib_num_books,
            'books_by_score': books_by_score,
            'books_flat': np.array(flat_books, dtype=np.int32),
            'books_offsets': np.array(offsets, dtype=np.int32),
            'books_lengths': np.array(lengths, dtype=np.int32),
            'book_libs_flat': np.array(book_libs_flat, dtype=np.int32),
            'book_libs_offsets': np.array(book_libs_offsets, dtype=np.int32),
            'book_libs_lengths': np.array(book_libs_lengths, dtype=np.int32),
        }
        return self._flat

    def rebuild_workspace(self):
        """
        Allocate reusable NumPy buffers for `Solution.from_order` rebuilds.
        This avoids repeated allocation of O(num_books) arrays during search.
        """
        if self._rebuild_workspace is None:
            max_order = max(1, self.num_libs)
            max_books = max(1, self.num_books)
            self._rebuild_workspace = {
                "signed_buffer": np.empty(max_order, dtype=np.int32),
                "out_selected": np.empty(max_order, dtype=np.int32),
                "out_books": np.empty(max_books, dtype=np.int32),
                "out_book_libs": np.empty(max_books, dtype=np.int32),
                "out_selected_global": np.empty(max_order, dtype=np.int32),
                "out_books_global": np.empty(max_books, dtype=np.int32),
                "out_book_libs_global": np.empty(max_books, dtype=np.int32),
                "out_selected_balanced": np.empty(max_order, dtype=np.int32),
                "out_books_balanced": np.empty(max_books, dtype=np.int32),
                "out_book_libs_balanced": np.empty(max_books, dtype=np.int32),
            }
        return self._rebuild_workspace

    def evaluation_workspace(self):
        """
        Allocate reusable buffers for fast scoring calls.

        Local search evaluates many neighboring orders. Reusing these arrays
        avoids allocating and clearing full book/library masks for every move.
        """
        if self._evaluation_workspace is None:
            max_books = max(1, self.num_books)
            max_libs = max(1, self.num_libs)
            self._evaluation_workspace = {
                "scanned": np.zeros(max_books, dtype=np.uint8),
                "touched_books": np.empty(max_books, dtype=np.int32),
                "active": np.zeros(max_libs, dtype=np.uint8),
                "remaining_capacity": np.zeros(max_libs, dtype=np.int32),
                "remaining_candidates": np.zeros(max_libs, dtype=np.int32),
                "positions": np.zeros(max_libs, dtype=np.int32),
                "touched_libs": np.empty(max_libs, dtype=np.int32),
            }
        return self._evaluation_workspace

    def use_balanced_assignment(self):
        """
        Enable the extra global assignment mode where it is cheap enough.

        The balanced mode is most valuable on small/medium library-count
        instances with heavy book overlap. Keeping it off for very large
        library-count instances avoids slowing the high-frequency local-search
        exact checks.
        """
        return self.num_libs <= 3000

    def use_sequential_assignment_only(self):
        """Use only the official sequential library-order scorer."""
        return self.assignment_mode == "sequential"

    def order_array_view(self, signed_order):
        workspace = self.rebuild_workspace()
        signed_buffer = workspace["signed_buffer"]
        order_len = len(signed_order)
        if order_len:
            signed_buffer[:order_len] = signed_order
        return signed_buffer[:order_len]

    def fast_evaluate(self, signed_order):
        """
        Evaluate a signed library ordering using Numba-accelerated code.
        A pure Python implementation is used when Numba is unavailable.
        """
        from models.evaluation import (
            fast_evaluate_balanced_global_workspace,
            fast_evaluate_global_workspace,
            fast_evaluate_sequential_workspace,
        )
        flat = self.to_flat_arrays()
        workspace = self.evaluation_workspace()
        signed_arr = self.order_array_view(signed_order)
        sequential_score = int(fast_evaluate_sequential_workspace(
            signed_arr, flat['libs_signup'], flat['libs_rate'],
            flat['books_flat'], flat['books_offsets'],
            flat['books_lengths'], flat['book_scores'],
            flat['total_days'], workspace["scanned"],
            workspace["touched_books"]))
        if self.use_sequential_assignment_only():
            return sequential_score

        global_score = int(fast_evaluate_global_workspace(
            signed_arr, flat['libs_signup'], flat['libs_rate'],
            flat['lib_num_books'], flat['books_by_score'],
            flat['book_libs_flat'], flat['book_libs_offsets'],
            flat['book_libs_lengths'], flat['book_scores'],
            flat['total_days'], workspace["active"],
            workspace["remaining_capacity"],
            workspace["remaining_candidates"], workspace["positions"],
            workspace["touched_libs"]))
        if self.use_balanced_assignment():
            balanced_score = int(fast_evaluate_balanced_global_workspace(
                signed_arr, flat['libs_signup'], flat['libs_rate'],
                flat['lib_num_books'], flat['books_by_score'],
                flat['book_libs_flat'], flat['book_libs_offsets'],
                flat['book_libs_lengths'], flat['book_scores'],
                flat['total_days'], workspace["active"],
                workspace["remaining_capacity"], workspace["positions"],
                workspace["touched_libs"]))
            return max(sequential_score, global_score, balanced_score)
        return max(sequential_score, global_score)

    def screen_evaluate_sequential(self, signed_order):
        """
        Lowest-cost screening score used by local search.
        It lets the search evaluate substantially more moves per second before
        paying for the exact fast scorer and full solution rebuild.
        """
        from models.evaluation import fast_evaluate_sequential_workspace
        flat = self.to_flat_arrays()
        workspace = self.evaluation_workspace()
        signed_arr = self.order_array_view(signed_order)
        return int(fast_evaluate_sequential_workspace(
            signed_arr,
            flat['libs_signup'],
            flat['libs_rate'],
            flat['books_flat'],
            flat['books_offsets'],
            flat['books_lengths'],
            flat['book_scores'],
            flat['total_days'],
            workspace["scanned"],
            workspace["touched_books"],
        ))

    def calculate_upper_bound(self):
        """Naive upper bound: score of every distinct book available in libraries."""
        seen = bytearray(self.num_books)
        upper_bound = 0
        for book_ids in self.lib_book_ids:
            for book_id in book_ids:
                if not seen[book_id]:
                    seen[book_id] = 1
                    upper_bound += self.scores[book_id]
        return upper_bound

