class InstanceData:
    num_books = 0
    num_libs = 0
    num_days = 0
    scores = []
    libs = []
    book_libs = []
    upper_bound = 0

    def __init__(self, num_books, num_libs, num_days, scores, libs):
        self.num_books = num_books
        self.num_libs = num_libs
        self.num_days = num_days
        self.scores = scores
        self.libs = libs

        # ── Original book_libs: list-of-lists indexed by book_id ─────────────
        self.book_libs = [[] for _ in range(num_books)]
        for i, lib in enumerate(libs):
            for book in lib.books:
                self.book_libs[book.id].append(i)

        # ── Per-library flat data (mirrors InstanceDataAdapter) ───────────────
        self.lib_book_ids      = [[book.id for book in lib.books] for lib in libs]
        self.lib_signup_days   = [lib.signup_days     for lib in libs]
        self.lib_books_per_day = [lib.books_per_day   for lib in libs]
        self.lib_num_books     = [len(lib.books)       for lib in libs]

        # ── Book frequency ────────────────────────────────────────────────────
        self.book_freq = {}
        for lib_books in self.lib_book_ids:
            for book_id in lib_books:
                self.book_freq[book_id] = self.book_freq.get(book_id, 0) + 1

        # ── Rarity-weighted scores ────────────────────────────────────────────
        self.effective_scores = []
        for book_id in range(num_books):
            freq = self.book_freq.get(book_id, 1)
            self.effective_scores.append(scores[book_id] / max(1, freq))

        # ── books_by_score: book IDs sorted by descending score ───────────────
        valid_book_ids = set(self.book_freq.keys())
        self.books_by_score = sorted(
            valid_book_ids,
            key=lambda bid: scores[bid] if bid < len(scores) else 0,
            reverse=True,
        )

        # ── Pre-sort books within each library by descending score ────────────
        self.lib_book_ids = [
            sorted(
                lib_books,
                key=lambda bid: scores[bid] if bid < len(scores) else 0,
                reverse=True,
            )
            for lib_books in self.lib_book_ids
        ]

        # ── Flat array cache (built lazily, once) ─────────────────────────────
        self._flat_cache = None

    # ── Flat array builder ────────────────────────────────────────────────────
    def to_flat_arrays(self):
        import numpy as np

        books_flat, books_offsets, books_lengths = [], [], []
        for lib_books in self.lib_book_ids:
            books_offsets.append(len(books_flat))
            books_lengths.append(len(lib_books))
            books_flat.extend(lib_books)

        book_libs_flat, book_libs_offsets, book_libs_lengths = [], [], []
        for book_id in range(self.num_books):
            libs_for_book = self.book_libs[book_id]
            book_libs_offsets.append(len(book_libs_flat))
            book_libs_lengths.append(len(libs_for_book))
            book_libs_flat.extend(libs_for_book)

        return {
            # Scalars
            'n_libs':            np.int32(self.num_libs),
            'n_books':           np.int32(self.num_books),
            'total_days':        np.int32(self.num_days),
            # Per-library
            'libs_signup':       np.array(self.lib_signup_days,   dtype=np.int32),
            'libs_rate':         np.array(self.lib_books_per_day, dtype=np.int32),
            'lib_num_books':     np.array(self.lib_num_books,     dtype=np.int32),
            # Flat books-per-library
            'books_flat':        np.array(books_flat,             dtype=np.int32),
            'books_offsets':     np.array(books_offsets,          dtype=np.int32),
            'books_lengths':     np.array(books_lengths,          dtype=np.int32),
            # Per-book
            'book_scores':       np.array(self.scores,            dtype=np.int32),
            'books_by_score':    np.array(self.books_by_score,    dtype=np.int32),
            # Book → libraries (flat)
            'book_libs_flat':    np.array(book_libs_flat,         dtype=np.int32),
            'book_libs_offsets': np.array(book_libs_offsets,      dtype=np.int32),
            'book_libs_lengths': np.array(book_libs_lengths,      dtype=np.int32),
            # Frequency
            'book_freq':         np.array(
                [self.book_freq.get(b, 0) for b in range(self.num_books)],
                dtype=np.int32,
            ),
        }

    def _get_flat(self):
        """Return cached flat arrays — built exactly once per object."""
        if self._flat_cache is None:
            self._flat_cache = self.to_flat_arrays()
        return self._flat_cache

    # ── Evaluation methods required by local_search.py ────────────────────────
    def screen_evaluate_sequential(self, order):
        """
        Fast proxy score for the local search main loop.
        Calls the JIT-accelerated fast_evaluate_sequential().
        """
        import numpy as np
        from models.evaluation import fast_evaluate_sequential

        flat = self._get_flat()
        order_arr = np.array(order, dtype=np.int32)
        return int(fast_evaluate_sequential(
            order_arr,
            flat['libs_signup'],
            flat['libs_rate'],
            flat['books_flat'],
            flat['books_offsets'],
            flat['books_lengths'],
            flat['book_scores'],
            flat['total_days'],
        ))

    def screen_evaluate(self, order):
        """
        Exact sequential score for the polish phase.
        Calls the JIT-accelerated fast_evaluate().
        """
        import numpy as np
        from models.evaluation import fast_evaluate

        flat = self._get_flat()
        order_arr = np.array(order, dtype=np.int32)
        return int(fast_evaluate(
            order_arr,
            flat['libs_signup'],
            flat['libs_rate'],
            flat['books_flat'],
            flat['books_offsets'],
            flat['books_lengths'],
            flat['book_scores'],
            flat['total_days'],
        ))

    def potential_array(self, mode):
        """
        Compute per-library potential scores for greedy constructors.
        Mirrors the identical method in InstanceDataAdapter.
        Supported modes: 'top_raw', 'cap_raw', 'top_rare', 'cap_rare'
        """
        potentials = []
        for lib_id in range(self.num_libs):
            lib_books = self.lib_book_ids[lib_id]

            if mode == "top_raw":
                potential = sum(
                    self.scores[book_id]
                    for book_id in lib_books
                    if book_id < len(self.scores)
                )

            elif mode == "cap_raw":
                max_books = self.lib_books_per_day[lib_id] * self.num_days
                book_scores = sorted(
                    (self.scores[book_id]
                     for book_id in lib_books
                     if book_id < len(self.scores)),
                    reverse=True,
                )
                potential = sum(book_scores[:max_books])

            elif mode == "top_rare":
                potential = sum(
                    self.effective_scores[book_id]
                    for book_id in lib_books
                    if book_id < len(self.effective_scores)
                )

            elif mode == "cap_rare":
                max_books = self.lib_books_per_day[lib_id] * self.num_days
                weighted_scores = sorted(
                    (self.effective_scores[book_id]
                     for book_id in lib_books
                     if book_id < len(self.effective_scores)),
                    reverse=True,
                )
                potential = sum(weighted_scores[:max_books])

            else:
                potential = 1000.0  # safe fallback for unknown modes

            potentials.append(potential)

        return potentials

    # ── Utility methods (unchanged from original) ─────────────────────────────
    def describe(self):
        print('There are', self.num_books, "books", self.num_libs,
              "libraries", "and", self.num_days, "days for scanning")
        print('The scores of the books are',
              ','.join(str(x) for x in self.scores), "(in order)")
        print()
        for i, l in enumerate(self.libs):
            print(f'Library {l.id} has {l.num_books} books, the signup '
                  f'process takes {l.signup_days} days, and the library '
                  f'can ship {l.books_per_day} books per day.')
            print(f'The books in library {l.id} are: '
                  + ', '.join(f'book {x}' for x in l.books[:-1])
                  + f', and book {l.books[-1]}.')
        print()
        for i, l in enumerate(self.book_libs):
            print(f'Book {i} Exists in Libraries:',
                  ' and '.join(str(x) for x in l))

    def calculate_upper_bound(self):
        """Sum of scores of all unique books across all libraries."""
        unique_books = set()
        for lib in self.libs:
            for book in lib.books:
                unique_books.add(book.id)
        return sum(self.scores[book_id] for book_id in unique_books)


# ── Sanity check — will print on every import ─────────────────────────────────
assert hasattr(InstanceData, 'screen_evaluate_sequential'), \
    "CRITICAL: screen_evaluate_sequential missing from InstanceData!"
assert hasattr(InstanceData, 'screen_evaluate'), \
    "CRITICAL: screen_evaluate missing from InstanceData!"
