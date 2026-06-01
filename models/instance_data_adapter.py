class InstanceDataAdapter:
    """Complete minimal adapter with all required attributes"""

    def __init__(self, instance_data):
        print("🔧 Creating complete minimal adapter...")
        self.original = instance_data

        print("🔄 Pre-computing all data...")

        self._num_libs = instance_data.num_libs
        self._num_books = instance_data.num_books
        self._num_days = instance_data.num_days

        self._lib_book_ids = [[book.id for book in lib.books] for lib in instance_data.libs]
        self._lib_signup_days = [lib.signup_days for lib in instance_data.libs]
        self._lib_books_per_day = [lib.books_per_day for lib in instance_data.libs]
        self._scores = list(instance_data.scores)

        # lib_num_books
        self._lib_num_books = [len(lib.books) for lib in instance_data.libs]

        # libs property for backward compatibility
        self._libs = instance_data.libs

        # Compute book frequencies
        print("🔄 Computing book frequencies...")
        self._book_freq = {}
        for lib_books in self._lib_book_ids:
            for book_id in lib_books:
                self._book_freq[book_id] = self._book_freq.get(book_id, 0) + 1

        # Compute effective scores (rarity-weighted)
        print("🔄 Computing effective scores...")
        self._effective_scores = []
        for book_id in range(len(self._scores)):
            if book_id in self._book_freq:
                effective_score = self._scores[book_id] / max(1, self._book_freq[book_id])
            else:
                effective_score = self._scores[book_id]
            self._effective_scores.append(effective_score)

        # FIX: Build book_libs mapping FIRST
        print("🔄 Building book_libs mapping...")
        self._book_libs = {}
        for lib_id, lib_books in enumerate(self._lib_book_ids):
            for book_id in lib_books:
                if book_id not in self._book_libs:
                    self._book_libs[book_id] = []
                self._book_libs[book_id].append(lib_id)

        # FIX: Only include book IDs that exist in libraries
        print("🔄 Building books_by_score...")
        valid_book_ids = set(self._book_libs.keys())
        self._books_by_score = sorted(
            valid_book_ids,  # Only books that exist in libraries
            key=lambda book_id: self._scores[book_id] if book_id < len(self._scores) else 0,
            reverse=True
        )

        # Pre-sort books by score in each library for efficiency
        print("🔄 Pre-sorting books by score in libraries...")
        self._lib_book_ids = [
            sorted(lib_books, key=lambda bid: self._scores[bid] if bid < len(self._scores) else 0, reverse=True)
            for lib_books in self._lib_book_ids
        ]

        print("✅ Complete minimal adapter created")
        # Add at the end of InstanceDataAdapter.__init__():
        print(f"📊 Validation:")
        print(f"  Total books: {len(self._scores)}")
        print(f"  Books in libraries: {len(self._book_libs)}")
        print(f"  Books in books_by_score: {len(self._books_by_score)}")

        # Verify all books in books_by_score exist in book_libs
        for book_id in self._books_by_score[:5]:  # Check first 5
            assert book_id in self._book_libs, f"Book {book_id} in books_by_score but not in book_libs"

        print("✅ Validation passed")

    def to_flat_arrays(self):
        """Convert to flat arrays format required by solution._rebuild_fast()"""
        import numpy as np

        # Build flattened book arrays per library
        books_flat = []
        books_offsets = []
        books_lengths = []

        for lib_books in self._lib_book_ids:
            books_offsets.append(len(books_flat))
            books_lengths.append(len(lib_books))
            books_flat.extend(lib_books)

        # Build book_libs flat arrays (which libraries contain each book)
        book_libs_flat = []
        book_libs_offsets = []
        book_libs_lengths = []

        for book_id in range(self._num_books):
            libs_for_book = self._book_libs.get(book_id, [])
            book_libs_offsets.append(len(book_libs_flat))
            book_libs_lengths.append(len(libs_for_book))
            book_libs_flat.extend(libs_for_book)

        return {
            # Scalars
            'n_libs': np.int32(self._num_libs),
            'n_books': np.int32(self._num_books),
            'total_days': np.int32(self._num_days),

            # Per-library arrays
            'libs_signup': np.array(self._lib_signup_days, dtype=np.int32),
            'libs_rate': np.array(self._lib_books_per_day, dtype=np.int32),
            'lib_num_books': np.array(self._lib_num_books, dtype=np.int32),

            # Flat book arrays (per library)
            'books_flat': np.array(books_flat, dtype=np.int32),
            'books_offsets': np.array(books_offsets, dtype=np.int32),
            'books_lengths': np.array(books_lengths, dtype=np.int32),

            # Per-book arrays
            'book_scores': np.array(self._scores, dtype=np.int32),
            'books_by_score': np.array(self._books_by_score, dtype=np.int32),  # ← ADDED

            # Book → libraries mapping (flat)
            'book_libs_flat': np.array(book_libs_flat, dtype=np.int32),
            'book_libs_offsets': np.array(book_libs_offsets, dtype=np.int32),
            'book_libs_lengths': np.array(book_libs_lengths, dtype=np.int32),

            # Book frequency
            'book_freq': np.array(
                [self._book_freq.get(b, 0) for b in range(self._num_books)],
                dtype=np.int32
            ),
        }


    # Properties
    @property
    def num_libs(self):
        return self._num_libs

    @property
    def num_books(self):
        return self._num_books

    @property
    def num_days(self):
        return self._num_days

    @property
    def lib_book_ids(self):
        return self._lib_book_ids

    @property
    def lib_signup_days(self):
        return self._lib_signup_days

    @property
    def lib_books_per_day(self):
        return self._lib_books_per_day

    @property
    def scores(self):
        return self._scores

    @property
    def effective_scores(self):
        return self._effective_scores

    @property
    def book_freq(self):
        return self._book_freq

    # ADD: Missing properties
    @property
    def lib_num_books(self):
        return self._lib_num_books

    @property
    def libs(self):
        return self._libs

    # ADD: Missing properties for Solution._assign_books_global
    @property
    def books_by_score(self): return self._books_by_score

    @property
    def book_libs(self): return self._book_libs

    # Methods
    def potential_array(self, mode):
        print(f"📊 Computing potential_array: {mode}")

        potentials = []
        for lib_id in range(self.num_libs):
            lib_books = self._lib_book_ids[lib_id]

            if mode == "top_raw":
                potential = sum(self._scores[book_id] if book_id < len(self._scores) else 0
                                for book_id in lib_books)
            elif mode == "cap_raw":
                max_books = self._lib_books_per_day[lib_id] * self._num_days
                book_scores = sorted([self._scores[book_id] if book_id < len(self._scores) else 0
                                      for book_id in lib_books], reverse=True)
                potential = sum(book_scores[:max_books])
            elif mode == "top_rare":
                potential = sum(self._effective_scores[book_id] if book_id < len(self._effective_scores) else 0
                                for book_id in lib_books)
            elif mode == "cap_rare":
                max_books = self._lib_books_per_day[lib_id] * self._num_days
                weighted_scores = sorted(
                    [self._effective_scores[book_id] if book_id < len(self._effective_scores) else 0
                     for book_id in lib_books], reverse=True)
                potential = sum(weighted_scores[:max_books])
            else:
                potential = 1000.0  # Default fallback

            potentials.append(potential)

        print(f"✅ potential_array completed: {mode}")
        return potentials

    def fast_evaluate(self, order):
        """Realistic fast evaluation using pre-computed data"""
        total_score = 0
        used_books = set()
        day = 0

        for lib_id in order:
            signup = self._lib_signup_days[lib_id]
            if day + signup >= self._num_days:
                continue

            day += signup
            remaining_days = self._num_days - day
            max_books = remaining_days * self._lib_books_per_day[lib_id]
            if max_books <= 0:
                continue

            count = 0
            # Books are pre-sorted in adapter for efficiency
            for book_id in self._lib_book_ids[lib_id]:
                if book_id in used_books:
                    continue
                if book_id < len(self._scores):
                    total_score += self._scores[book_id]
                used_books.add(book_id)
                count += 1
                if count >= max_books:
                    break

        return total_score

        # ── Cached flat arrays ───────────────────────────────────────────────────
        _flat_cache = None

        def _get_flat(self):
            """Return cached flat arrays, building once on first call."""
            if self._flat_cache is None:
                self._flat_cache = self.to_flat_arrays()
            return self._flat_cache

        def screen_evaluate_sequential(self, order):
            """
            Fast proxy score used by the local search main loop.
            Wraps fast_evaluate_sequential() from evaluation.py.
            """
            import numpy as np
            from models.evaluation import fast_evaluate_sequential

            flat = self._get_flat()
            order_arr = np.array(order, dtype=np.int32)
            return int(fast_evaluate_sequential(
                order_arr,
                flat["libs_signup"],
                flat["libs_rate"],
                flat["books_flat"],
                flat["books_offsets"],
                flat["books_lengths"],
                flat["book_scores"],
                flat["total_days"],
            ))

        def screen_evaluate(self, order):
            """
            Exact sequential score used by the polish phase.
            Wraps fast_evaluate() from evaluation.py.
            """
            import numpy as np
            from models.evaluation import fast_evaluate

            flat = self._get_flat()
            order_arr = np.array(order, dtype=np.int32)
            return int(fast_evaluate(
                order_arr,
                flat["libs_signup"],
                flat["libs_rate"],
                flat["books_flat"],
                flat["books_offsets"],
                flat["books_lengths"],
                flat["book_scores"],
                flat["total_days"],
            ))


    # Pickle support
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __reduce__(self):
        return (self.__class__, (self.original,))



