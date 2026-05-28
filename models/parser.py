"""Input parser for Book Scanning instances."""

import sys

from .instance_data import InstanceData
from .library import Library


class Parser:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self):
        Library._id_counter = 0
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                return self._parse_stream(file)
        except FileNotFoundError:
            self._exit_with_error(f"File not found: {self.file_path}")
        except PermissionError:
            self._exit_with_error(f"Permission denied when accessing: {self.file_path}")
        except ValueError as exc:
            self._exit_with_error(f"Error parsing file: {exc}")
        except Exception as exc:
            self._exit_with_error(f"Unexpected error when parsing file: {exc}")

    def _parse_stream(self, file):
        first_line = file.readline().strip()
        if not first_line:
            raise ValueError("file is empty or first line is missing")

        parts = first_line.split()
        if len(parts) != 3:
            raise ValueError(
                f"first line should contain exactly 3 integers, got {len(parts)}"
            )

        try:
            num_books, num_libs, num_days = map(int, parts)
        except ValueError as exc:
            raise ValueError("first line should contain integers only") from exc

        if num_books < 0 or num_libs < 0 or num_days < 0:
            raise ValueError(
                "all header values must be non-negative: "
                f"books={num_books}, libraries={num_libs}, days={num_days}"
            )

        scores_line = file.readline().strip()
        if not scores_line:
            raise ValueError("book scores line is missing")

        try:
            scores = list(map(int, scores_line.split()))
        except ValueError as exc:
            raise ValueError("book scores must be integers") from exc

        if len(scores) != num_books:
            raise ValueError(f"expected {num_books} book scores, got {len(scores)}")

        libraries = []
        for lib_index in range(num_libs):
            books_count, signup_days, books_per_day = self._parse_library_header(
                file, lib_index
            )
            books = self._parse_library_books(file, lib_index, books_count, num_books)
            libraries.append(
                Library(books_count, signup_days, books_per_day, books, scores)
            )

        return InstanceData(num_books, num_libs, num_days, scores, libraries)

    def _parse_library_header(self, file, lib_index):
        lib_header = file.readline().strip()
        if not lib_header:
            raise ValueError(f"library {lib_index} header is missing")

        parts = lib_header.split()
        if len(parts) != 3:
            raise ValueError(
                f"library {lib_index} header should contain exactly 3 integers"
            )

        try:
            books_count, signup_days, books_per_day = map(int, parts)
        except ValueError as exc:
            raise ValueError(
                f"library {lib_index} header must contain integers only"
            ) from exc

        if books_count < 0 or signup_days < 0 or books_per_day < 0:
            raise ValueError(f"library {lib_index} values must be non-negative")

        return books_count, signup_days, books_per_day

    def _parse_library_books(self, file, lib_index, books_count, num_books):
        books_line = file.readline().strip()
        if not books_line:
            raise ValueError(f"book list for library {lib_index} is missing")

        try:
            books = list(map(int, books_line.split()))
        except ValueError as exc:
            raise ValueError(
                f"book IDs for library {lib_index} must be integers"
            ) from exc

        if len(books) != books_count:
            raise ValueError(
                f"library {lib_index} should have {books_count} books, got {len(books)}"
            )

        if any(book_id < 0 or book_id >= num_books for book_id in books):
            raise ValueError(f"library {lib_index} contains invalid book ID(s)")

        return books

    @staticmethod
    def _exit_with_error(message):
        print(message)
        sys.exit(1)
