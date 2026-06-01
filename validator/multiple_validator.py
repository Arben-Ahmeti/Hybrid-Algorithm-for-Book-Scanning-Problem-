import os
from validator.validator import validate_solution


def validate_all_solutions(input_dir='input', output_dir='output'):
    print("\n=== Validating All Solutions ===")

    input_files = discover_input_files(input_dir)
    if not input_files:
        print(f"No input files found in {input_dir}")
        return False

    valid_count = 0
    invalid_count = 0

    for input_path in input_files:
        relative_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)

        print(f"\nValidating {relative_path}...")
        if not os.path.exists(output_path):
            print("  Missing output file")
            invalid_count += 1
            continue

        result = validate_solution(input_path, output_path, isConsoleApplication=True)
        if result == "Valid":
            print("  Valid")
            valid_count += 1
        else:
            print("  Invalid")
            invalid_count += 1

    print("\n=== Validation Summary ===")
    print(f"Total files checked: {len(input_files)}")
    print(f"Valid solutions: {valid_count}")
    print(f"Invalid solutions: {invalid_count}")

    return valid_count == len(input_files)


def discover_input_files(input_dir):
    input_files = []
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()
        for name in sorted(files):
            if name.endswith('.txt'):
                input_files.append(os.path.join(root, name))
    return input_files


if __name__ == "__main__":
    validate_all_solutions()
