import sys

from models import Parser
from models.initial_solution import InitialSolution
import os
from models.genetic_solver import GeneticSolver
from validator.multiple_validator import validate_all_solutions

def run_instances(output_dir='output'):
    print(output_dir)
    directory = os.listdir('input')
    results = []
    os.makedirs(output_dir, exist_ok=True)

    for file in directory:
        if file.endswith('.txt'):
            print(f'Computing ./input/{file}')
            parser = Parser(f'./input/{file}')
            instance = parser.parse()
            initial_solution = InitialSolution.generate_initial_solution(instance)
            genetic_solver = GeneticSolver(initial_solution=initial_solution, instance=instance)
            solution = genetic_solver.solve()
            score = solution.fitness_score
            results.append((file, score))
            print(f"Final score for {file}: {score:,}")
            output_file = os.path.join(output_dir, file)
            solution.export(output_file)
            print("----------------------")

    print("\nValidating all solutions...")
    validate_all_solutions(input_dir='input', output_dir=output_dir)

    # Print summary of all instances
    print("\nSummary of all instances:")
    print("-" * 50)
    print(f"{'Instance':<20} {'Score':>15}")
    print("-" * 50)
    for file, score in results:
        print(f"{file:<20} {score:>15,}")
    print("-" * 50)

    # Write summary to a text file
    summary_file = os.path.join(output_dir, 'summary_results.txt')
    with open(summary_file, 'w') as f:
        f.write("Summary of all instances:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Instance':<20} {'Score':>15}\n")
        f.write("-" * 50 + "\n")
        for file, score in results:
            f.write(f"{file:<20} {score:>15,}\n")
        f.write("-" * 50 + "\n")
    print(f"\nSummary has been written to: {summary_file}")


def main():
    if len(sys.argv) > 1:
        subdir = sys.argv[1]
        run_instances(f"./output/{subdir}")
    else:
        print("No argument provided. Saving outputs to ./output")
        run_instances()


if __name__ == "__main__":
    main()
