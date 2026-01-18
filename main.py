import argparse
import glob
import os

from models.initial_solution import InitialSolution
from models.genetic_solver import GeneticSolver
from models import Parser

INPUT_INSTANCES_DIR = 'input'
OUTPUT_INSTANCES_DIR = 'output'

MINUTES_TO_RUN = 10

def main(version: str) -> None:
    output_sub_dir = os.path.join(OUTPUT_INSTANCES_DIR, version)
    os.makedirs(output_sub_dir, exist_ok=True)

    instance_paths = glob.glob(f'{INPUT_INSTANCES_DIR}/*.txt')

    for instance_path in instance_paths:
        parser = Parser(instance_path)
        instance = parser.parse()
        initial_solution = InitialSolution.generate_initial_solution(instance)
        genetic_solver = GeneticSolver(initial_solution=initial_solution, 
                                       instance=instance,
                                       time_limit_sec=MINUTES_TO_RUN * 60)
        solution = genetic_solver.solve()
        score = solution.fitness_score

        instance_name = os.path.basename(instance_path)
        print(instance_name, score, f'version: {version}')
        output_file = os.path.join(output_sub_dir, instance_name)
        solution.export(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', type=str, required=True)

    args = parser.parse_args()
    main(args.version)
