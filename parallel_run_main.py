import glob
import os
from concurrent.futures import ProcessPoolExecutor

from models import Parser
from models.initial_solution import InitialSolution
from models.genetic_solver import GeneticSolver
from models.ils_solver import ILS_Solver

INPUT_INSTANCES_DIR = 'input'
OUTPUT_INSTANCES_DIR = 'output'

MINUTES_TO_RUN = 1
NUM_CORES = 1


def run_solver(version: str, instance_path: str) -> None:
    output_sub_dir = os.path.join(OUTPUT_INSTANCES_DIR, version)
    os.makedirs(output_sub_dir, exist_ok=True)

    parser = Parser(instance_path)
    instance = parser.parse()
    initial_solution = InitialSolution.generate_initial_solution(instance)
    genetic_solver = GeneticSolver(initial_solution=initial_solution, 
                                    instance=instance,
                                    time_limit_sec=60)
                                   #MINUTES_TO_RUN * 60)
    solution = genetic_solver.solve()
    score = solution.fitness_score
#Ketu nje mundesi integrimi - bb
    #solver = ILS_Solver()
    ##parser = Parser(instance_path)
    ##data = parser.parse()
    #result = solver.iterated_local_search(
     #   instance, #data,
      #  time_limit=60,# MINUTES_TO_RUN * 60,
       # max_iterations=1000
    #)
    # - bb

    instance_name = os.path.basename(instance_path)
    print(instance_name, score, f'version: {version}')
    output_file = os.path.join(output_sub_dir, instance_name)
    solution.export(output_file)


def main():
    instance_paths = glob.glob(f'{INPUT_INSTANCES_DIR}/*.txt')
    jobs = []

    for v in range(1, 11):
        version = f'v{v}'
        for path in instance_paths:
            jobs.append((version, path))

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(run_solver, version, path) for version, path in jobs]

        for future in futures:
            future.result()  


if __name__ == '__main__':
    main()
