import glob
import os
from concurrent.futures import ProcessPoolExecutor

from models import Parser
#from models.InstanceDataAdapter import InstanceDataAdapter
from models.instance_data_adapter import InstanceDataAdapter
from models.initial_solution import InitialSolution
from models.genetic_solver import GeneticSolver
from models.ils_solver import ILS_Solver

INPUT_INSTANCES_DIR = 'input'
OUTPUT_INSTANCES_DIR = 'output'

MINUTES_TO_RUN = 1
NUM_CORES = 1


def prepare_everything(instance_paths):
    prepared_data = {}

    for instance_path in instance_paths:
        parser = Parser(instance_path)
        instance = parser.parse()  # Original instance
        adapted_instance = InstanceDataAdapter(instance)  # Adapter

        initial_solution, _ = InitialSolution.generate_initial_solution(adapted_instance)

        prepared_data[instance_path] = {
            'adapter': adapted_instance,
            'original_instance': instance,  # STORE ORIGINAL
            'initial_solution': initial_solution
        }

    return prepared_data
# Keep your original run_solver as-is, but create this wrapper:

def run_solver_optimized(version: str, instance_path: str, adapted_instances_cache={}, original_instances_cache={}):
    if instance_path not in adapted_instances_cache:
        print("Parsing instance file...")
        parser = Parser(instance_path)
        instance = parser.parse()
        original_instances_cache[instance_path] = instance  # Store original
        adapted_instances_cache[instance_path] = InstanceDataAdapter(instance)

    adapted_instance = adapted_instances_cache[instance_path]
    original_instance = original_instances_cache[instance_path]  # Retrieve original
    return run_solver(version, instance_path, adapted_instance, original_instance)  # Pass both


def run_solver(version: str, instance_path: str, adapted_instance=None, original_instance=None, initial_solution=None) -> None:
    print("🔧 Starting run_solver...")

    output_sub_dir = os.path.join(OUTPUT_INSTANCES_DIR, version)
    os.makedirs(output_sub_dir, exist_ok=True)
    print("✅ Output directory created")

    # Only parse and create adapter if not provided
    if adapted_instance is None:
        print("📖 Parsing instance file...")
        parser = Parser(instance_path)
        instance = parser.parse()
        print(f"✅ Instance parsed: {instance.num_libs} libs, {instance.num_books} books")

        print("🔧 Creating adapter...")
        adapted_instance = InstanceDataAdapter(instance)
        print("✅ Adapter created")
    else:
        print(f"✅ Using pre-adapted instance: {adapted_instance.num_libs} libs, {adapted_instance.num_books} books")

    # print("🎯 Generating initial solution...")
    # # FIX: Handle the tuple return value
    # initial_solution, candidates = InitialSolution.generate_initial_solution(adapted_instance)
    # print(f"✅ Initial solution generated with fitness: {initial_solution.fitness_score}")
    if initial_solution is None:
        print("Generating initial solution...")
        initial_solution, candidates = InitialSolution.generate_initial_solution(adapted_instance)
    else:
        print(f"Using pre-generated initial solution: {initial_solution.fitness_score}")

    print("Creating GeneticSolver...")
    genetic_solver = GeneticSolver(
        initial_solution=initial_solution,
        instance=adapted_instance,
        original_instance=original_instance,  # ADD THIS
        time_limit_sec=60
    )
    print("✅ GeneticSolver created")

    print("🚀 Starting solve...")
    solution = genetic_solver.solve()
    score = solution.fitness_score
    print(f"🏁 Solve completed with score: {score}")

    instance_name = os.path.basename(instance_path)
    print(instance_name, score, f'version: {version}')
    output_file = os.path.join(output_sub_dir, instance_name)
    solution.export(output_file)


def main():
    instance_paths = glob.glob(f'{INPUT_INSTANCES_DIR}/*.txt')

    # Pre-prepare everything ONCE before spawning workers
    print("Pre-preparing all instances...")
    prepared = {}
    for path in instance_paths:
        parser = Parser(path)
        instance = parser.parse()
        adapter = InstanceDataAdapter(instance)
        prepared[path] = {
            'adapter': adapter,
            'original_instance': instance
        }
    print(f"Pre-preparation complete for {len(prepared)} instances")

    jobs = []
    for v in range(1, 11):
        version = f'v{v}'
        for path in instance_paths:
            jobs.append((version, path, prepared[path]))

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [
            executor.submit(run_solver_with_prepared, version, path, prep)
            for version, path, prep in jobs
        ]
        for future in futures:
            future.result()

def run_solver_with_prepared(version: str, instance_path: str, prepared: dict) -> None:
    run_solver(
        version,
        instance_path,
        adapted_instance=prepared['adapter'],
        original_instance=prepared['original_instance']
    )

if __name__ == '__main__':
    main()
