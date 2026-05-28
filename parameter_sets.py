"""Named solver configurations shared by the CLI and tuning scripts."""

from copy import deepcopy


ALPHA_POOLS = {
    "default": [0.5, 1.0, 1.5, 2.0],
    "explore": [0.4, 0.5, 0.75, 1.0, 1.5, 2.0],
}


DEFAULT_GA_PARAMS = {
    "population_size": 100,
    "generations": 500,
    "mutation_steps": 5,
    "mutation_prob": 0.39,
    "crossover_rate": 0.33,
    "immigrant_frac": 0.06,
    "local_refine_steps": 4,
    "elite_refine_interval": 5,
    "elite_refine_count": 2,
    "steady_state_ratio": 0.25,
    "mutation_accept_worse_prob": 0.08,
    "steady_state_offspring_factor": 0.55,
    "initial_mutation_ratio": 0.45,
    "mutation_screen_attempts": 0,
    "mutation_exact_checks": 2,
    "local_screen_attempts": 0,
    "local_exact_checks": 4,
    "selection_weights": {
        "tournament": 3.0,
        "roulette": 2.0,
        "rank": 1.0,
    },
    "operator_weights": {
        "swap_signed_with_unsigned": 2.4,
        "sampled_best_exchange": 2.2,
        "diversity_swap": 1.8,
        "critical_path_insert": 1.6,
        "block_reinsert": 1.4,
        "move_signed": 1.3,
        "reverse_segment": 1.0,
        "remove_library": 0.9,
        "swap_signed": 0.8,
        "insert_library": 0.8,
        "swap_neighbor_libraries": 0.5,
    },
}


DEFAULT_ILS_PARAMS = {
    "accept_worse_prob": 0.0320,
    "sa_final_temperature_ratio": 0.1699,
    "restart_threshold": 3,
    "perturb_strength_base": 2,
    "perturb_strength_growth": 0,
    "grasp_rcl": 0.1308,
    "restart_init_budget_ratio": 0.1491,
    "perturb_replace_bias": 0.6794,
    "restart_fresh_probability": 0.4035,
    "local_no_improve_limit": 316,
    "ls_order_weight": 0.6317,
    "ls_insert_weight": 2.3634,
    "ls_strategic_weight": 1.9964,
    "enable_initial_local_search": False,
    "alphas": ALPHA_POOLS["explore"],
}


DEFAULT_HYBRID_PARAMS = {
    "time_limit": 600.0,
    "hybrid_mode": "ga_with_parallel_ils",
    "ga_ratio": 1.0,
    "init_max_time": 120.0,
    "grasp_max_time": 5.0,
    "refine_initial_count": 3,
    "improvement_enabled": True,
    "ils_every_generations": 1,
    "ils_candidates": 6,
    "ils_time_limit": 10.0,
    "ils_reserve_time": 10.0,
    "ils_workers": 6,
    "ils_merge_policy": "best",
}


SEQUENTIAL_HYBRID_PARAMS = {
    **DEFAULT_HYBRID_PARAMS,
    "hybrid_mode": "ga_with_ils",
    "ils_candidates": 1,
    "ils_workers": 1,
}


AFTER_GENS_HYBRID_PARAMS = {
    **DEFAULT_HYBRID_PARAMS,
    "hybrid_mode": "ga_then_ils",
    "ga_ratio": 0.55,
    "ils_every_generations": 0,
    "ils_candidates": 1,
    "ils_time_limit": 1.0,
    "ils_reserve_time": 2.0,
    "ils_workers": 1,
    "ils_merge_policy": "all",
}


PARAMETER_SETS = {
    "default": {
        "description": "GA with parallel ILS using six worker candidates.",
        "hybrid": DEFAULT_HYBRID_PARAMS,
        "ga": DEFAULT_GA_PARAMS,
        "improvement": DEFAULT_ILS_PARAMS,
    },
    "ga_with_parallel_ils": {
        "description": "Run ILS workers inside GA with six parallel candidates.",
        "hybrid": DEFAULT_HYBRID_PARAMS,
        "ga": DEFAULT_GA_PARAMS,
        "improvement": DEFAULT_ILS_PARAMS,
    },
    "sequential": {
        "description": "Compatibility alias for ga_with_ils.",
        "hybrid": SEQUENTIAL_HYBRID_PARAMS,
        "ga": DEFAULT_GA_PARAMS,
        "improvement": DEFAULT_ILS_PARAMS,
    },
    "ga_with_ils": {
        "description": "Run ILS inside GA with one inline candidate.",
        "hybrid": SEQUENTIAL_HYBRID_PARAMS,
        "ga": DEFAULT_GA_PARAMS,
        "improvement": DEFAULT_ILS_PARAMS,
    },
    "ga_then_ils": {
        "description": "Run GA first, then run ILS from the best GA solution.",
        "hybrid": AFTER_GENS_HYBRID_PARAMS,
        "ga": DEFAULT_GA_PARAMS,
        "improvement": DEFAULT_ILS_PARAMS,
    },
    "after_gens": {
        "description": "Compatibility alias for ga_then_ils.",
        "hybrid": AFTER_GENS_HYBRID_PARAMS,
        "ga": DEFAULT_GA_PARAMS,
        "improvement": DEFAULT_ILS_PARAMS,
    },
}


DEFAULT_PARAMETER_SET_NAME = "default"


def get_parameter_set(name):
    return deepcopy(PARAMETER_SETS[name])


def available_parameter_sets():
    return list(PARAMETER_SETS)
