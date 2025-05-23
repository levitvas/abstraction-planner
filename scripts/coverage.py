import os
import sys
import time
import json
import random
import itertools  # For combinations when growing patterns

# Ensure the parent directory is in sys.path to find other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sas_parser.parser import Parser
from heuristics.hff_heuristic import hff
from heuristics.abstract_heuristic import abstract_heuristic
from utils.interesting_patterns import find_interesting_patterns, get_grown_patterns, \
    select_best_patterns_with_goal_coverage_optimized
from a_star_search import gbfs

# --- Configuration ---
PROBLEMS_DIR = '../problems'
OUTPUT_FILE = 'v2_20_init.json'  # New output file name
TIME_LIMIT_SECONDS = 200.0

# MDP Configuration
MDP_TARGET_PATTERN_SIZE = 2
MDP_INITIAL_PATTERN_SIZE_FOR_GROWING = 2  # Grow from size 2 to 3
MDP_NUM_PATTERNS_TO_SELECT = 20
MDP_GAMMA = 0.99
MDP_TIE_BREAKING_SEARCH = "average"  # For GBFS alternation


# --- Helper Function to call GBFS (adapted from planner.py's solve_sas) ---
def run_gbfs_for_coverage(parser: Parser,
                          heuristics_list_with_names: list,
                          tie_breaking: str,
                          gbfs_time_limit: float):
    if gbfs_time_limit <= 0:  # No time left
        return -1, -1, 0.0  # cost, expanded, time_taken_by_gbfs

    gbfs_start_time = time.time()
    all_facts = []
    for var_idx, var_obj in enumerate(parser.variables):
        domain_size = len(var_obj.variables)
        for domain_val_idx in range(domain_size):
            all_facts.append((var_idx, domain_val_idx))

    result_tuple, expanded_states = gbfs(
        facts=all_facts,
        init_state=parser.begin_state.variables,
        actions=parser.operators,
        goal_state=parser.end_variables,
        heuristics=heuristics_list_with_names,
        var_len=len(parser.variables),
        tie_breaking=tie_breaking,
        time_limit=gbfs_time_limit
    )
    gbfs_end_time = time.time()
    gbfs_time_taken = gbfs_end_time - gbfs_start_time

    if result_tuple is None or result_tuple == (-1, None):
        return -1, -1, gbfs_time_taken  # Use -1 for expanded if no solution/timeout

    cost = result_tuple[0]
    return cost, expanded_states, gbfs_time_taken


# --- Main Coverage Script ---
def main():
    all_results_data = {}  # Store results per problem: { "domain/problem.sas": { ... } }
    summary = {
        "FF": {"solved": 0, "total": 0, "time_sum": 0.0, "cost_sum": 0, "expanded_sum": 0},
        "MDP_Grow": {"solved": 0, "total": 0, "time_sum": 0.0, "cost_sum": 0, "expanded_sum": 0, "vi_time_sum": 0.0}
    }

    if not os.path.isdir(PROBLEMS_DIR):
        print(f"Error: Problems directory not found: {PROBLEMS_DIR}")
        return

    for domain_name in sorted(os.listdir(PROBLEMS_DIR)):
        domain_path = os.path.join(PROBLEMS_DIR, domain_name)
        if not os.path.isdir(domain_path):
            continue

        print(f"\n--- Domain: {domain_name} ---")
        for problem_filename in sorted(os.listdir(domain_path)):
            if not problem_filename.endswith('.sas'):
                continue

            problem_key = f"{domain_name}/{problem_filename}"
            all_results_data[problem_key] = {"problem_info": {"domain": domain_name, "problem": problem_filename}}
            print(f"  Processing: {problem_key}")

            try:
                with open(os.path.join(domain_path, problem_filename)) as f:
                    lines = f.read().split('\n')
                parser = Parser(lines)
            except Exception as e:
                print(f"    Error parsing {problem_filename}: {e}")
                all_results_data[problem_key]["FF"] = {"solved": False, "error": str(e)}
                all_results_data[problem_key]["MDP_Grow"] = {"solved": False, "error": str(e)}
                continue

            # --- 1. Run with FF Heuristic ---
            summary["FF"]["total"] += 1
            ff_heuristic_entry = [("hff", hff)]

            # ff_cost, ff_expanded, ff_search_time = run_gbfs_for_coverage(
            #     parser, ff_heuristic_entry, MDP_TIE_BREAKING_SEARCH, TIME_LIMIT_SECONDS
            # )
            ff_cost, ff_expanded, ff_search_time = -1, -1, -1

            ff_solved = (ff_cost != -1)
            if ff_solved:
                summary["FF"]["solved"] += 1
                summary["FF"]["time_sum"] += ff_search_time
                summary["FF"]["cost_sum"] += ff_cost
                summary["FF"]["expanded_sum"] += ff_expanded

            all_results_data[problem_key]["FF"] = {
                "solved": ff_solved, "cost": ff_cost, "expanded": ff_expanded,
                "time_search_seconds": round(ff_search_time, 3)
            }
            print(f"    FF: Solved={ff_solved}, Cost={ff_cost}, Exp={ff_expanded}, SearchTime={ff_search_time:.2f}s")

            # --- 2. Run with MDP Heuristic (Pattern Growing & Selection) ---
            summary["MDP_Grow"]["total"] += 1
            time_mdp_overall_start = time.time()

            mdp_cost, mdp_expanded, mdp_vi_time, mdp_search_time = -1, -1, 0.0, 0.0
            selected_mdp_patterns_for_run = []
            mdp_error = None

            try:
                goal_vars_indices = [pos for pos, val in enumerate(parser.end_state.variables) if val != -1]
                all_vars_indices = [range(0, len(parser.variables), 1)]

                # Step 2a: Find initial patterns (size MDP_INITIAL_PATTERN_SIZE_FOR_GROWING)
                time_pat_sel_start = time.time()
                initial_patterns_tuples = find_interesting_patterns(
                    all_vars_indices, parser.operators, goal_vars_indices, MDP_INITIAL_PATTERN_SIZE_FOR_GROWING
                )
                # Convert set of tuples to list of lists
                initial_patterns_list = [list(p) for p in initial_patterns_tuples
                                         if len(p) == MDP_INITIAL_PATTERN_SIZE_FOR_GROWING]

                if not initial_patterns_list:
                    print(f"    MDP_Grow: No initial patterns of size {MDP_INITIAL_PATTERN_SIZE_FOR_GROWING} found.")
                    raise ValueError(f"No base patterns of size {MDP_INITIAL_PATTERN_SIZE_FOR_GROWING}")

                # Sort the initial patterns by average cost for use in get_grown_patterns or direct selection
                pattern_heuristic_pairs_initial = []
                for pattern in initial_patterns_list:
                    h_instance = abstract_heuristic(
                        parser.begin_state.variables, parser.end_state.variables,
                        parser, MDP_GAMMA, pattern
                    )
                    pattern_heuristic_pairs_initial.append((pattern, h_instance))

                sorted_pattern_heuristics_initial = sorted(
                    pattern_heuristic_pairs_initial,
                    key=lambda x: ( x[1](parser.begin_state.variables),
                        -x[1].get_average_cost() if hasattr(x[1], 'get_average_cost') else 0),
                    reverse=True
                )
                # Sort by goal coverage and avg MDP
                # sorted_pattern_heuristics_initial = select_best_patterns_with_goal_coverage_optimized(sorted_pattern_heuristics_initial, MDP_NUM_PATTERNS_TO_SELECT,goal_vars_indices)


                sorted_initial_patterns = [pair[0] for pair in sorted_pattern_heuristics_initial]

                # Check if growing is needed based on target size vs initial size
                if MDP_TARGET_PATTERN_SIZE > MDP_INITIAL_PATTERN_SIZE_FOR_GROWING:
                    print(
                        f"    MDP_Grow: Growing patterns from size {MDP_INITIAL_PATTERN_SIZE_FOR_GROWING} to {MDP_TARGET_PATTERN_SIZE}")
                    # Select the top patterns to use as a basis for growing
                    previous_MDP_patterns = sorted_initial_patterns[:MDP_NUM_PATTERNS_TO_SELECT]

                    # Grow patterns using the original get_grown_patterns function
                    candidate_target_patterns = get_grown_patterns(
                        previous_MDP_patterns,
                        MDP_TARGET_PATTERN_SIZE,
                        [MDP_NUM_PATTERNS_TO_SELECT],
                        sorted_initial_patterns
                    )

                    if not candidate_target_patterns:
                        print(
                            f"    MDP_Grow: No patterns of size {MDP_TARGET_PATTERN_SIZE} after growing. Skipping MDP.")
                        raise ValueError(f"No patterns of size {MDP_TARGET_PATTERN_SIZE} after growing.")
                else:
                    print(f"    MDP_Grow: No growing needed, already at target size {MDP_TARGET_PATTERN_SIZE}")
                    # No growing needed, use the initial patterns directly
                    candidate_target_patterns = sorted_initial_patterns

                time_pat_sel_end = time.time()
                pattern_selection_time = time_pat_sel_end - time_pat_sel_start

                # Step 2c: Instantiate heuristics for the candidate patterns and sort them
                time_vi_start = time.time()
                pattern_heuristic_pairs = []
                for pattern in candidate_target_patterns:
                    if time.time() - time_mdp_overall_start > TIME_LIMIT_SECONDS:
                        print("    MDP_Grow: Time limit reached during VI for pattern candidates.")
                        raise TimeoutError("Timeout during VI for pattern candidates")

                    h_instance = abstract_heuristic(
                        parser.begin_state.variables, parser.end_state.variables,
                        parser, MDP_GAMMA, pattern
                    )
                    pattern_heuristic_pairs.append((pattern, h_instance))

                # Sort by average_cost, higher values first
                sorted_pattern_heuristics = sorted(
                    pattern_heuristic_pairs,
                    key=lambda x: (x[1](parser.begin_state.variables),
                        -x[1].get_average_cost() if hasattr(x[1], 'get_average_cost') else 0),
                    reverse=True
                )
                # sorted_pattern_heuristics = select_best_patterns_with_goal_coverage_optimized(sorted_pattern_heuristics, MDP_NUM_PATTERNS_TO_SELECT, goal_vars_indices)

                time_vi_end = time.time()
                mdp_vi_time = (time_vi_end - time_vi_start) + pattern_selection_time  # Add pattern selection time here

                # Step 2d: Select top N patterns
                selected_mdp_heuristics_with_names = []
                if sorted_pattern_heuristics:
                    top_n_pairs = sorted_pattern_heuristics[:MDP_NUM_PATTERNS_TO_SELECT]
                    selected_mdp_patterns_for_run = [pair[0] for pair in top_n_pairs]
                    selected_mdp_heuristics_with_names = [("abs", pair[1]) for pair in top_n_pairs]

                if not selected_mdp_heuristics_with_names:
                    print(f"    MDP_Grow: No MDP heuristics selected after sorting. Skipping.")
                    raise ValueError("No MDP heuristics selected after sorting")

                # Step 2e: Run GBFS with selected MDP heuristics
                remaining_time_for_gbfs = TIME_LIMIT_SECONDS - (time.time() - time_mdp_overall_start)

                if remaining_time_for_gbfs > 0:
                    print(f"    MDP_Grow: {len(selected_mdp_heuristics_with_names)} Running GBFS with selected MDP heuristics...")
                    mdp_cost, mdp_expanded, mdp_search_time = run_gbfs_for_coverage(
                        parser, selected_mdp_heuristics_with_names, MDP_TIE_BREAKING_SEARCH, remaining_time_for_gbfs
                    )
                else:
                    print(f"    MDP_Grow: Not enough time for GBFS (VI/Setup took {mdp_vi_time:.2f}s).")

            except TimeoutError as te:
                mdp_error = str(te)
                print(f"    MDP_Grow: Timeout - {mdp_error}")
            except Exception as e:
                mdp_error = str(e)
                print(f"    Error during MDP_Grow processing for {problem_filename}: {e}")

            time_mdp_overall_end = time.time()
            mdp_total_time = time_mdp_overall_end - time_mdp_overall_start

            mdp_solved = (mdp_cost != -1)
            if mdp_solved:
                summary["MDP_Grow"]["solved"] += 1
                summary["MDP_Grow"]["time_sum"] += mdp_total_time
                summary["MDP_Grow"]["cost_sum"] += mdp_cost
                summary["MDP_Grow"]["expanded_sum"] += mdp_expanded
                summary["MDP_Grow"]["vi_time_sum"] += mdp_vi_time

            all_results_data[problem_key]["MDP_Grow"] = {
                "solved": mdp_solved, "cost": mdp_cost, "expanded": mdp_expanded,
                "time_total_seconds": round(mdp_total_time, 3),
                "time_vi_pattern_seconds": round(mdp_vi_time, 3),
                "time_search_seconds": round(mdp_search_time, 3),
                "patterns_used": selected_mdp_patterns_for_run,
                "error": mdp_error
            }
            print(
                f"    MDP_Grow: Solved={mdp_solved}, Cost={mdp_cost}, Exp={mdp_expanded}, TotalTime={mdp_total_time:.2f}s (VI/Pat: {mdp_vi_time:.2f}s, Search: {mdp_search_time:.2f}s)")

            # Save intermediate results after each problem
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(all_results_data, f, indent=2)
            print(f"    Intermediate results for {problem_key} saved.")

    # --- Print Summary ---
    print("\n\n--- Coverage Summary ---")
    for approach in ["FF", "MDP_Grow"]:
        s = summary[approach]
        if s["total"] > 0:
            avg_time_solved_total = s["time_sum"] / s["solved"] if s["solved"] > 0 else 0
            avg_cost_solved = s["cost_sum"] / s["solved"] if s["solved"] > 0 else 0
            avg_exp_solved = s["expanded_sum"] / s["solved"] if s["solved"] > 0 else 0
            print(f"{approach}:")
            print(f"  Solved: {s['solved']} / {s['total']} problems")
            if s['solved'] > 0:
                print(f"  Avg Total Time (solved): {avg_time_solved_total:.2f}s")
                if approach == "MDP_Grow":
                    avg_vi_time_solved = s["vi_time_sum"] / s["solved"] if s["solved"] > 0 else 0
                    print(f"  Avg VI/Pattern Time (solved): {avg_vi_time_solved:.2f}s")
                print(f"  Avg Cost (solved): {avg_cost_solved:.2f}")
                print(f"  Avg Expanded (solved): {avg_exp_solved:.2f}")
        else:
            print(f"{approach}: No problems attempted.")

    print(f"\nFinal detailed results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()