import concurrent
from functools import partial
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from heuristics.abstract_heuristic import abstract_heuristic
from heuristics.pdb_heuristic import pdb_heuristic
from planner import solve_sas

import json
import random
from typing import Dict, List

from sas_parser.parser import Parser
from utils.interesting_patterns import find_interesting_patterns, get_grown_patterns, \
    select_best_patterns_with_goal_coverage


def solve_problem(parser: Parser, projections: List[List[int]], tie_breaking: str, gamma: float, mdp_heuristics=None) -> Dict:
    time_limit = 120  # seconds

    # PDB heuristic
    pdb_heuristics = [
        ("abs", pdb_heuristic(parser.begin_state.variables,
                              parser.end_state.variables,
                              parser, gamma, x)) for x in projections
    ]
    pdb_cost, pdb_expanded = solve_sas(parser, pdb_heuristics, tie_breaking, time_limit)

    # If MDP heuristics are not provided, create them
    if mdp_heuristics is None:
        # MDP heuristic
        mdp_heuristics = [
            ("abs", abstract_heuristic(parser.begin_state.variables,
                                       parser.end_state.variables,
                                       parser, gamma, x)) for x in projections
        ]

    alt_cost, alt_expanded = solve_sas(parser, mdp_heuristics, tie_breaking, time_limit)
    return {
        'pdb': {
            'cost': pdb_cost,
            'expanded_states': pdb_expanded
        },
        'mdp': {
            'cost': alt_cost,
            'expanded_states': alt_expanded
        }
    }


def process_pattern_type(pattern_type, sorted_pairs, parser, pattern_size, pattern_amount, goal_states, gamma,
                         previous_MDP_patterns=None):
    print(f"Processing pattern type: {pattern_type} with {pattern_amount} patterns of size {pattern_size}")

    # Select the best patterns according to the sorted heuristic
    selected_patterns = [pair[0] for pair in sorted_pairs[:pattern_amount]]
    precom_heuristic = [('abs', pair[1]) for pair in sorted_pairs[:pattern_amount]]

    print(f"Solving with pattern type: {pattern_type}")
    result = solve_problem(parser, selected_patterns, "average", gamma, mdp_heuristics=precom_heuristic)

    # Return only necessary information to avoid deep dictionary merges
    return {
        'pattern_type': pattern_type,
        'pattern_size': pattern_size,
        'pattern_amount': pattern_amount,
        'selected_patterns': selected_patterns,
        'result': result,
        'values': precom_heuristic[0][1].values if pattern_type == 'sorted_patterns_init' else None
    }

def process_sas_files(output_file: str, problems_dir: str) -> None:
    results = []
    pattern_amount = [2, 3, 4, 5, 6]
    pattern_sizes = [2, 3, 4]
    # Use growing somehow, use all patterns of size 2, then solve, sort and pick best
    # For size 3, try to grow the patterns from 2, by adding one variable at a time, and doing that, then sort
    # for 4, do the same, but with 3 variables, and so on

    gamma = 0.99

    # Current methods: random, average
    tie_breaking_methods = ['random', 'average']

    domains = [f.name for f in os.scandir(problems_dir) if f.is_dir()]
    for problem_domain in domains:
        domain_result = {
            'domain': problem_domain,
            'problems': {}
        }

        joint_path = os.path.join(problems_dir, problem_domain)
        for filename in os.listdir(joint_path):
            if not filename.endswith('.sas'):
                continue

            sas_path = os.path.join(joint_path, filename)
            print(f"Processing {filename}")
            # try:
            problem_results = {
                'problem_name': filename,
                'sorted_patterns_init': {},
                'sorted_patterns_avg': {},
                'sorted_patterns_goal_avg': {},
            }

            # Parse problem once
            with open(sas_path) as f:
                lines = f.read().split('\n')
            parser = Parser(lines)

            goal_states = [pos for pos, variable in enumerate(parser.end_state.variables)
                           if variable != -1]

            previous_MDP_patterns = []
            prev_best_V_values = None
            sorted_size_two_patterns = []
            for pattern_size in pattern_sizes:
                print(f"Starting pattern size {pattern_size}")
                # Get interesting patterns of size pattern_size
                interesting_patterns = find_interesting_patterns(
                    [range(0, len(parser.variables), 1)],
                    parser.operators,
                    goal_states,
                    pattern_size
                )

                all_patterns = [list(pat) for pat in interesting_patterns
                                if len(pat) == pattern_size]
                # print(f"Found {len(all_patterns)} interesting patterns of size {pattern_size}")

                # If we are at size 2, then we can just use the patterns
                if pattern_size > 2:
                    all_patterns = get_grown_patterns(previous_MDP_patterns, pattern_size, pattern_amount, sorted_size_two_patterns)

                print("Starting pattern selection(MDP)")
                pattern_heuristic_pairs = [(pattern, abstract_heuristic(parser.begin_state.variables,
                                                                        parser.end_state.variables,
                                                                        parser, gamma, pattern, prev_V=prev_best_V_values)) for pattern in all_patterns]
                print("Finished pattern selection(MDP)")

                # Sort by initial state with avg tie breaking
                init_state_sort_ptrns = sorted(pattern_heuristic_pairs, key=lambda x: (x[1](parser.begin_state.variables), x[1].get_average_cost()), reverse=True)
                print(len(init_state_sort_ptrns))

                # Sort by average of all values'
                avg_sort_ptrns = sorted(pattern_heuristic_pairs, key=lambda x: x[1].get_average_cost(), reverse=True)

                # Sort by goal coverage and avg MDP
                goal_covered_ptrns = select_best_patterns_with_goal_coverage(avg_sort_ptrns, pattern_amount[0], goal_states)

                if pattern_size == 2:
                    sorted_size_two_patterns = [pair[0] for pair in init_state_sort_ptrns]

                # Process all pattern types in parallel
                pattern_types = {
                    'sorted_patterns_init': init_state_sort_ptrns,
                    'sorted_patterns_avg': avg_sort_ptrns,
                    'sorted_patterns_goal_avg': goal_covered_ptrns
                }

                # Two types of pattern selections
                # 0 is random, 1 is sorted by MDP value of init state (without 0)
                # for select_type in range(3):
                    # string and pairs
                    # def get_sorted_pattern_heuristic_pairs():
                    #     if select_type == 0:
                    #         return 'sorted_patterns_init', init_state_sort_ptrns
                    #     elif select_type == 1:
                    #         return 'sorted_patterns_avg', avg_sort_ptrns
                    #     elif select_type == 2:
                    #         return 'sorted_patterns_goal_avg', goal_covered_ptrns
                    #     return None

                    # select_type_name, sorted_pattern_heuristic_pairs = get_sorted_pattern_heuristic_pairs()

                    # Initialize the dictionary for the current pattern size
                    # if f'pattern_size_{pattern_size}' not in problem_results[select_type_name]:
                    #     problem_results[select_type_name][f'pattern_size_{pattern_size}'] = {}
                    # For each number of patterns
                for idx, num_patterns in enumerate(pattern_amount):
                    print(f"  Processing {filename} with {num_patterns} patterns of size {pattern_size}")
                    if len(all_patterns) >= num_patterns:

                        with concurrent.futures.ProcessPoolExecutor() as executor:
                            futures = {}

                            for pattern_type, sorted_pairs in pattern_types.items():
                                # Submit with specific keyword arguments to avoid conflicts
                                future = executor.submit(
                                    process_pattern_type,
                                    pattern_type=pattern_type,
                                    sorted_pairs=sorted_pairs,
                                    parser=parser,
                                    pattern_size=pattern_size,
                                    pattern_amount=num_patterns,
                                    goal_states=goal_states,
                                    gamma=gamma,
                                    previous_MDP_patterns=previous_MDP_patterns
                                )
                                futures[future] = pattern_type

                            for future in concurrent.futures.as_completed(futures):
                                data = future.result()
                                pattern_type = data['pattern_type']

                                # Create nested dictionaries if they don't exist
                                if f'pattern_size_{data["pattern_size"]}' not in problem_results[pattern_type]:
                                    problem_results[pattern_type][f'pattern_size_{data["pattern_size"]}'] = {}

                                # Add this specific pattern amount result without overwriting others
                                problem_results[pattern_type][f'pattern_size_{data["pattern_size"]}'][
                                    f'pattern_amount_{data["pattern_amount"]}'] = {
                                    'patterns': data['selected_patterns'],
                                    'tie_break_type': 'average',
                                    'growing_method': 'conjunction',
                                    'tie_breaking_result': data['result']
                                }

                                # If this is the init_state pattern type, update the previous patterns
                                if pattern_type == 'sorted_patterns_init':
                                    if data['values'] is not None:
                                        previous_MDP_patterns = data['selected_patterns']
                                        prev_best_V_values = data['values']
                    else:
                        print(f"Warning: Only {len(all_patterns)} patterns available for {filename}")
                        # # Use all available patterns
                        # tie_breaking_results = {}
                        # for tie_breaking in tie_breaking_methods:
                        #     print(f"Using {tie_breaking} tie-breaking")
                        #     result = solve_problem(parser, all_patterns, tie_breaking)
                        #     tie_breaking_results[tie_breaking] = result
                        #
                        # problem_results['pattern_variations'][f'pattern_size_{pattern_size}'][f'pattern_amount_{num_patterns}'] = {
                        #     'patterns': all_patterns,
                        #     'tie_breaking_results': tie_breaking_results
                        # }
                        # break
                        pass

                    domain_result['problems'][filename] = problem_results
                    results.append(domain_result)
                    # Save intermediate results after each problem
                    print(f"Saving intermediate results after completing {filename}")
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    results.pop()
                    print(f"Intermediate results saved successfully")
                print("Exited the first loop")

            print("Exited the second loop")

            # except Exception as e:
            #     print(f"Error processing {filename}: {str(e)}")
            #     continue
        print(f"Results for domain {problem_domain} completed")
        results.append(domain_result)

    print("Writing final results to file")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    process_sas_files('comparison_FINAL.json', '../problems')
