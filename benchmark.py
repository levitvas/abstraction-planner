from heuristics.abstract_heuristic import abstract_heuristic
from heuristics.pdb_heuristic import pdb_heuristic
from planner import create_plan, solve_sas

import os
import json
import random
from typing import Dict, List, Tuple

from sas_parser.parser import Parser
from utils.interesting_patterns import find_interesting_patterns


# process_sas_files('prev_results.json', '/path/to/problems')
def solve_problem(parser: Parser, projections: List[List[int]], tie_breaking: str) -> Dict:
    """
    Solve a problem using both PDB and MDP` heuristics with specified tie-breaking.
    Includes timeout handling.

    Args:
        parser: Parsed SAS file
        projections: List of patterns to use
        tie_breaking: Tie-breaking method ('random' or 'average')
    """
    gamma = 0.95
    time_limit = 30  # seconds

    # PDB heuristic
    pdb_heuristics = [
        ("abs", pdb_heuristic(parser.begin_state.variables,
                              parser.end_state.variables,
                              parser, gamma, x)) for x in projections
    ]
    pdb_cost, pdb_expanded = solve_sas(parser, pdb_heuristics, tie_breaking, time_limit)

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


def process_sas_files(output_file: str, problems_dir: str) -> None:
    """
    Process all SAS files comparing PDB vs MDP heuristics
    with different numbers of patterns (3, 6, 12) of size 3,
    and different tie-breaking methods.
    """
    results = []
    pattern_amount = [2, 3, 4, 5, 6]
    pattern_sizes = [2, 3, 4]

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
            try:
                problem_results = {
                    'problem_name': filename,
                    'random_patterns': {},
                    'sorted_patterns': {},
                }

                # Parse problem once
                with open(sas_path) as f:
                    lines = f.read().split('\n')
                parser = Parser(lines)

                goal_states = [pos for pos, variable in enumerate(parser.end_state.variables)
                               if variable != -1]

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

                    print(f"Found {len(all_patterns)} interesting patterns of size {pattern_size}")
                    print("Starting pattern selection(MDP)")
                    # Pick the first num_patterns patterns according to best reward from the solved MDP
                    # So first create and solve a mdp and solve for each interesting pattern
                    gamma = 0.95
                    pattern_heuristic_pairs = [(pattern, abstract_heuristic(parser.begin_state.variables,
                                                                            parser.end_state.variables,
                                                                            parser, gamma, pattern)(parser.begin_state.variables)) for pattern in all_patterns]

                    # Ignores states that are 0 directly, since they potentially can't solve much
                    # sorted_pattern_heuristic_pairs = sorted(pattern_heuristic_pairs, key=lambda x: float('inf') if x[1] == 0 else x[1])
                    sorted_pattern_heuristic_pairs = sorted(pattern_heuristic_pairs, key=lambda x: x[1])

                    # Two types of pattern selections
                    # 0 is random, 1 is sorted by MDP value of init state (without 0)
                    for select_type in range(2):
                        select_type_name = 'random_patterns' if select_type == 0 else 'sorted_patterns'
                        # Initialize the dictionary for the current pattern size
                        if f'pattern_size_{pattern_size}' not in problem_results[select_type_name]:
                            problem_results[select_type_name][f'pattern_size_{pattern_size}'] = {}
                        # For each number of patterns
                        for idx, num_patterns in enumerate(pattern_amount):
                            print(f"  Processing {filename} with {num_patterns} patterns of size {pattern_size}")
                            if len(all_patterns) >= num_patterns:

                                if select_type == 0:
                                    print(f"  Processing {filename} with random pattern selection")
                                    # Select patterns randomly
                                    selected_patterns = random.sample(all_patterns, num_patterns)
                                else:
                                    print(f"  Processing {filename} with sorted by MDP pattern selection")
                                    # Select the best patterns according to the sorted heuristic
                                    selected_patterns = [pair[0] for pair in sorted_pattern_heuristic_pairs][:num_patterns]

                                # Test with different tie-breaking methods
                                tie_breaking_results = {}
                                for tie_breaking in tie_breaking_methods:
                                    print(f"    Using {tie_breaking} tie-breaking")
                                    result = solve_problem(parser, selected_patterns, tie_breaking)
                                    tie_breaking_results[tie_breaking] = result

                                problem_results[select_type_name][f'pattern_size_{pattern_size}'][f'pattern_amount_{num_patterns}'] = {
                                    'patterns': selected_patterns,
                                    'tie_breaking_results': tie_breaking_results
                                }
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

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        print(f"Results for domain {problem_domain} completed")
        results.append(domain_result)

    print("Writing final results to file")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    process_sas_files('comparison_newest.json', 'problems')
