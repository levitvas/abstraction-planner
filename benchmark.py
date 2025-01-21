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
    Solve a problem using both PDB and alternation heuristics with specified tie-breaking.
    Includes timeout handling.

    Args:
        parser: Parsed SAS file
        projections: List of patterns to use
        tie_breaking: Tie-breaking method ('random' or 'average')
    """
    gamma = 0.9
    time_limit = 180  # seconds

    # PDB heuristic
    pdb_heuristics = [
        ("abs", pdb_heuristic(parser.begin_state.variables,
                              parser.end_state.variables,
                              parser, gamma, x)) for x in projections
    ]
    pdb_cost, pdb_expanded = solve_sas(parser, pdb_heuristics, tie_breaking, time_limit)

    # Alternation heuristic
    alt_heuristics = [
        ("abs", abstract_heuristic(parser.begin_state.variables,
                                   parser.end_state.variables,
                                   parser, gamma, x)) for x in projections
    ]


    alt_cost, alt_expanded = solve_sas(parser, alt_heuristics, tie_breaking, time_limit)
    return {
        'pdb': {
            'cost': pdb_cost,
            'expanded_states': pdb_expanded
        },
        'alternation': {
            'cost': alt_cost,
            'expanded_states': alt_expanded
        }
    }


def process_sas_files(output_file: str, problems_dir: str) -> None:
    """
    Process all SAS files comparing PDB vs alternation heuristics
    with different numbers of patterns (3, 6, 12) of size 3,
    and different tie-breaking methods.
    """
    results = []
    num_patterns_options = [3, 6, 12]
    pattern_size = 3
    tie_breaking_methods = ['random', 'average']

    for filename in os.listdir(problems_dir):
        if not filename.endswith('.sas'):
            continue

        sas_path = os.path.join(problems_dir, filename)
        print(f"Processing {filename}")
        try:
            problem_results = {
                'problem': filename,
                'pattern_variations': {}
            }

            # Parse problem once
            with open(sas_path) as f:
                lines = f.read().split('\n')
            parser = Parser(lines)

            # Get interesting patterns of size 3
            goal_states = [pos for pos, variable in enumerate(parser.end_state.variables)
                           if variable != -1]
            interesting_patterns = find_interesting_patterns(
                [range(0, len(parser.variables), 1)],
                parser.operators,
                goal_states,
                pattern_size
            )

            all_patterns = [list(pat) for pat in interesting_patterns
                            if len(pat) == pattern_size]

            # For each number of patterns
            for num_patterns in num_patterns_options:
                print(f"  Processing {filename} with {num_patterns} patterns")
                if len(all_patterns) >= num_patterns:
                    # Select patterns once for this number to ensure fair comparison
                    selected_patterns = random.sample(all_patterns, num_patterns)

                    # Test with different tie-breaking methods
                    tie_breaking_results = {}
                    print(selected_patterns)
                    for tie_breaking in tie_breaking_methods:
                        print(f"    Using {tie_breaking} tie-breaking")
                        result = solve_problem(parser, selected_patterns, tie_breaking)
                        tie_breaking_results[tie_breaking] = result

                    problem_results['pattern_variations'][f'num_{num_patterns}'] = {
                        'patterns': selected_patterns,
                        'tie_breaking_results': tie_breaking_results
                    }
                else:
                    print(f"Warning: Only {len(all_patterns)} patterns available for {filename}")
                    # Use all available patterns
                    tie_breaking_results = {}
                    for tie_breaking in tie_breaking_methods:
                        print(f"Using {tie_breaking} tie-breaking")
                        result = solve_problem(parser, all_patterns, tie_breaking)
                        tie_breaking_results[tie_breaking] = result

                    problem_results['pattern_variations'][f'num_{num_patterns}'] = {
                        'patterns': all_patterns,
                        'tie_breaking_results': tie_breaking_results
                    }
                    break

            results.append(problem_results)
            # Save intermediate results after each problem
            print(f"Saving intermediate results after completing {filename}")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    print("Writing final results to file")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    process_sas_files('comparison_results.json', 'problems')
