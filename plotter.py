import os

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import json


def create_domain_layout(domain_name, problems, figsize=(15, 12)):
    """
    Automatically create appropriate subplot layout based on number of problems
    - 2 problems: side by side (1×2)
    - 3 problems: triangle layout
    - 4 problems: square layout (2×2)
    - Others: automatic grid

    Args:
        domain_name: Name of the domain for the title
        problems: List of problem filenames
        figsize: Figure size tuple (width, height)

    Returns:
        fig: The figure object
        axs: Array of axes objects (flattened for consistent indexing)
    """
    num_problems = len(problems)
    problem_names = [p.split('/')[-1] for p in problems]  # Extract filenames

    if num_problems == 2:
        # Two problems - side by side
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        axs = axs.flatten()

    elif num_problems == 3:
        # Three problems - triangle layout
        fig = plt.figure(figsize=figsize)

        # Create triangle arrangement (2 on top, 1 centered below)
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)  # Bottom spans both columns
        axs = [ax1, ax2, ax3]

    elif num_problems == 4:
        # Four problems - square layout (2×2)
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.flatten()

    # Set domain title and adjust layout
    domain_title = domain_name.split('/')[-1] if '/' in domain_name else domain_name
    fig.suptitle(f"{domain_title}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for title

    # Set individual plot titles
    for i, problem in enumerate(problem_names):
        axs[i].set_title(problem)

    return fig, axs

def plot_results(results_file: str, output_dir: str):
    """
    Create plots from the results JSON file.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    tie_breaking_methods = ['random']
    pattern_sizes = ['0_num_1', "1_num_1", "2_num_1", "3_num_1", "4_num_1", "5_num_1", "6_num_1"]
    print(f"Loaded {len(results)} results")

    # Create plots for each domain
    for domain in results:
        domain_name = domain['domain']

        domain_results = {}

        # For each problem in the domain
        for problem in domain['problems']:
            problem = domain['problems'][problem]
            problem_name = problem['problem_name']

            domain_results[problem_name] = {}
            domain_results[problem_name]['mdp'] = {}
            domain_results[problem_name]['mdp']['cost'] = []
            domain_results[problem_name]['mdp']['expanded'] = []

            domain_results[problem_name]['pdb'] = {}
            domain_results[problem_name]['pdb']['cost'] = []
            domain_results[problem_name]['pdb']['expanded'] = []

            domain_results[problem_name]['tie_breaking_cost'] = {}
            domain_results[problem_name]['tie_breaking_cost']['random'] = []
            domain_results[problem_name]['tie_breaking_cost']['average'] = []

            domain_results[problem_name]['pattern_size_cost'] = {}

            domain_results[problem_name]['pattern_select_type'] = {}

            print(f"Creating plots for {domain_name}/{problem_name}")

            pattern_types = ['random_patterns', 'sorted_patterns']
            for pattern_type in pattern_types:
                domain_results[problem_name]['pattern_select_type'][pattern_type] = []
                for pattern_size in problem[pattern_type]:
                    if pattern_size not in domain_results[problem_name]['pattern_size_cost']:
                        domain_results[problem_name]['pattern_size_cost'][pattern_size] = []

                    for pattern_amount in problem[pattern_type][pattern_size]:

                        for tie_breaking_type in problem[pattern_type][pattern_size][pattern_amount]['tie_breaking_results']:

                            short_name = problem[pattern_type][pattern_size][pattern_amount]['tie_breaking_results'][tie_breaking_type]
                            for heuristic_type in short_name:
                                resulting_cost = short_name[heuristic_type]['cost'] if short_name[heuristic_type]['cost'] != -1 else 0

                                # Tie-breaking cost
                                domain_results[problem_name]['tie_breaking_cost'][tie_breaking_type].append(
                                    resulting_cost)

                                if tie_breaking_type == 'average':
                                    # Either MDP or PDB
                                    # I compute without random, since there is no point, because its always worse
                                    domain_results[problem_name][heuristic_type]['cost'].append(resulting_cost)
                                    domain_results[problem_name][heuristic_type]['expanded'].append(
                                        short_name[heuristic_type]['expanded_states'])

                                    # Pattern type costs
                                    domain_results[problem_name]['pattern_size_cost'][pattern_size].append(resulting_cost)

                                    # Pattern selection type
                                    domain_results[problem_name]['pattern_select_type'][pattern_type].append(resulting_cost)

        # Create plots for the domain
        os.makedirs(f'{output_dir}/{domain_name}', exist_ok=True)
        # Start with bar plot between average tie breaking cost and expanded states as well as solved problems
        fig, axs = create_domain_layout(domain_name, domain_results.keys())
        for i, problem_name in enumerate(domain_results):
            problem = domain_results[problem_name]
            ax = axs[i]

            # Create bar plot for average tie breaking cost
            ax.bar(['Random', 'Average'], [np.sum(problem['tie_breaking_cost']['random'])/np.count_nonzero(problem['tie_breaking_cost']['random']),
                                           np.sum(problem['tie_breaking_cost']['average'])/np.count_nonzero(problem['tie_breaking_cost']['average'])], color=['skyblue', 'lightcoral'])
            ax.set_title(f"{problem_name} - Average Tie Breaking Cost")
            ax.set_ylabel('Average Cost')
            ax.set_xlabel('Tie Breaking Method')

        plt.savefig(f'{output_dir}/{domain_name}/average_tie_breaking_cost.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Now for the success rate
        fig, axs = create_domain_layout(domain_name, domain_results.keys())
        for i, problem_name in enumerate(domain_results):
            problem = domain_results[problem_name]
            ax = axs[i]

            # Create bar plot for number of solved problems for random vs average tie-breaking
            # the cost is 0 when the problem is not solved
            ax.bar(['Random', 'Average'], [np.count_nonzero(problem['tie_breaking_cost']['random']),
                                           np.count_nonzero(problem['tie_breaking_cost']['average'])], color=['skyblue', 'lightcoral'])
            ax.set_title(f"{problem_name} - Solved Problems")
            ax.set_ylabel('Number of Solved Problems')
            ax.set_xlabel('Tie Breaking Method')

        plt.savefig(f'{output_dir}/{domain_name}/solved_problems.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Now scatter plots for MDP vs PDB cost
        fig, axs = create_domain_layout(domain_name, domain_results.keys())
        for i, problem_name in enumerate(domain_results):
            problem = domain_results[problem_name]
            ax = axs[i]

            ax.scatter(problem['mdp']['cost'], problem['pdb']['cost'], alpha=0.8, color="green")
            ax.plot([0, max(problem['mdp']['cost'])], [0, max(problem['mdp']['cost'])], 'r--', alpha=0.5)
            ax.set_title(f"{problem_name} - MDP vs PDB Cost")
            ax.set_xlabel('MDP Cost')
            ax.set_ylabel('PDB Cost')
        plt.savefig(f'{output_dir}/{domain_name}/mdp_vs_pdb_cost.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Now scatter plots for MDP vs PDB expanded states
        fig, axs = create_domain_layout(domain_name, domain_results.keys())
        for i, problem_name in enumerate(domain_results):
            problem = domain_results[problem_name]
            ax = axs[i]

            ax.scatter(problem['mdp']['expanded'], problem['pdb']['expanded'], alpha=0.8, color="green")
            ax.plot([0, max(problem['mdp']['expanded'])], [0, max(problem['mdp']['expanded'])], 'r--', alpha=0.5)
            ax.set_title(f"{problem_name} - MDP vs PDB Expanded States")
            ax.set_xlabel('MDP Expanded States')
            ax.set_ylabel('PDB Expanded States')
            ax.set_xscale('log')
            ax.set_yscale('log')
        plt.savefig(f'{output_dir}/{domain_name}/mdp_vs_pdb_expanded.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, axs = create_domain_layout(domain_name, domain_results.keys())
        for i, problem_name in enumerate(domain_results):
            problem = domain_results[problem_name]
            ax = axs[i]

            for h_type in ['mdp', 'pdb']:
                ax.bar(h_type, np.count_nonzero(problem[h_type]['cost']))
            ax.set_title(f"{problem_name} - Solved Problems")
            ax.set_ylabel('Number of Solved Problems')
            ax.set_xlabel('Pattern Size')
        plt.savefig(f'{output_dir}/{domain_name}/solved_problems_mdp_vs_pdb.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Now bar plot between average pattern size cost
        fig, axs = create_domain_layout(domain_name, domain_results.keys())
        for i, problem_name in enumerate(domain_results):
            problem = domain_results[problem_name]
            ax = axs[i]

            for pattern_size in problem['pattern_size_cost']:
                ax.bar(pattern_size, np.sum(problem['pattern_size_cost'][pattern_size])/np.count_nonzero(problem['pattern_size_cost'][pattern_size]))
            ax.set_title(f"{problem_name} - Average Pattern Size Cost")
            ax.set_ylabel('Average Cost')
            ax.set_xlabel('Pattern Size')
        plt.savefig(f'{output_dir}/{domain_name}/average_pattern_size_cost.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Now the same, but with the amount of solved problems
        fig, axs = create_domain_layout(domain_name, domain_results.keys())
        for i, problem_name in enumerate(domain_results):
            problem = domain_results[problem_name]
            ax = axs[i]

            for pattern_size in problem['pattern_size_cost']:
                ax.bar(pattern_size, np.count_nonzero(problem['pattern_size_cost'][pattern_size]))
            ax.set_title(f"{problem_name} - Solved Problems")
            ax.set_ylabel('Number of Solved Problems')
            ax.set_xlabel('Pattern Size')
        plt.savefig(f'{output_dir}/{domain_name}/solved_problems_pattern_size.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Now pattern select type, the average. Which got more
        fig, axs = create_domain_layout(domain_name, domain_results.keys())
        for i, problem_name in enumerate(domain_results):
            problem = domain_results[problem_name]
            ax = axs[i]

            ax.bar(['Random', 'Sorted'], [np.sum(problem['pattern_select_type']['random_patterns'])/np.count_nonzero(problem['pattern_select_type']['random_patterns']),
                                           np.sum(problem['pattern_select_type']['sorted_patterns'])/np.count_nonzero(problem['pattern_select_type']['sorted_patterns'])], color=['skyblue', 'lightcoral'])
            ax.set_title(f"{problem_name} - Average Pattern Selection Cost")
            ax.set_ylabel('Average Cost')
            ax.set_xlabel('Pattern Selection Type')
        plt.savefig(f'{output_dir}/{domain_name}/average_pattern_selection_cost.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Now which solved more problems
        fig, axs = create_domain_layout(domain_name, domain_results.keys())
        for i, problem_name in enumerate(domain_results):
            problem = domain_results[problem_name]
            ax = axs[i]

            ax.bar(['Random', 'Sorted'], [np.count_nonzero(problem['pattern_select_type']['random_patterns']),
                                           np.count_nonzero(problem['pattern_select_type']['sorted_patterns'])], color=['skyblue', 'lightcoral'])
            ax.set_title(f"{problem_name} - Solved Problems")
            ax.set_ylabel('Number of Solved Problems')
            ax.set_xlabel('Pattern Selection Type')
        plt.savefig(f'{output_dir}/{domain_name}/solved_problems_pattern_selection.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    os.makedirs('plots_new', exist_ok=True)
    plot_results('comparison_results_newest_diff.json', 'plots_new')
