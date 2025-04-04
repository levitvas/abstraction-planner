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
    - 6 problems: 2×3 grid
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

    elif num_problems == 6:
        # Six problems - 2×3 grid
        fig, axs = plt.subplots(2, 3, figsize=figsize)
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

    # tie_breaking_methods = ['random']
    # pattern_sizes = ['0_num_1', "1_num_1", "2_num_1", "3_num_1", "4_num_1", "5_num_1", "6_num_1"]

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

            # Graph - All configurations combined
            domain_results[problem_name]['mdp'] = {'cost': [], 'expanded': []}
            domain_results[problem_name]['pdb'] = {'cost': [], 'expanded': []}

            # Graph - plots for the gamma values, cost and expanded states
            domain_results[problem_name]['gamma'] = {}

            # Graph - plots for the pattern size cost, grouped by pattern amount
            domain_results[problem_name]['pattern_size_cost'] = {}

            # Graph - plots for the pattern selection type, random or sorted
            domain_results[problem_name]['pattern_select_type'] = {}

            print(f"Creating plots for {domain_name}/{problem_name}")

            pattern_types = ['random_patterns', 'sorted_patterns']
            # problems/problem_name
            for pattern_type in pattern_types:
                domain_results[problem_name]['pattern_select_type'][pattern_type] = []
                # problems/problem_name/gamma
                for gamma in problem[pattern_type]:
                    if gamma not in domain_results[problem_name]['gamma']:
                        domain_results[problem_name]['gamma'][gamma] = {}
                        domain_results[problem_name]['gamma'][gamma]['mdp'] = {'cost': [], 'expanded': []}
                        domain_results[problem_name]['gamma'][gamma]['pdb'] = {'cost': [], 'expanded': []}

                    # problems/problem_name/
                    for pattern_size in problem[pattern_type][gamma]:
                        if pattern_size not in domain_results[problem_name]['pattern_size_cost']:
                            domain_results[problem_name]['pattern_size_cost'][pattern_size] = {'cost': [], 'mdp': [0], 'pdb': [0], 'expand': [0], 'mdp_expand': [0], 'pdb_expand': [0]}
                        # problems/problem_name/gamma/pattern_type/pattern_size
                        for pattern_amount in problem[pattern_type][gamma][pattern_size]:

                            short_name = problem[pattern_type][gamma][pattern_size][pattern_amount]['tie_breaking_result']
                            for heuristic_type in short_name:
                                resulting_cost = short_name[heuristic_type]['cost'] if short_name[heuristic_type]['cost'] != -1 else 0
                                expanded_states = short_name[heuristic_type]['expanded_states'] if short_name[heuristic_type]['expanded_states'] != -1 else 0

                                # Either MDP or PDB
                                domain_results[problem_name][heuristic_type]['cost'].append(resulting_cost)
                                domain_results[problem_name][heuristic_type]['expanded'].append(expanded_states)

                                # Gamma
                                domain_results[problem_name]['gamma'][gamma][heuristic_type]['cost'].append(resulting_cost)
                                domain_results[problem_name]['gamma'][gamma][heuristic_type]['expanded'].append(expanded_states)

                                # Pattern type costs
                                domain_results[problem_name]['pattern_size_cost'][pattern_size]['cost'].append(resulting_cost)
                                domain_results[problem_name]['pattern_size_cost'][pattern_size][heuristic_type].append(resulting_cost)
                                domain_results[problem_name]['pattern_size_cost'][pattern_size]['expand'].append(expanded_states)
                                domain_results[problem_name]['pattern_size_cost'][pattern_size][f'{heuristic_type}_expand'].append(expanded_states)

                                # Pattern selection type
                                domain_results[problem_name]['pattern_select_type'][pattern_type].append(resulting_cost)

        print(f"Done collecting data for {domain_name}")

        # Create plots for the domain
        os.makedirs(f'{output_dir}/{domain_name}', exist_ok=True)

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

        # Now bar plot between average pattern size cost and scatter between mdp vs pdb grouped by size
        # on each row is a pattern size len bar graphs + scatter plot
        # row amount = problem amount
        # fig, axs = create_domain_layout(domain_name, domain_results.keys())

        fig, axs = plt.subplots(len(domain_results.keys()), 2, figsize=(15, 15))
        for i, problem_name in enumerate(domain_results):
            problem = domain_results[problem_name]

            # Bar plot for average pattern size cost
            ax_bar = axs[i, 0]
            for pattern_size in problem['pattern_size_cost']:
                ax_bar.bar(pattern_size, np.sum(problem['pattern_size_cost'][pattern_size]['cost']) / np.count_nonzero(
                    problem['pattern_size_cost'][pattern_size]['cost']))
            ax_bar.set_title(f"{problem_name} - Average Pattern Size Cost")
            ax_bar.set_ylabel('Average Cost')
            ax_bar.set_xlabel('Pattern Size')

            # Scatter plot for MDP vs PDB cost
            ax_scatter = axs[i, 1]
            for pattern_size in problem['pattern_size_cost']:
                ax_scatter.scatter(problem['pattern_size_cost'][pattern_size]['mdp'], problem['pattern_size_cost'][pattern_size]['pdb'], alpha=0.8, label=f'Pattern Size: {pattern_size}')
            ax_scatter.plot([0, max(problem['pattern_size_cost'][pattern_size]['mdp'])], [0, max(problem['pattern_size_cost'][pattern_size]['pdb'])], 'r--', alpha=0.5)

            ax_scatter.set_title(f"{problem_name} - MDP vs PDB Cost")
            ax_scatter.set_xlabel('MDP Cost')
            ax_scatter.set_ylabel('PDB Cost')
            ax_scatter.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{domain_name}/average_pattern_size_cost_and_scatter.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

        fig, axs = plt.subplots(len(domain_results.keys()), 2, figsize=(15, 15))
        for i, problem_name in enumerate(domain_results):
            problem = domain_results[problem_name]

            # Bar plot for average pattern size cost
            ax_bar = axs[i, 0]
            for pattern_size in problem['pattern_size_cost']:
                ax_bar.bar(pattern_size, np.sum(problem['pattern_size_cost'][pattern_size]['expand']) / np.count_nonzero(
                    problem['pattern_size_cost'][pattern_size]['expand']))
            ax_bar.set_title(f"{problem_name} - Average Pattern Size Cost")
            ax_bar.set_ylabel('Average Cost')
            ax_bar.set_xlabel('Pattern Size')

            # Scatter plot for MDP vs PDB cost
            ax_scatter = axs[i, 1]
            for pattern_size in problem['pattern_size_cost']:
                ax_scatter.scatter(problem['pattern_size_cost'][pattern_size]['mdp_expand'],
                                   problem['pattern_size_cost'][pattern_size]['pdb_expand'], alpha=0.8,
                                   label=f'Pattern Size: {pattern_size}')
            ax_scatter.plot([0, max(problem['pattern_size_cost'][pattern_size]['mdp_expand'])],
                            [0, max(problem['pattern_size_cost'][pattern_size]['pdb_expand'])], 'r--', alpha=0.5)

            ax_scatter.set_title(f"{problem_name} - MDP vs PDB Expanded")
            ax_scatter.set_xscale('log')
            ax_scatter.set_yscale('log')
            ax_scatter.set_xlabel('MDP Expanded')
            ax_scatter.set_ylabel('PDB Expanded')
            ax_scatter.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{domain_name}/average_pattern_size_exp_and_scatter.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Gamma bar plots
        # columns for gamma values, rows for problems
        fig, axs = plt.subplots(len(domain_results.keys()), 3, figsize=(15, 15))
        for i, problem_name in enumerate(domain_results):
            for j, gamma in enumerate(domain_results[problem_name]['gamma']):
                problem = domain_results[problem_name]['gamma'][gamma]

                # Bar plot for average pattern size cost
                ax_bar = axs[i, j]
                ax_bar.bar(['MDP', 'PDB'], [np.sum(problem['mdp']['cost']) / np.count_nonzero(problem['mdp']['cost']),
                                            np.sum(problem['pdb']['cost']) / np.count_nonzero(problem['pdb']['cost'])], color=['skyblue', 'lightcoral'])
                ax_bar.set_title(f"{problem_name}-Avg Cost-Gamma: {gamma}")
                ax_bar.set_ylabel('Average Cost')
                ax_bar.set_xlabel('Heuristic')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{domain_name}/average_cost_gamma.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Gamma scatter plots
        # columns for gamma values, rows for problems
        fig, axs = plt.subplots(len(domain_results.keys()), 3, figsize=(15, 15))
        for i, problem_name in enumerate(domain_results):
            for j, gamma in enumerate(domain_results[problem_name]['gamma']):
                problem = domain_results[problem_name]['gamma'][gamma]

                # Scatter plot for MDP vs PDB cost
                ax_scatter = axs[i, j]
                ax_scatter.scatter(problem['mdp']['cost'], problem['pdb']['cost'], alpha=0.8, color="green")
                ax_scatter.plot([0, max(problem['mdp']['cost'])], [0, max(problem['mdp']['cost'])], 'r--', alpha=0.5)
                ax_scatter.set_title(f"{problem_name} - MDP vs PDB Cost - Gamma: {gamma}")
                ax_scatter.set_xlabel('MDP Cost')
                ax_scatter.set_ylabel('PDB Cost')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{domain_name}/mdp_vs_pdb_cost_gamma.png', dpi=300, bbox_inches='tight')
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
    os.makedirs('../plots_newest', exist_ok=True)
    plot_results('comparison_newest.json', '../plots_newest')
