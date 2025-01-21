import os

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import json

def plot_results(results_file: str, output_dir: str):
    """
    Create plots from the results JSON file.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    tie_breaking_methods = ['random', 'average']
    pattern_sizes = ['num_3', 'num_6', 'num_12']

    # First create the bar plots
    for tie_breaking in tie_breaking_methods:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        pattern_labels = []
        pdb_costs = []
        alt_costs = []
        pdb_expanded = []
        alt_expanded = []

        for pattern_size in pattern_sizes:
            valid_problems = 0
            pdb_cost_sum = 0
            alt_cost_sum = 0
            pdb_exp_sum = 0
            alt_exp_sum = 0

            for problem in results:
                if pattern_size in problem['pattern_variations']:
                    variation = problem['pattern_variations'][pattern_size]
                    tie_breaking_result = variation['tie_breaking_results'][tie_breaking]

                    if (tie_breaking_result['pdb']['cost'] != -1 and
                            tie_breaking_result['alternation']['cost'] != -1):
                        valid_problems += 1
                        pdb_cost_sum += tie_breaking_result['pdb']['cost']
                        alt_cost_sum += tie_breaking_result['alternation']['cost']
                        pdb_exp_sum += tie_breaking_result['pdb']['expanded_states']
                        alt_exp_sum += tie_breaking_result['alternation']['expanded_states']

            if valid_problems > 0:
                pattern_labels.append(pattern_size.replace('num_', ''))
                pdb_costs.append(pdb_cost_sum / valid_problems)
                alt_costs.append(alt_cost_sum / valid_problems)
                pdb_expanded.append(pdb_exp_sum / valid_problems)
                alt_expanded.append(alt_exp_sum / valid_problems)

        x = np.arange(len(pattern_labels))
        width = 0.35

        ax1.bar(x - width / 2, pdb_costs, width, label='PDB', color='skyblue')
        ax1.bar(x + width / 2, alt_costs, width, label='Alternation', color='lightcoral')
        ax1.set_ylabel('Average Plan Cost')
        ax1.set_title(f'Plan Cost Comparison ({tie_breaking} tie-breaking)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(pattern_labels)
        ax1.legend()
        ax1.set_xlabel('Number of Patterns')

        ax2.bar(x - width / 2, pdb_expanded, width, label='PDB', color='skyblue')
        ax2.bar(x + width / 2, alt_expanded, width, label='Alternation', color='lightcoral')
        ax2.set_ylabel('Average Expanded States')
        ax2.set_title(f'Expanded States Comparison ({tie_breaking} tie-breaking)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(pattern_labels)
        ax2.legend()
        ax2.set_xlabel('Number of Patterns')
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_bars_{tie_breaking}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Now create scatter plots for each pattern size and tie breaking method
    for tie_breaking in tie_breaking_methods:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        max_cost, min_cost = 0, float('inf')
        max_exp, min_exp = 0, float('inf')
        for pattern_size in pattern_sizes:
            pdb_costs = []
            alt_costs = []
            pdb_expanded = []
            alt_expanded = []
            problem_names = []

            for problem in results:
                if pattern_size in problem['pattern_variations']:
                    variation = problem['pattern_variations'][pattern_size]
                    tie_breaking_result = variation['tie_breaking_results'][tie_breaking]

                    if (tie_breaking_result['pdb']['cost'] != -1 and
                            tie_breaking_result['alternation']['cost'] != -1):
                        pdb_costs.append(tie_breaking_result['pdb']['cost'])
                        alt_costs.append(tie_breaking_result['alternation']['cost'])
                        pdb_expanded.append(tie_breaking_result['pdb']['expanded_states'])
                        alt_expanded.append(tie_breaking_result['alternation']['expanded_states'])
                        problem_names.append(problem['problem'])

            if pdb_costs:  # Only create plots if we have data
                # Plan Cost scatter plot
                max_cost = max(max(max(pdb_costs), max(alt_costs)), max_cost)
                min_cost = min(min(min(pdb_costs), min(alt_costs)), min_cost)
                ax1.scatter(pdb_costs, alt_costs, alpha=0.8, label=pattern_size)
                ax1.plot([min_cost, max_cost], [min_cost, max_cost], 'r--', alpha=0.5)  # diagonal line
                ax1.set_xlabel('PDB Cost')
                ax1.set_ylabel('Alternation Cost')
                ax1.set_title(f'Plan Cost Comparison\n{pattern_size} patterns, {tie_breaking} tie-breaking')
                ax1.legend()

                # Add some padding to the limits
                ax1.set_xlim(min_cost * 0.95, max_cost * 1.05)
                ax1.set_ylim(min_cost * 0.95, max_cost * 1.05)

                # Expanded States scatter plot (log scale)
                ax2.scatter(pdb_expanded, alt_expanded, alpha=0.8, label=pattern_size)
                max_exp = max(max(max(pdb_expanded), max(alt_expanded)), max_exp)
                min_exp = min(min(min(pdb_expanded), min(alt_expanded)), min_exp)
                ax2.plot([min_exp, max_exp], [min_exp, max_exp], 'r--', alpha=0.5)  # diagonal line
                ax2.set_xlabel('PDB Expanded States')
                ax2.set_ylabel('Alternation Expanded States')
                ax2.set_title(f'Expanded States Comparison\n{pattern_size} patterns, {tie_breaking} tie-breaking')
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                ax2.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/scatter_{tie_breaking}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Create summary statistics
    print("\nSummary Statistics:")
    for tie_breaking in tie_breaking_methods:
        print(f"\nTie-breaking method: {tie_breaking}")
        for pattern_size in pattern_sizes:
            solved_count = {'pdb': 0, 'alternation': 0}
            total_problems = 0

            for problem in results:
                if pattern_size in problem['pattern_variations']:
                    total_problems += 1
                    variation = problem['pattern_variations'][pattern_size]
                    tie_breaking_result = variation['tie_breaking_results'][tie_breaking]

                    if tie_breaking_result['pdb']['cost'] != -1:
                        solved_count['pdb'] += 1
                    if tie_breaking_result['alternation']['cost'] != -1:
                        solved_count['alternation'] += 1

            if total_problems > 0:
                print(f"\nPattern size {pattern_size}:")
                print(f"PDB solved: {solved_count['pdb']}/{total_problems} "
                      f"({solved_count['pdb'] * 100 / total_problems:.1f}%)")
                print(f"Alternation solved: {solved_count['alternation']}/{total_problems} "
                      f"({solved_count['alternation'] * 100 / total_problems:.1f}%)")

if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    plot_results('prev_results.json', 'plots')