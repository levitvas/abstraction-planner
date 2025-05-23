import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'figure.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
})


def load_and_combine_results(json_files: List[str]) -> pd.DataFrame:
    """Load multiple JSON files and combine results."""
    all_data = []
    hff_data = {}

    # Load all files
    for i, json_file in enumerate(json_files):
        print(f"Loading {json_file}...")
        with open(json_file, 'r') as f:
            data = json.load(f)

        # If first file, extract HFF data
        if i == 0:
            for problem_key, problem_data in data.items():
                if 'FF' in problem_data:
                    hff_data[problem_key] = problem_data['FF']

        # Extract MDP data from all files
        for problem_key, problem_data in data.items():
            if 'MDP_Grow' in problem_data and 'problem_info' in problem_data:
                domain = problem_data['problem_info']['domain']
                problem = problem_data['problem_info']['problem']

                mdp_result = problem_data['MDP_Grow']
                hff_result = hff_data.get(problem_key, {'solved': False, 'cost': -1, 'expanded': -1})

                # Create config name from filename (no special case for first file)
                config_name = os.path.splitext(os.path.basename(json_file))[0]

                record = {
                    'domain': domain,
                    'problem': problem,
                    'problem_key': problem_key,
                    'hff_solved': hff_result.get('solved', False),
                    'hff_cost': hff_result.get('cost', -1),
                    'hff_expanded': hff_result.get('expanded', -1),
                    'mdp_solved': mdp_result.get('solved', False),
                    'mdp_cost': mdp_result.get('cost', -1),
                    'mdp_expanded': mdp_result.get('expanded', -1),
                    'config_name': config_name
                }
                all_data.append(record)

    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} configurations from {len(json_files)} files")
    return df


def create_clear_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Create two clear, large scatter plots: Cost and Expansion."""

    # Filter for valid data where both methods solved
    valid_df = df[(df['hff_solved']) & (df['mdp_solved']) &
                  (df['hff_cost'] > 0) & (df['mdp_cost'] > 0) &
                  (df['hff_expanded'] > 0) & (df['mdp_expanded'] > 0)]

    if valid_df.empty:
        print("No valid data for plotting")
        return

    # Filter outliers for better visualization
    cost_df = valid_df[(valid_df['hff_cost'] < 65) & (valid_df['mdp_cost'] < 65)]
    exp_df = valid_df  # Keep all expansion data

    configs = sorted(valid_df['config_name'].unique())

    # Use distinct colors but SAME markers (circles)
    colors = sns.color_palette("Set1", len(configs))

    # Create two large, clear plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Cost Comparison
    if not cost_df.empty:
        for i, config in enumerate(configs):
            config_data = cost_df[cost_df['config_name'] == config]
            if not config_data.empty:
                ax1.scatter(config_data['hff_cost'], config_data['mdp_cost'],
                            alpha=0.8, c=[colors[i]], s=120, marker='o',
                            edgecolors='black', linewidth=1.2, label=config)

        # Add diagonal reference line
        max_cost = max(cost_df['hff_cost'].max(), cost_df['mdp_cost'].max())
        min_cost = min(cost_df['hff_cost'].min(), cost_df['mdp_cost'].min())
        ax1.plot([min_cost, max_cost], [min_cost, max_cost], 'k-', alpha=0.7, linewidth=3, label='Equal Performance')

        # Add performance regions with better visibility
        ax1.fill_between([min_cost, max_cost], [min_cost, max_cost], [max_cost, max_cost],
                         color='red', alpha=0.04, label='HFF Better Region')
        ax1.fill_between([min_cost, max_cost], [min_cost, min_cost], [min_cost, max_cost],
                         color='green', alpha=0.05, label='MDP Better Region')

        ax1.set_xlabel('HFF Solution Cost', fontweight='bold')
        ax1.set_ylabel('MDP Solution Cost', fontweight='bold')
        ax1.set_title('Solution Quality: MDP vs HFF', fontweight='bold', fontsize=18)
        ax1.grid(True, alpha=0.4)

        # Add overall statistics
        total_cost_points = len(cost_df)
        mdp_cost_wins = (cost_df['mdp_cost'] < cost_df['hff_cost']).sum()
        cost_ties = (cost_df['mdp_cost'] == cost_df['hff_cost']).sum()
        hff_cost_wins = total_cost_points - mdp_cost_wins - cost_ties

        stats_text = f'Total Points: {total_cost_points}\nMDP Wins: {mdp_cost_wins} ({mdp_cost_wins / total_cost_points * 100:.1f}%)\nHFF Wins: {hff_cost_wins} ({hff_cost_wins / total_cost_points * 100:.1f}%)\nTies: {cost_ties}'
        ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
                 horizontalalignment='right', verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Plot 2: Expansion Comparison
    if not exp_df.empty:
        for i, config in enumerate(configs):
            config_data = exp_df[exp_df['config_name'] == config]
            if not config_data.empty:
                ax2.scatter(config_data['hff_expanded'], config_data['mdp_expanded'],
                            alpha=0.8, c=[colors[i]], s=120, marker='o',
                            edgecolors='black', linewidth=1.2, label=config)

        # Add diagonal reference line
        max_exp = max(exp_df['hff_expanded'].max(), exp_df['mdp_expanded'].max())
        min_exp = min(exp_df['hff_expanded'].min(), exp_df['mdp_expanded'].min())
        ax2.plot([min_exp, max_exp], [min_exp, max_exp], 'k-', alpha=0.7, linewidth=3, label='Equal Performance')

        ax2.set_xlabel('HFF Expanded States', fontweight='bold')
        ax2.set_ylabel('MDP Expanded States', fontweight='bold')
        ax2.set_title('Search Efficiency: MDP vs HFF', fontweight='bold', fontsize=18)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.4, which='both')

        # Add overall statistics
        total_exp_points = len(exp_df)
        mdp_exp_wins = (exp_df['mdp_expanded'] < exp_df['hff_expanded']).sum()
        exp_ties = (exp_df['mdp_expanded'] == exp_df['hff_expanded']).sum()
        hff_exp_wins = total_exp_points - mdp_exp_wins - exp_ties

        stats_text = f'Total Points: {total_exp_points}\nMDP Wins: {mdp_exp_wins} ({mdp_exp_wins / total_exp_points * 100:.1f}%)\nHFF Wins: {hff_exp_wins} ({hff_exp_wins / total_exp_points * 100:.1f}%)\nTies: {exp_ties}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                 horizontalalignment='left', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Add common legend below both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(configs) + 3, 6),
               bbox_to_anchor=(0.5, 0.02), frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    plt.savefig(f'{output_dir}/mdp_vs_hff_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved clear comparison plots to {output_dir}/mdp_vs_hff_comparison.png")


def create_simple_comparison_table(df: pd.DataFrame, output_dir: str):
    """Create a simple domain coverage table plus analysis."""

    configs = sorted(df['config_name'].unique())
    domains = sorted(df['domain'].unique())

    # Write domain coverage table
    with open(f'{output_dir}/domain_coverage_table.txt', 'w', encoding='utf-8') as f:
        f.write("DOMAIN COVERAGE TABLE\n")
        f.write("=" * 80 + "\n\n")

        # Table header
        f.write(f"{'Domain':<15}")
        f.write(f"{'HFF':<12}")
        for config in configs:
            f.write(f"{config[:16]:<18}")
        f.write("\n")
        f.write("-" * 80 + "\n")

        # Table rows - one per domain
        total_problems_all = 0
        hff_total_solved = 0
        config_totals = {config: 0 for config in configs}

        for domain in domains:
            domain_df = df[df['domain'] == domain]

            # Get unique problems in this domain
            unique_problems = domain_df['problem_key'].unique()
            total_problems = len(unique_problems)
            total_problems_all += total_problems

            f.write(f"{domain:<15}")

            # HFF column - count unique problems solved by HFF
            hff_solved = 0
            for problem_key in unique_problems:
                problem_data = domain_df[domain_df['problem_key'] == problem_key].iloc[0]
                if problem_data['hff_solved']:
                    hff_solved += 1
            hff_total_solved += hff_solved
            f.write(f"{hff_solved}/{total_problems:<8}")

            # MDP config columns
            for config in configs:
                mdp_solved = 0
                for problem_key in unique_problems:
                    config_problem_data = domain_df[
                        (domain_df['problem_key'] == problem_key) &
                        (domain_df['config_name'] == config)
                        ]
                    if not config_problem_data.empty and config_problem_data.iloc[0]['mdp_solved']:
                        mdp_solved += 1

                config_totals[config] += mdp_solved
                f.write(f"{mdp_solved}/{total_problems:<8}")
            f.write("\n")

        # Add totals row
        f.write("-" * 80 + "\n")
        f.write(f"{'TOTAL':<15}")
        f.write(f"{hff_total_solved}/{total_problems_all:<8}")

        for config in configs:
            f.write(f"{config_totals[config]}/{total_problems_all:<8}")
        f.write("\n")

        # Add simple analysis
        f.write("\n\nPERFORMANCE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")

        # Coverage analysis
        f.write("COVERAGE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(
            f"HFF Baseline: {hff_total_solved}/{total_problems_all} ({hff_total_solved / total_problems_all * 100:.1f}%)\n")

        for config in configs:
            solved = config_totals[config]
            rate = solved / total_problems_all * 100
            improvement = solved - hff_total_solved
            f.write(f"{config}: {solved}/{total_problems_all} ({rate:.1f}%) [{improvement:+d} vs HFF]\n")

        # Cost/Expansion performance analysis
        f.write(f"\nQUALITY ANALYSIS (when both solve):\n")
        f.write("-" * 35 + "\n")

        for config in configs:
            config_df = df[df['config_name'] == config]

            # Cost analysis
            both_cost = config_df[
                config_df['hff_solved'] & config_df['mdp_solved'] &
                (config_df['hff_cost'] > 0) & (config_df['mdp_cost'] > 0)
                ]

            if not both_cost.empty:
                mdp_cost_wins = (both_cost['mdp_cost'] < both_cost['hff_cost']).sum()
                cost_ties = (both_cost['mdp_cost'] == both_cost['hff_cost']).sum()
                cost_win_rate = mdp_cost_wins / len(both_cost) * 100

                cost_improvements = ((both_cost['hff_cost'] - both_cost['mdp_cost']) / both_cost['hff_cost'] * 100)
                avg_cost_improvement = cost_improvements.mean()
            else:
                cost_win_rate = avg_cost_improvement = 0
                mdp_cost_wins = cost_ties = 0

            # Expansion analysis
            both_exp = config_df[
                (config_df['hff_expanded'] > 0) & (config_df['mdp_expanded'] > 0)
                ]

            if not both_exp.empty:
                mdp_exp_wins = (both_exp['mdp_expanded'] < both_exp['hff_expanded']).sum()
                exp_win_rate = mdp_exp_wins / len(both_exp) * 100
            else:
                exp_win_rate = 0
                mdp_exp_wins = 0

            f.write(f"\n{config}:\n")
            f.write(
                f"  Cost: {mdp_cost_wins}/{len(both_cost)} wins ({cost_win_rate:.1f}%), avg improvement {avg_cost_improvement:+.1f}%\n")
            f.write(f"  Expansions: {mdp_exp_wins}/{len(both_exp)} wins ({exp_win_rate:.1f}%)\n")

        # Best performing configs
        f.write(f"\nBEST PERFORMERS:\n")
        f.write("-" * 15 + "\n")

        # Best coverage
        best_coverage_config = max(configs, key=lambda c: config_totals[c])
        best_coverage_improvement = config_totals[best_coverage_config] - hff_total_solved
        f.write(f"Coverage: {best_coverage_config} (+{best_coverage_improvement} problems vs HFF)\n")

        # Best cost performance
        best_cost_config = None
        best_cost_win_rate = 0

        for config in configs:
            config_df = df[df['config_name'] == config]
            both_cost = config_df[
                config_df['hff_solved'] & config_df['mdp_solved'] &
                (config_df['hff_cost'] > 0) & (config_df['mdp_cost'] > 0)
                ]

            if not both_cost.empty:
                mdp_cost_wins = (both_cost['mdp_cost'] < both_cost['hff_cost']).sum()
                win_rate = mdp_cost_wins / len(both_cost) * 100
                if win_rate > best_cost_win_rate:
                    best_cost_win_rate = win_rate
                    best_cost_config = config

        if best_cost_config:
            f.write(f"Cost Quality: {best_cost_config} ({best_cost_win_rate:.1f}% win rate vs HFF)\n")

    print(f"Saved domain coverage table with analysis to {output_dir}/domain_coverage_table.txt")


def main():
    # Configuration - modify these paths
    json_files = [
        'v3_5_init.json',  # File with both HFF and MDP
        # Add your other MDP-only files here:
        'v2_30_avg_minus.json',
        'v3_20_avg_minus.json',
        'v2_30_init.json',
        'v2_20_init.json',
        'v3_20_avg.json',
        'v2_40_avg.json',
    ]

    output_dir = 'hff_mdp_comparison_results'

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Check if files exist
    existing_files = [f for f in json_files if os.path.exists(f)]
    if not existing_files:
        print("No JSON files found! Please update the json_files list.")
        return

    # Load and combine data
    df = load_and_combine_results(existing_files)
    if df.empty:
        print("No data loaded!")
        return

    print(f"Loaded {df['domain'].nunique()} domains, {len(df)} total instances")
    print(f"Configurations: {', '.join(sorted(df['config_name'].unique()))}")

    # Create outputs
    create_clear_comparison_plots(df, output_dir)
    create_simple_comparison_table(df, output_dir)

    print(f"\nComparison complete! Results saved to {output_dir}/")


if __name__ == '__main__':
    main()