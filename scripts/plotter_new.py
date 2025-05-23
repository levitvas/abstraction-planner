import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# --- Configuration & Styling ---
DEFAULT_GAMMA = 0.99


def setup_plotting_style():
    """Sets clean academic style for plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': 'gray',
    })


# Academic color palette
COLORS = {
    'mdp': '#2E86C1',  # Professional blue
    'pdb': '#E74C3C',  # Professional red
    'tie': '#95A5A6',  # Gray
    'better_region': '#E8F6F3',  # Light green
    'worse_region': '#FADBD8',  # Light red
}

# Pattern type mapping for cleaner labels
PATTERN_LABELS = {
    'sorted_init': 'Initial State',
    'sorted_avg': 'Average',
    'sorted_goal_avg': 'Goal Average',
    'random': 'Random'
}

setup_plotting_style()


# --- Data Loading and Preprocessing (keeping your existing functions) ---
def parse_experiment_data(raw_data: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Parses the raw JSON data into a structured DataFrame."""
    records = []
    print("Starting data parsing...")

    for domain_item in raw_data:
        domain_name = domain_item.get('domain')
        if not domain_name:
            continue

        problems = domain_item.get('problems', {})
        for problem_name, problem_details in problems.items():
            for pattern_gen_type in ['sorted_patterns_init', 'sorted_patterns_avg', 'sorted_patterns_goal_avg']:
                if pattern_gen_type not in problem_details:
                    continue

                # Map pattern types to cleaner names
                pattern_type_map = {
                    'sorted_patterns_init': 'sorted_init',
                    'sorted_patterns_avg': 'sorted_avg',
                    'sorted_patterns_goal_avg': 'sorted_goal_avg'
                }
                pattern_type_label = pattern_type_map[pattern_gen_type]
                patterns_by_size = problem_details[pattern_gen_type]

                for size_key, patterns_by_amount in patterns_by_size.items():
                    size_match = re.search(r'pattern_size_(\d+)', size_key)
                    if not size_match:
                        continue
                    pattern_size = int(size_match.group(1))

                    for amount_key, config_details in patterns_by_amount.items():
                        amount_match = re.search(r'pattern_amount_(\d+)', amount_key)
                        if not amount_match:
                            continue
                        pattern_amount = int(amount_match.group(1))

                        gamma_value = config_details.get('gamma', DEFAULT_GAMMA)
                        results = config_details.get('tie_breaking_result', {})
                        pdb_res = results.get('pdb', {})
                        mdp_res = results.get('mdp', {})

                        pdb_cost = pdb_res.get('cost', -1)
                        pdb_expanded = pdb_res.get('expanded_states', -1)
                        mdp_cost = mdp_res.get('cost', -1)
                        mdp_expanded = mdp_res.get('expanded_states', -1)

                        # Skip if neither method produced valid results
                        if pdb_cost <= 0 and mdp_cost <= 0:
                            continue

                        record = {
                            'domain': domain_name,
                            'problem': problem_name,
                            'pattern_type': pattern_type_label,
                            'gamma': gamma_value,
                            'pattern_size': pattern_size,
                            'pattern_amount': pattern_amount,
                            'pdb_cost': pdb_cost,
                            'mdp_cost': mdp_cost,
                            'pdb_expanded': pdb_expanded,
                            'mdp_expanded': mdp_expanded,
                        }
                        records.append(record)

    if not records:
        print("ERROR: No valid records processed.")
        return None

    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} records from {df['domain'].nunique()} domains.")
    return df


def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates comparison metrics and performance indicators."""
    if df.empty:
        return df

    # Solved status
    df['pdb_solved'] = df['pdb_cost'] > 0
    df['mdp_solved'] = df['mdp_cost'] > 0

    # Winner determination for cost
    df['cost_winner'] = 'tie'
    mdp_only = df['mdp_solved'] & (~df['pdb_solved'])
    pdb_only = df['pdb_solved'] & (~df['mdp_solved'])
    both_solved = df['pdb_solved'] & df['mdp_solved']

    df.loc[mdp_only, 'cost_winner'] = 'mdp'
    df.loc[pdb_only, 'cost_winner'] = 'pdb'
    df.loc[both_solved & (df['mdp_cost'] < df['pdb_cost']), 'cost_winner'] = 'mdp'
    df.loc[both_solved & (df['pdb_cost'] < df['mdp_cost']), 'cost_winner'] = 'pdb'

    # Performance improvement (positive means MDP is better)
    df['cost_improvement'] = np.nan
    df.loc[both_solved, 'cost_improvement'] = (df['pdb_cost'] - df['mdp_cost']) / df['pdb_cost'] * 100

    # Expanded states analysis
    df['pdb_expanded_valid'] = df['pdb_expanded'] > 0
    df['mdp_expanded_valid'] = df['mdp_expanded'] > 0

    df['expanded_winner'] = 'tie'
    mdp_exp_only = df['mdp_expanded_valid'] & (~df['pdb_expanded_valid'])
    pdb_exp_only = df['pdb_expanded_valid'] & (~df['mdp_expanded_valid'])
    both_exp_valid = df['pdb_expanded_valid'] & df['mdp_expanded_valid']

    df.loc[mdp_exp_only, 'expanded_winner'] = 'mdp'
    df.loc[pdb_exp_only, 'expanded_winner'] = 'pdb'
    df.loc[both_exp_valid & (df['mdp_expanded'] < df['pdb_expanded']), 'expanded_winner'] = 'mdp'
    df.loc[both_exp_valid & (df['pdb_expanded'] < df['mdp_expanded']), 'expanded_winner'] = 'pdb'

    df['expanded_improvement'] = np.nan
    df.loc[both_exp_valid, 'expanded_improvement'] = (df['pdb_expanded'] - df['mdp_expanded']) / df[
        'pdb_expanded'] * 100

    # Clean pattern type labels
    df['pattern_type_clean'] = df['pattern_type'].map(PATTERN_LABELS)

    print("Derived metrics calculated.")
    return df


def load_and_preprocess_data(json_filepath: str) -> Optional[pd.DataFrame]:
    """Main data loading function."""
    try:
        with open(json_filepath, 'r') as f:
            raw_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data: {e}")
        return None

    df_parsed = parse_experiment_data(raw_data)
    if df_parsed is None or df_parsed.empty:
        return None

    return calculate_derived_metrics(df_parsed)


# --- Publication-Quality Plotting Functions ---

def plot_overall_performance_summary(df: pd.DataFrame, output_dir: Path):
    """5.4.1 Overall Performance Summary - Scatter plots comparing MDP-H vs PDB-H"""
    if df.empty:
        return

    # Filter for cases where both methods solved the problem
    both_solved = df[df['pdb_solved'] & df['mdp_solved']].copy()
    both_expanded = df[df['pdb_expanded_valid'] & df['mdp_expanded_valid']].copy()

    if both_solved.empty and both_expanded.empty:
        print("No data for overall performance comparison.")
        return

    # Create a distinct color palette for pattern types
    pattern_types = df['pattern_type_clean'].unique()
    colors = sns.color_palette("hls", len(pattern_types))
    pattern_color_dict = dict(zip(pattern_types, colors))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Cost comparison
    if not both_solved.empty:
        # Remove outliers for better visualization
        cost_df = both_solved[(both_solved['pdb_cost'] < 100) & (both_solved['mdp_cost'] < 100)]

        if not cost_df.empty:
            # Create scatter with color by pattern type
            plot_data_cost = cost_df.copy()
            domain = plot_data_cost['domain'].iloc[0]  # Assuming 'domain' column exists in the DataFrame
            jitter_strength = 0.7 if domain == 'sokoban-sat' else 0.1 if domain == 'elevator' else 0  # Default to 0 for others
            plot_data_cost['pdb_cost_j'] = plot_data_cost['pdb_cost'] + np.random.uniform(-jitter_strength,
                                                                                          jitter_strength,
                                                                                          size=len(plot_data_cost))
            plot_data_cost['mdp_cost_j'] = plot_data_cost['mdp_cost'] + np.random.uniform(-jitter_strength,
                                                                                          jitter_strength,
                                                                                          size=len(plot_data_cost))

            alpha = 0.5 if domain == 'sokoban-sat' else 0.7

            for pattern_type in pattern_types:
                pattern_data_j = plot_data_cost[plot_data_cost['pattern_type_clean'] == pattern_type]
                if not pattern_data_j.empty:
                    ax1.scatter(pattern_data_j['pdb_cost_j'], pattern_data_j['mdp_cost_j'],
                                alpha=alpha, c=[pattern_color_dict[pattern_type]], s=40,
                                edgecolors='white', linewidth=0.5, label=pattern_type,)

            # Add diagonal line
            max_cost = max(cost_df['pdb_cost'].max(), cost_df['mdp_cost'].max())
            min_cost = min(cost_df['pdb_cost'].min(), cost_df['mdp_cost'].min())
            ax1.plot([min_cost, max_cost], [min_cost, max_cost], 'k--', alpha=0.5, linewidth=1)

            # Shade regions
            ax1.fill_between([min_cost, max_cost], [min_cost, max_cost], [max_cost, max_cost],
                             color=COLORS['worse_region'], alpha=0.3, label='PDB-H Better')
            ax1.fill_between([min_cost, max_cost], [min_cost, min_cost], [min_cost, max_cost],
                             color=COLORS['better_region'], alpha=0.3, label='MDP-H Better')

            ax1.set_xlabel('PDB-H Solution Cost')
            ax1.set_ylabel('MDP-H Solution Cost')
            ax1.set_title('Solution Quality Comparison')

            # Add performance statistics
            wins_mdp = (cost_df['mdp_cost'] < cost_df['pdb_cost']).sum()
            tie = (cost_df['mdp_cost'] == cost_df['pdb_cost']).sum()
            total = len(cost_df)
            ax1.text(0.05, 0.95, f'MDP-H Wins/tie/total: {wins_mdp}/{tie}/{total} ({wins_mdp / total * 100:.1f}%)',
                     transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Expanded states comparison
    if not both_expanded.empty:
        # Use log scale for expanded states
        exp_df = both_expanded[both_expanded['pdb_expanded'] > 0]

        if not exp_df.empty:
            # Create scatter with color by pattern type
            domain = exp_df['domain'].iloc[0]
            jitter_strength = 500 if domain == 'sokoban-sat' else 0  # Default to 0 for others
            exp_df['pdb_expanded_j'] = exp_df['pdb_expanded'] + np.random.uniform(-jitter_strength,
                                                                                          jitter_strength,
                                                                                          size=len(exp_df))
            exp_df['mdp_expanded_j'] = exp_df['mdp_expanded'] + np.random.uniform(-jitter_strength,
                                                                                          jitter_strength,
                                                                                          size=len(exp_df))

            alpha = 0.5 if domain == 'sokoban-sat' else 0.7
            for pattern_type in pattern_types:
                pattern_data = exp_df[exp_df['pattern_type_clean'] == pattern_type]
                if not pattern_data.empty:
                    ax2.scatter(pattern_data['pdb_expanded_j'], pattern_data['mdp_expanded_j'],
                                alpha=alpha, c=[pattern_color_dict[pattern_type]], s=50,
                                edgecolors='white', linewidth=0.5, label=pattern_type)

            # Add diagonal line in log space
            ax2.plot([exp_df['pdb_expanded'].min(), exp_df['pdb_expanded'].max()],
                     [exp_df['pdb_expanded'].min(), exp_df['pdb_expanded'].max()],
                     'k--', alpha=0.5, linewidth=1)

            ax2.set_xlabel('PDB-H Expanded States')
            ax2.set_ylabel('MDP-H Expanded States')
            ax2.set_title('Search Efficiency Comparison')
            ax2.set_xscale('log')
            ax2.set_yscale('log')

            # Add performance statistics
            wins_mdp = (exp_df['mdp_expanded'] < exp_df['pdb_expanded']).sum()
            tie = (exp_df['mdp_expanded'] == exp_df['pdb_expanded']).sum()
            total = len(exp_df)
            ax2.text(0.05, 0.95, f'MDP-H Wins/tie/total: {wins_mdp}/{tie}/{total} ({wins_mdp / total * 100:.1f}%)',
                     transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Common legend - show pattern types only once
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Filter out the shaded region labels for better legend
    pattern_labels = {k: v for k, v in by_label.items() if k in pattern_types}

    # Add legend with pattern types only
    if pattern_labels:
        fig.legend(pattern_labels.values(), pattern_labels.keys(),
                   loc='lower center', ncol=min(len(pattern_labels), 4),
                   bbox_to_anchor=(0.5, 0.02), title="Pattern Selection Strategy")

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved overall performance summary plot.")


def plot_pattern_strategy_impact(df: pd.DataFrame, output_dir: Path):
    """5.4.2 Impact of Pattern Selection Strategy"""
    if df.empty:
        return

    # Check if we have enough variety in pattern types
    if df['pattern_type_clean'].nunique() < 2:
        print("Skipping pattern strategy impact - insufficient pattern type variety.")
        return

    # Win rate analysis by pattern type
    pattern_stats = []
    for pattern_type in df['pattern_type_clean'].unique():
        pattern_df = df[df['pattern_type_clean'] == pattern_type]

        cost_wins = (pattern_df['cost_winner'] == 'mdp').sum()
        cost_total = len(pattern_df[pattern_df['cost_winner'] != 'tie'])
        cost_win_rate = cost_wins / cost_total * 100 if cost_total > 0 else 0

        exp_wins = (pattern_df['expanded_winner'] == 'mdp').sum()
        exp_total = len(pattern_df[pattern_df['expanded_winner'] != 'tie'])
        exp_win_rate = exp_wins / exp_total * 100 if exp_total > 0 else 0

        # Average improvement for cases where both solved
        both_solved = pattern_df[pattern_df['pdb_solved'] & pattern_df['mdp_solved']]
        avg_cost_improvement = both_solved['cost_improvement'].mean() if not both_solved.empty else 0

        both_expanded = pattern_df[pattern_df['pdb_expanded_valid'] & pattern_df['mdp_expanded_valid']]
        avg_exp_improvement = both_expanded['expanded_improvement'].mean() if not both_expanded.empty else 0

        pattern_stats.append({
            'Pattern Strategy': pattern_type,
            'Cost Win Rate (%)': cost_win_rate,
            'Expansion Win Rate (%)': exp_win_rate,
            'Avg Cost Improvement (%)': avg_cost_improvement,
            'Avg Expansion Improvement (%)': avg_exp_improvement,
            'Total Instances': len(pattern_df)
        })

    stats_df = pd.DataFrame(pattern_stats)

    # Create visualization with adjusted figure size
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Win rates
    x_pos = range(len(stats_df))

    bars1 = ax1.bar(x_pos, stats_df['Cost Win Rate (%)'], color=COLORS['mdp'], alpha=0.7)
    ax1.set_xlabel('Pattern Selection Strategy')
    ax1.set_ylabel('MDP-H Win Rate (%)')
    ax1.set_title('Cost Performance by Pattern Strategy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stats_df['Pattern Strategy'], rotation=45, ha='right')
    ax1.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(x_pos, stats_df['Expansion Win Rate (%)'], color=COLORS['mdp'], alpha=0.7)
    ax2.set_xlabel('Pattern Selection Strategy')
    ax2.set_ylabel('MDP-H Win Rate (%)')
    ax2.set_title('Search Efficiency by Pattern Strategy')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stats_df['Pattern Strategy'], rotation=45, ha='right')
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 100)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Average improvements
    bars3 = ax3.bar(x_pos, stats_df['Avg Cost Improvement (%)'], color=COLORS['mdp'], alpha=0.7)
    ax3.set_xlabel('Pattern Selection Strategy')
    ax3.set_ylabel('Average Cost Improvement (%)')
    ax3.set_title('Average Cost Improvement (MDP-H vs PDB-H)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(stats_df['Pattern Strategy'], rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + (0.5 if height >= 0 else -1),
                 f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    bars4 = ax4.bar(x_pos, stats_df['Avg Expansion Improvement (%)'], color=COLORS['mdp'], alpha=0.7)
    ax4.set_xlabel('Pattern Selection Strategy')
    ax4.set_ylabel('Average Expansion Improvement (%)')
    ax4.set_title('Average Search Efficiency Improvement')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(stats_df['Pattern Strategy'], rotation=45, ha='right')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + (1 if height >= 0 else -2),
                 f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    # Ensure tight layout works or use alternative
    try:
        plt.tight_layout()
    except:
        plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.15, top=0.95)

    plt.savefig(output_dir / 'pattern_strategy_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save summary table
    with open(output_dir / 'pattern_strategy_summary.txt', 'w') as f:
        f.write("Pattern Selection Strategy Impact Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(stats_df.to_string(index=False, float_format='{:.2f}'.format))

    print("Saved pattern strategy impact analysis.")


def create_single_consolidated_table(df: pd.DataFrame, output_dir: Path):
    """Creates a single consolidated table with statistics for each domain plus overall totals."""
    if df.empty:
        return

    # List to hold table rows (domain stats + overall)
    all_stats = []

    # First calculate stats for each domain
    for domain in sorted(df['domain'].unique()):
        domain_df = df[df['domain'] == domain]

        total_instances = len(domain_df)

        # Solution stats
        mdp_solved = domain_df['mdp_solved'].sum()
        pdb_solved = domain_df['pdb_solved'].sum()
        both_solved = (domain_df['pdb_solved'] & domain_df['mdp_solved']).sum()

        # Cost performance
        cost_wins_mdp = (domain_df['cost_winner'] == 'mdp').sum()
        cost_wins_pdb = (domain_df['cost_winner'] == 'pdb').sum()
        cost_ties = (domain_df['cost_winner'] == 'tie').sum()

        # Get non-tie cost win rate
        total_decided = cost_wins_mdp + cost_wins_pdb
        cost_win_rate = cost_wins_mdp / total_decided * 100 if total_decided > 0 else 0

        # Cost improvement
        both_cost_df = domain_df[domain_df['pdb_solved'] & domain_df['mdp_solved']]
        avg_cost_improv = both_cost_df['cost_improvement'].mean() if not both_cost_df.empty else 0

        # Expanded states
        exp_wins_mdp = (domain_df['expanded_winner'] == 'mdp').sum()
        exp_wins_pdb = (domain_df['expanded_winner'] == 'pdb').sum()
        exp_ties = (domain_df['expanded_winner'] == 'tie').sum()

        # Get non-tie expanded win rate
        exp_total_decided = exp_wins_mdp + exp_wins_pdb
        exp_win_rate = exp_wins_mdp / exp_total_decided * 100 if exp_total_decided > 0 else 0

        # Expanded improvement
        both_exp_df = domain_df[domain_df['pdb_expanded_valid'] & domain_df['mdp_expanded_valid']]
        avg_exp_improv = both_exp_df['expanded_improvement'].mean() if not both_exp_df.empty else 0

        # Add domain stats to table rows
        all_stats.append({
            'Domain': domain,
            'Total': total_instances,
            'MDP Solved': mdp_solved,
            'PDB Solved': pdb_solved,
            'Both Solved': both_solved,
            'Cost MDP Wins': cost_wins_mdp,
            'Cost PDB Wins': cost_wins_pdb,
            'Cost Ties': cost_ties,
            'Cost Win %': cost_win_rate,
            'Avg Cost Improv %': avg_cost_improv,
            'Exp MDP Wins': exp_wins_mdp,
            'Exp PDB Wins': exp_wins_pdb,
            'Exp Ties': exp_ties,
            'Exp Win %': exp_win_rate,
            'Avg Exp Improv %': avg_exp_improv
        })

    # Now calculate overall stats across all domains
    total_instances = len(df)

    # Solution stats
    total_mdp_solved = df['mdp_solved'].sum()
    total_pdb_solved = df['pdb_solved'].sum()
    total_both_solved = (df['pdb_solved'] & df['mdp_solved']).sum()

    # Cost performance
    total_cost_wins_mdp = (df['cost_winner'] == 'mdp').sum()
    total_cost_wins_pdb = (df['cost_winner'] == 'pdb').sum()
    total_cost_ties = (df['cost_winner'] == 'tie').sum()

    # Get non-tie cost win rate
    total_cost_decided = total_cost_wins_mdp + total_cost_wins_pdb
    total_cost_win_rate = total_cost_wins_mdp / total_cost_decided * 100 if total_cost_decided > 0 else 0

    # Cost improvement
    total_both_cost_df = df[df['pdb_solved'] & df['mdp_solved']]
    total_avg_cost_improv = total_both_cost_df['cost_improvement'].mean() if not total_both_cost_df.empty else 0

    # Expanded states
    total_exp_wins_mdp = (df['expanded_winner'] == 'mdp').sum()
    total_exp_wins_pdb = (df['expanded_winner'] == 'pdb').sum()
    total_exp_ties = (df['expanded_winner'] == 'tie').sum()

    # Get non-tie expanded win rate
    total_exp_decided = total_exp_wins_mdp + total_exp_wins_pdb
    total_exp_win_rate = total_exp_wins_mdp / total_exp_decided * 100 if total_exp_decided > 0 else 0

    # Expanded improvement
    total_both_exp_df = df[df['pdb_expanded_valid'] & df['mdp_expanded_valid']]
    total_avg_exp_improv = total_both_exp_df['expanded_improvement'].mean() if not total_both_exp_df.empty else 0

    # Add overall row
    all_stats.append({
        'Domain': 'OVERALL',
        'Total': total_instances,
        'MDP Solved': total_mdp_solved,
        'PDB Solved': total_pdb_solved,
        'Both Solved': total_both_solved,
        'Cost MDP Wins': total_cost_wins_mdp,
        'Cost PDB Wins': total_cost_wins_pdb,
        'Cost Ties': total_cost_ties,
        'Cost Win %': total_cost_win_rate,
        'Avg Cost Improv %': total_avg_cost_improv,
        'Exp MDP Wins': total_exp_wins_mdp,
        'Exp PDB Wins': total_exp_wins_pdb,
        'Exp Ties': total_exp_ties,
        'Exp Win %': total_exp_win_rate,
        'Avg Exp Improv %': total_avg_exp_improv
    })

    # Convert to DataFrame for easier manipulation
    stats_df = pd.DataFrame(all_stats)

    # Write table to file
    with open(output_dir / 'consolidated_performance_table.txt', 'w') as f:
        f.write("MDP-H vs PDB-H PERFORMANCE COMPARISON\n")
        f.write("=" * 120 + "\n\n")

        # First section: Solution statistics
        f.write("DOMAIN PERFORMANCE STATISTICS\n")
        f.write("-" * 120 + "\n")

        # Table header
        # Shorter column headers for better readability
        f.write(
            "{:<15} | {:>6} | {:>10} {:>10} {:>10} | {:>9} {:>9} {:>8} {:>9} {:>13} | {:>9} {:>9} {:>8} {:>9} {:>13}\n".format(
                "Domain", "Total",
                "MDP Solved", "PDB Solved", "Both Solv",
                "MDP Wins", "PDB Wins", "Ties", "MDP Win%", "Avg Improv%",
                "MDP Wins", "PDB Wins", "Ties", "MDP Win%", "Avg Improv%"
            ))

        # Section headers for cost and expansion
        f.write("{:<15} | {:>6} | {:>10} {:>10} {:>10} | {:>38} | {:>38}\n".format(
            "", "", "", "", "", "--------------- COST ---------------", "------------ EXPANSIONS ------------"
        ))

        f.write("-" * 120 + "\n")

        # Data rows for each domain
        for i, row in enumerate(all_stats):
            # Special formatting for overall row
            if i == len(all_stats) - 1:  # Last row is overall
                f.write("-" * 120 + "\n")  # Separator before overall row

            f.write(
                "{:<15} | {:>6} | {:>10} {:>10} {:>10} | {:>9} {:>9} {:>8} {:>9.2f} {:>13.2f} | {:>9} {:>9} {:>8} {:>9.2f} {:>13.2f}\n".format(
                    row['Domain'], row['Total'],
                    row['MDP Solved'], row['PDB Solved'], row['Both Solved'],
                    row['Cost MDP Wins'], row['Cost PDB Wins'], row['Cost Ties'], row['Cost Win %'],
                    row['Avg Cost Improv %'],
                    row['Exp MDP Wins'], row['Exp PDB Wins'], row['Exp Ties'], row['Exp Win %'], row['Avg Exp Improv %']
                ))

    print(f"Created consolidated performance table in {output_dir / 'consolidated_performance_table.txt'}")
    return stats_df


def plot_pattern_cardinality_effects(df: pd.DataFrame, output_dir: Path):
    """5.4.3 Influence of Pattern Cardinality (Size and Number)"""
    if df.empty:
        return

    # Effect of Pattern Size
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # ---- Pattern Size Effect ----
    size_stats = []
    for size in sorted(df['pattern_size'].unique()):
        size_df = df[df['pattern_size'] == size]

        # Cost metrics
        cost_wins = (size_df['cost_winner'] == 'mdp').sum()
        cost_total = len(size_df[size_df['cost_winner'] != 'tie'])
        cost_win_rate = cost_wins / cost_total * 100 if cost_total > 0 else 0

        both_solved = size_df[size_df['pdb_solved'] & size_df['mdp_solved']]
        avg_cost_improvement = both_solved['cost_improvement'].mean() if not both_solved.empty else 0

        # Expanded states metrics
        exp_wins = (size_df['expanded_winner'] == 'mdp').sum()
        exp_total = len(size_df[size_df['expanded_winner'] != 'tie'])
        exp_win_rate = exp_wins / exp_total * 100 if exp_total > 0 else 0

        both_expanded = size_df[size_df['pdb_expanded_valid'] & size_df['mdp_expanded_valid']]
        avg_exp_improvement = both_expanded['expanded_improvement'].mean() if not both_expanded.empty else 0

        size_stats.append({
            'size': size,
            'cost_win_rate': cost_win_rate,
            'cost_improvement': avg_cost_improvement,
            'exp_win_rate': exp_win_rate,
            'exp_improvement': avg_exp_improvement,
            'instances': len(size_df)
        })

    size_df_stats = pd.DataFrame(size_stats)

    # Plot pattern size effects for cost
    ax1.plot(size_df_stats['size'], size_df_stats['cost_win_rate'], 'o-', color=COLORS['mdp'], linewidth=2,
             markersize=6)
    ax1.set_xlabel('Pattern Size')
    ax1.set_ylabel('MDP-H Win Rate (%)')
    ax1.set_title('Cost Performance vs Pattern Size')
    ax1.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    ax2.plot(size_df_stats['size'], size_df_stats['cost_improvement'], 'o-', color=COLORS['mdp'], linewidth=2,
             markersize=6)
    ax2.set_xlabel('Pattern Size')
    ax2.set_ylabel('Average Cost Improvement (%)')
    ax2.set_title('Average Cost Improvement vs Pattern Size')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # Plot pattern size effects for expanded states
    ax3.plot(size_df_stats['size'], size_df_stats['exp_win_rate'], 'o-', color=COLORS['mdp'], linewidth=2, markersize=6)
    ax3.set_xlabel('Pattern Size')
    ax3.set_ylabel('MDP-H Win Rate (%)')
    ax3.set_title('Expanded States Performance vs Pattern Size')
    ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    ax4.plot(size_df_stats['size'], size_df_stats['exp_improvement'], 'o-', color=COLORS['mdp'], linewidth=2,
             markersize=6)
    ax4.set_xlabel('Pattern Size')
    ax4.set_ylabel('Average Expanded States Improvement (%)')
    ax4.set_title('Expanded States Improvement vs Pattern Size')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)

    # Try to use tight_layout, fall back to subplots_adjust if it fails
    try:
        plt.tight_layout()
    except:
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.savefig(output_dir / 'pattern_size_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---- Pattern Amount Effect ----
    amount_stats = []
    for amount in sorted(df['pattern_amount'].unique()):
        amount_df = df[df['pattern_amount'] == amount]

        # Cost metrics
        cost_wins = (amount_df['cost_winner'] == 'mdp').sum()
        cost_total = len(amount_df[amount_df['cost_winner'] != 'tie'])
        cost_win_rate = cost_wins / cost_total * 100 if cost_total > 0 else 0

        both_solved = amount_df[amount_df['pdb_solved'] & amount_df['mdp_solved']]
        avg_cost_improvement = both_solved['cost_improvement'].mean() if not both_solved.empty else 0

        # Expanded states metrics
        exp_wins = (amount_df['expanded_winner'] == 'mdp').sum()
        exp_total = len(amount_df[amount_df['expanded_winner'] != 'tie'])
        exp_win_rate = exp_wins / exp_total * 100 if exp_total > 0 else 0

        both_expanded = amount_df[amount_df['pdb_expanded_valid'] & amount_df['mdp_expanded_valid']]
        avg_exp_improvement = both_expanded['expanded_improvement'].mean() if not both_expanded.empty else 0

        amount_stats.append({
            'amount': amount,
            'cost_win_rate': cost_win_rate,
            'cost_improvement': avg_cost_improvement,
            'exp_win_rate': exp_win_rate,
            'exp_improvement': avg_exp_improvement,
            'instances': len(amount_df)
        })

    amount_df_stats = pd.DataFrame(amount_stats)

    # Plot pattern amount effects
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Cost metrics
    ax1.plot(amount_df_stats['amount'], amount_df_stats['cost_win_rate'], 'o-', color=COLORS['mdp'], linewidth=2,
             markersize=6)
    ax1.set_xlabel('Number of Patterns')
    ax1.set_ylabel('MDP-H Win Rate (%)')
    ax1.set_title('Cost Performance vs Number of Patterns')
    ax1.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    ax2.plot(amount_df_stats['amount'], amount_df_stats['cost_improvement'], 'o-', color=COLORS['mdp'], linewidth=2,
             markersize=6)
    ax2.set_xlabel('Number of Patterns')
    ax2.set_ylabel('Average Cost Improvement (%)')
    ax2.set_title('Average Cost Improvement vs Number of Patterns')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # Expanded states metrics
    ax3.plot(amount_df_stats['amount'], amount_df_stats['exp_win_rate'], 'o-', color=COLORS['mdp'], linewidth=2,
             markersize=6)
    ax3.set_xlabel('Number of Patterns')
    ax3.set_ylabel('MDP-H Win Rate (%)')
    ax3.set_title('Expanded States Performance vs Number of Patterns')
    ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    ax4.plot(amount_df_stats['amount'], amount_df_stats['exp_improvement'], 'o-', color=COLORS['mdp'], linewidth=2,
             markersize=6)
    ax4.set_xlabel('Number of Patterns')
    ax4.set_ylabel('Average Expanded States Improvement (%)')
    ax4.set_title('Expanded States Improvement vs Number of Patterns')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)

    # Try to use tight_layout, fall back to subplots_adjust if it fails
    try:
        plt.tight_layout()
    except:
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.savefig(output_dir / 'pattern_amount_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create heatmaps showing interaction between size and amount (with improved handling)
    if (df['pattern_size'].nunique() > 1 and df['pattern_amount'].nunique() > 1):
        # First figure: Cost metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Cost Performance by Pattern Size and Amount', fontsize=16)

        # Win rate heatmap
        try:
            pivot_cost_wins = df.groupby(['pattern_size', 'pattern_amount'], ).apply(
                lambda x: (x['cost_winner'] == 'mdp').sum() / len(x[x['cost_winner'] != 'tie']) * 100
                if len(x[x['cost_winner'] != 'tie']) > 0 else 0,
            ).unstack(fill_value=np.nan)

            if not pivot_cost_wins.empty and pivot_cost_wins.size > 0 and not pivot_cost_wins.isna().all().all():
                sns.heatmap(pivot_cost_wins, annot=True, fmt='.1f', cmap='RdYlBu_r', center=50,
                            cbar_kws={'label': 'MDP-H Win Rate (%)'}, ax=ax1)
                ax1.set_title('MDP-H Win Rate (%)')
                ax1.set_xlabel('Number of Patterns')
                ax1.set_ylabel('Pattern Size')
            else:
                ax1.text(0.5, 0.5, "Insufficient data for heatmap", ha='center', va='center')
                ax1.set_title('Cost Win Rate - No Data')
        except Exception as e:
            ax1.text(0.5, 0.5, f"Error creating heatmap: {str(e)}", ha='center', va='center')
            ax1.set_title('Cost Win Rate - Error')

        # Average improvement heatmap
        try:
            improvement_data = df[df['pdb_solved'] & df['mdp_solved']]
            if not improvement_data.empty:
                pivot_cost_improvement = improvement_data.groupby(['pattern_size', 'pattern_amount'])[
                    'cost_improvement'].mean().unstack(fill_value=np.nan)

                if not pivot_cost_improvement.empty and pivot_cost_improvement.size > 0 and not pivot_cost_improvement.isna().all().all():
                    sns.heatmap(pivot_cost_improvement, annot=True, fmt='.1f', cmap='RdYlBu_r', center=0,
                                cbar_kws={'label': 'Avg Cost Improvement (%)'}, ax=ax2)
                    ax2.set_title('Average Cost Improvement (%)')
                    ax2.set_xlabel('Number of Patterns')
                    ax2.set_ylabel('Pattern Size')
                else:
                    ax2.text(0.5, 0.5, "Insufficient data for heatmap", ha='center', va='center')
                    ax2.set_title('Cost Improvement - No Data')
            else:
                ax2.text(0.5, 0.5, "No instances where both heuristics solved", ha='center', va='center')
                ax2.set_title('Cost Improvement - No Data')
        except Exception as e:
            ax2.text(0.5, 0.5, f"Error creating heatmap: {str(e)}", ha='center', va='center')
            ax2.set_title('Cost Improvement - Error')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(output_dir / 'pattern_cardinality_cost_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Second figure: Expanded States metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Expanded States Performance by Pattern Size and Amount', fontsize=16)

        # Win rate heatmap
        try:
            pivot_exp_wins = df.groupby(['pattern_size', 'pattern_amount'], ).apply(
                lambda x: (x['expanded_winner'] == 'mdp').sum() / len(x[x['expanded_winner'] != 'tie']) * 100
                if len(x[x['expanded_winner'] != 'tie']) > 0 else 0,
            ).unstack(fill_value=np.nan)

            if not pivot_exp_wins.empty and pivot_exp_wins.size > 0 and not pivot_exp_wins.isna().all().all():
                sns.heatmap(pivot_exp_wins, annot=True, fmt='.1f', cmap='RdYlBu_r', center=50,
                            cbar_kws={'label': 'MDP-H Win Rate (%)'}, ax=ax1)
                ax1.set_title('MDP-H Win Rate for Expanded States (%)')
                ax1.set_xlabel('Number of Patterns')
                ax1.set_ylabel('Pattern Size')
            else:
                ax1.text(0.5, 0.5, "Insufficient data for heatmap", ha='center', va='center')
                ax1.set_title('Expanded Win Rate - No Data')
        except Exception as e:
            ax1.text(0.5, 0.5, f"Error creating heatmap: {str(e)}", ha='center', va='center')
            ax1.set_title('Expanded Win Rate - Error')

        # Average improvement heatmap
        try:
            expansion_data = df[df['pdb_expanded_valid'] & df['mdp_expanded_valid']]
            if not expansion_data.empty:
                pivot_exp_improvement = expansion_data.groupby(['pattern_size', 'pattern_amount'])[
                    'expanded_improvement'].mean().unstack(fill_value=np.nan)

                if not pivot_exp_improvement.empty and pivot_exp_improvement.size > 0 and not pivot_exp_improvement.isna().all().all():
                    sns.heatmap(pivot_exp_improvement, annot=True, fmt='.1f', cmap='RdYlBu_r', center=0,
                                cbar_kws={'label': 'Avg Expanded States Improvement (%)'}, ax=ax2)
                    ax2.set_title('Average Expanded States Improvement (%)')
                    ax2.set_xlabel('Number of Patterns')
                    ax2.set_ylabel('Pattern Size')
                else:
                    ax2.text(0.5, 0.5, "Insufficient data for heatmap", ha='center', va='center')
                    ax2.set_title('Expanded Improvement - No Data')
            else:
                ax2.text(0.5, 0.5, "No instances with valid expanded states", ha='center', va='center')
                ax2.set_title('Expanded Improvement - No Data')
        except Exception as e:
            ax2.text(0.5, 0.5, f"Error creating heatmap: {str(e)}", ha='center', va='center')
            ax2.set_title('Expanded Improvement - Error')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(output_dir / 'pattern_cardinality_expanded_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Saved pattern cardinality heatmaps.")
    else:
        print("Skipping heatmaps - insufficient pattern size or amount variety.")

    # Save summary statistics for both cost and expanded states
    with open(output_dir / 'pattern_cardinality_summary.txt', 'w') as f:
        f.write("Pattern Cardinality Effects Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write("COST PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        f.write("\nEffect of Pattern Size on Cost:\n")
        f.write(size_df_stats[['size', 'cost_win_rate', 'cost_improvement', 'instances']].to_string(index=False,
                                                                                                    float_format='{:.2f}'.format))
        f.write("\n\nEffect of Number of Patterns on Cost:\n")
        f.write(amount_df_stats[['amount', 'cost_win_rate', 'cost_improvement', 'instances']].to_string(index=False,
                                                                                                        float_format='{:.2f}'.format))

        f.write("\n\nEXPANDED STATES PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        f.write("\nEffect of Pattern Size on Expanded States:\n")
        f.write(size_df_stats[['size', 'exp_win_rate', 'exp_improvement', 'instances']].to_string(index=False,
                                                                                                  float_format='{:.2f}'.format))
        f.write("\n\nEffect of Number of Patterns on Expanded States:\n")
        f.write(amount_df_stats[['amount', 'exp_win_rate', 'exp_improvement', 'instances']].to_string(index=False,
                                                                                                      float_format='{:.2f}'.format))

    print("Saved pattern cardinality effects analysis.")

    # Save summary statistics
    with open(output_dir / 'pattern_cardinality_summary.txt', 'w') as f:
        f.write("Pattern Cardinality Effects Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write("Effect of Pattern Size:\n")
        f.write(size_df_stats.to_string(index=False, float_format='{:.2f}'.format))
        f.write("\n\nEffect of Number of Patterns:\n")
        f.write(amount_df_stats.to_string(index=False, float_format='{:.2f}'.format))

    print("Saved pattern cardinality effects analysis.")


def plot_domain_comparison_overview(df: pd.DataFrame, output_dir: Path):
    """Creates an overview comparing performance across all domains."""
    if df.empty:
        return

    # Calculate domain-specific statistics
    domain_stats = []
    for domain in sorted(df['domain'].unique()):
        domain_df = df[df['domain'] == domain]

        # Win rates
        cost_wins = (domain_df['cost_winner'] == 'mdp').sum()
        cost_total = len(domain_df[domain_df['cost_winner'] != 'tie'])
        cost_win_rate = cost_wins / cost_total * 100 if cost_total > 0 else 0

        exp_wins = (domain_df['expanded_winner'] == 'mdp').sum()
        exp_total = len(domain_df[domain_df['expanded_winner'] != 'tie'])
        exp_win_rate = exp_wins / exp_total * 100 if exp_total > 0 else 0

        # Average improvements
        both_solved = domain_df[domain_df['pdb_solved'] & domain_df['mdp_solved']]
        avg_cost_improvement = both_solved['cost_improvement'].mean() if not both_solved.empty else 0

        both_expanded = domain_df[domain_df['pdb_expanded_valid'] & domain_df['mdp_expanded_valid']]
        avg_exp_improvement = both_expanded['expanded_improvement'].mean() if not both_expanded.empty else 0

        # Solve rates
        mdp_solve_rate = domain_df['mdp_solved'].mean() * 100
        pdb_solve_rate = domain_df['pdb_solved'].mean() * 100

        domain_stats.append({
            'Domain': domain,
            'Cost Win Rate (%)': cost_win_rate,
            'Expansion Win Rate (%)': exp_win_rate,
            'Avg Cost Improvement (%)': avg_cost_improvement,
            'Avg Expansion Improvement (%)': avg_exp_improvement,
            'MDP-H Solve Rate (%)': mdp_solve_rate,
            'PDB-H Solve Rate (%)': pdb_solve_rate,
            'Total Instances': len(domain_df)
        })

    stats_df = pd.DataFrame(domain_stats)

    # Create overview visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Sort by cost win rate for better visualization
    stats_df_sorted = stats_df.sort_values('Cost Win Rate (%)')

    # Cost win rates by domain
    bars1 = ax1.barh(stats_df_sorted['Domain'], stats_df_sorted['Cost Win Rate (%)'],
                     color=[COLORS['mdp'] if x >= 50 else COLORS['pdb'] for x in stats_df_sorted['Cost Win Rate (%)']],
                     alpha=0.7)
    ax1.set_xlabel('MDP-H Win Rate (%)')
    ax1.set_title('Cost Performance by Domain')
    ax1.axvline(x=50, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlim(0, 100)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars1, stats_df_sorted['Cost Win Rate (%)'])):
        ax1.text(value + 1, bar.get_y() + bar.get_height() / 2, f'{value:.1f}%',
                 va='center', ha='left' if value < 90 else 'right')

    # Average cost improvement by domain
    bars2 = ax2.barh(stats_df_sorted['Domain'], stats_df_sorted['Avg Cost Improvement (%)'],
                     color=[COLORS['mdp'] if x >= 0 else COLORS['pdb'] for x in
                            stats_df_sorted['Avg Cost Improvement (%)']],
                     alpha=0.7)
    ax2.set_xlabel('Average Cost Improvement (%)')
    ax2.set_title('Average Cost Improvement by Domain')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars2, stats_df_sorted['Avg Cost Improvement (%)'])):
        ax2.text(value + (0.5 if value >= 0 else -0.5), bar.get_y() + bar.get_height() / 2, f'{value:.1f}%',
                 va='center', ha='left' if value >= 0 else 'right')

    # Solve rates comparison
    x_pos = range(len(stats_df_sorted))
    width = 0.35

    ax3.bar([x - width / 2 for x in x_pos], stats_df_sorted['MDP-H Solve Rate (%)'],
            width, label='MDP-H', color=COLORS['mdp'], alpha=0.7)
    ax3.bar([x + width / 2 for x in x_pos], stats_df_sorted['PDB-H Solve Rate (%)'],
            width, label='PDB-H', color=COLORS['pdb'], alpha=0.7)

    ax3.set_xlabel('Domain')
    ax3.set_ylabel('Solve Rate (%)')
    ax3.set_title('Solve Rate Comparison by Domain')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(stats_df_sorted['Domain'], rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim(0, 100)

    # Instance counts by domain
    bars4 = ax4.bar(stats_df_sorted['Domain'], stats_df_sorted['Total Instances'],
                    color=COLORS['tie'], alpha=0.7)
    ax4.set_xlabel('Domain')
    ax4.set_ylabel('Number of Instances')
    ax4.set_title('Instance Count by Domain')
    ax4.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'domain_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed domain statistics table
    with open(output_dir / 'domain_comparison_table.txt', 'w') as f:
        f.write("Domain-Specific Performance Comparison\n")
        f.write("=" * 60 + "\n\n")

        # Sort by cost win rate for the table
        table_df = stats_df.sort_values('Cost Win Rate (%)', ascending=False)
        f.write("Domains ranked by MDP-H cost performance:\n\n")

        for _, row in table_df.iterrows():
            f.write(f"DOMAIN: {row['Domain']}\n")
            f.write(f"  Cost Win Rate: {row['Cost Win Rate (%)']:.1f}% ({row['Total Instances']} instances)\n")
            f.write(f"  Avg Cost Improvement: {row['Avg Cost Improvement (%)']:+.2f}%\n")
            f.write(f"  MDP-H Solve Rate: {row['MDP-H Solve Rate (%)']:.1f}%\n")
            f.write(f"  PDB-H Solve Rate: {row['PDB-H Solve Rate (%)']:.1f}%\n")
            f.write(
                f"  Performance: {'MDP-H Favored' if row['Cost Win Rate (%)'] > 50 else 'PDB-H Favored' if row['Cost Win Rate (%)'] < 50 else 'Balanced'}\n\n")

    print("Saved domain comparison overview.")


def analyze_single_domain(df: pd.DataFrame, domain_name: str, output_dir: Path):
    """Performs detailed analysis for a single domain."""
    domain_df = df[df['domain'] == domain_name].copy()

    if domain_df.empty:
        print(f"No data for domain {domain_name}")
        return

    domain_dir = output_dir / f"domain_{domain_name}"
    domain_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Domain {domain_name}: {len(domain_df)} instances")

    # Overall performance for this domain (always try this)
    plot_overall_performance_summary(domain_df, domain_dir)

    # Pattern strategy impact (only if enough variety)
    if domain_df['pattern_type_clean'].nunique() > 1:
        plot_pattern_strategy_impact(domain_df, domain_dir)
    else:
        print(
            f"    Skipping pattern strategy analysis - only {domain_df['pattern_type_clean'].nunique()} pattern type(s)")

    # Pattern cardinality effects (only if enough variety and instances)
    if (domain_df['pattern_size'].nunique() > 1 or domain_df['pattern_amount'].nunique() > 1) and len(domain_df) >= 10:
        try:
            plot_pattern_cardinality_effects(domain_df, domain_dir)
        except Exception as e:
            print(f"    Skipping pattern cardinality analysis for {domain_name}: {str(e)}")
    else:
        print(f"    Skipping pattern cardinality analysis - insufficient data variety or instances")

    # Domain-specific summary (always create this)
    create_domain_summary(domain_df, domain_name, domain_dir)

    print(f"  Completed analysis for domain: {domain_name}")


def create_domain_summary(df: pd.DataFrame, domain_name: str, output_dir: Path):
    """Creates a detailed summary for a specific domain."""
    if df.empty:
        return

    # Calculate statistics
    total_configs = len(df)
    problems = df['problem'].nunique()

    # ---------- SOLUTION STATISTICS ----------
    mdp_solved = df['mdp_solved'].sum()
    pdb_solved = df['pdb_solved'].sum()
    both_solved = (df['pdb_solved'] & df['mdp_solved']).sum()

    # ---------- COST PERFORMANCE ----------
    cost_wins_mdp = (df['cost_winner'] == 'mdp').sum()
    cost_wins_pdb = (df['cost_winner'] == 'pdb').sum()
    cost_ties = (df['cost_winner'] == 'tie').sum()

    # ---------- EXPANDED STATES PERFORMANCE ----------
    exp_wins_mdp = (df['expanded_winner'] == 'mdp').sum()
    exp_wins_pdb = (df['expanded_winner'] == 'pdb').sum()
    exp_ties = (df['expanded_winner'] == 'tie').sum()

    # ---------- COST IMPROVEMENT STATISTICS ----------
    both_solved_df = df[df['pdb_solved'] & df['mdp_solved']]
    if not both_solved_df.empty:
        avg_cost_improvement = both_solved_df['cost_improvement'].mean()
        median_cost_improvement = both_solved_df['cost_improvement'].median()
        std_cost_improvement = both_solved_df['cost_improvement'].std()
        positive_cost_improvements = (both_solved_df['cost_improvement'] > 0).sum()
    else:
        avg_cost_improvement = median_cost_improvement = std_cost_improvement = positive_cost_improvements = 0

    # ---------- EXPANDED STATES IMPROVEMENT STATISTICS ----------
    both_expanded_df = df[df['pdb_expanded_valid'] & df['mdp_expanded_valid']]
    if not both_expanded_df.empty:
        avg_exp_improvement = both_expanded_df['expanded_improvement'].mean()
        median_exp_improvement = both_expanded_df['expanded_improvement'].median()
        std_exp_improvement = both_expanded_df['expanded_improvement'].std()
        positive_exp_improvements = (both_expanded_df['expanded_improvement'] > 0).sum()
    else:
        avg_exp_improvement = median_exp_improvement = std_exp_improvement = positive_exp_improvements = 0

    # ---------- BEST/WORST CONFIGURATIONS ----------
    # Cost best/worst
    if not both_solved_df.empty:
        best_cost_config = both_solved_df.loc[both_solved_df['cost_improvement'].idxmax()]
        worst_cost_config = both_solved_df.loc[both_solved_df['cost_improvement'].idxmin()]

    # Expanded states best/worst
    if not both_expanded_df.empty:
        best_exp_config = both_expanded_df.loc[both_expanded_df['expanded_improvement'].idxmax()]
        worst_exp_config = both_expanded_df.loc[both_expanded_df['expanded_improvement'].idxmin()]

    with open(output_dir / f'{domain_name}_detailed_summary.txt', 'w') as f:
        f.write(f"DOMAIN: {domain_name.upper()}\n")
        f.write("=" * 60 + "\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total configurations: {total_configs}\n")
        f.write(f"Problems in domain: {problems}\n")
        f.write(f"MDP-H solve rate: {mdp_solved / total_configs * 100:.1f}% ({mdp_solved}/{total_configs})\n")
        f.write(f"PDB-H solve rate: {pdb_solved / total_configs * 100:.1f}% ({pdb_solved}/{total_configs})\n")
        f.write(f"Both solved: {both_solved / total_configs * 100:.1f}% ({both_solved}/{total_configs})\n\n")

        # ----- COST PERFORMANCE SECTION -----
        f.write("COST PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        total_cost_decided = cost_wins_mdp + cost_wins_pdb
        if total_cost_decided > 0:
            f.write(
                f"MDP-H wins: {cost_wins_mdp}/{total_cost_decided} ({cost_wins_mdp / total_cost_decided * 100:.1f}%)\n")
            f.write(
                f"PDB-H wins: {cost_wins_pdb}/{total_cost_decided} ({cost_wins_pdb / total_cost_decided * 100:.1f}%)\n")
        f.write(f"Ties: {cost_ties} ({cost_ties / total_configs * 100:.1f}%)\n")

        if both_solved > 0:
            f.write(f"Average cost improvement: {avg_cost_improvement:+.2f}%\n")
            f.write(f"Median cost improvement: {median_cost_improvement:+.2f}%\n")
            f.write(f"Standard deviation: {std_cost_improvement:.2f}%\n")
            f.write(
                f"Positive improvements: {positive_cost_improvements}/{both_solved} ({positive_cost_improvements / both_solved * 100:.1f}%)\n\n")

            if not both_solved_df.empty:
                f.write("Best cost improvement configuration:\n")
                f.write(f"  Improvement: {best_cost_config['cost_improvement']:+.2f}%\n")
                f.write(f"  Pattern: {best_cost_config['pattern_type_clean']}\n")
                f.write(f"  Size: {best_cost_config['pattern_size']}, Amount: {best_cost_config['pattern_amount']}\n")
                f.write(f"  Problem: {best_cost_config['problem']}\n\n")

                f.write("Worst cost improvement configuration:\n")
                f.write(f"  Improvement: {worst_cost_config['cost_improvement']:+.2f}%\n")
                f.write(f"  Pattern: {worst_cost_config['pattern_type_clean']}\n")
                f.write(f"  Size: {worst_cost_config['pattern_size']}, Amount: {worst_cost_config['pattern_amount']}\n")
                f.write(f"  Problem: {worst_cost_config['problem']}\n\n")

        # ----- EXPANDED STATES PERFORMANCE SECTION -----
        f.write("EXPANDED STATES PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        total_exp_decided = exp_wins_mdp + exp_wins_pdb
        if total_exp_decided > 0:
            f.write(f"MDP-H wins: {exp_wins_mdp}/{total_exp_decided} ({exp_wins_mdp / total_exp_decided * 100:.1f}%)\n")
            f.write(f"PDB-H wins: {exp_wins_pdb}/{total_exp_decided} ({exp_wins_pdb / total_exp_decided * 100:.1f}%)\n")
        f.write(f"Ties: {exp_ties} ({exp_ties / total_configs * 100:.1f}%)\n")

        if len(both_expanded_df) > 0:
            f.write(f"Average expanded states improvement: {avg_exp_improvement:+.2f}%\n")
            f.write(f"Median expanded states improvement: {median_exp_improvement:+.2f}%\n")
            f.write(f"Standard deviation: {std_exp_improvement:.2f}%\n")
            f.write(
                f"Positive improvements: {positive_exp_improvements}/{len(both_expanded_df)} ({positive_exp_improvements / len(both_expanded_df) * 100:.1f}%)\n\n")

            if not both_expanded_df.empty:
                f.write("Best expanded states improvement configuration:\n")
                f.write(f"  Improvement: {best_exp_config['expanded_improvement']:+.2f}%\n")
                f.write(f"  Pattern: {best_exp_config['pattern_type_clean']}\n")
                f.write(f"  Size: {best_exp_config['pattern_size']}, Amount: {best_exp_config['pattern_amount']}\n")
                f.write(f"  Problem: {best_exp_config['problem']}\n\n")

                f.write("Worst expanded states improvement configuration:\n")
                f.write(f"  Improvement: {worst_exp_config['expanded_improvement']:+.2f}%\n")
                f.write(f"  Pattern: {worst_exp_config['pattern_type_clean']}\n")
                f.write(f"  Size: {worst_exp_config['pattern_size']}, Amount: {worst_exp_config['pattern_amount']}\n")
                f.write(f"  Problem: {worst_exp_config['problem']}\n\n")

        # Per-problem breakdown if multiple problems
        if problems > 1:
            f.write("PER-PROBLEM BREAKDOWN\n")
            f.write("-" * 25 + "\n")
            for problem in sorted(df['problem'].unique()):
                prob_df = df[df['problem'] == problem]
                # Cost statistics
                cost_prob_wins = (prob_df['cost_winner'] == 'mdp').sum()
                cost_prob_total = len(prob_df[prob_df['cost_winner'] != 'tie'])
                # Expanded states statistics
                exp_prob_wins = (prob_df['expanded_winner'] == 'mdp').sum()
                exp_prob_total = len(prob_df[prob_df['expanded_winner'] != 'tie'])

                f.write(f"Problem: {problem}\n")
                if cost_prob_total > 0:
                    f.write(
                        f"  Cost Win Rate: {cost_prob_wins}/{cost_prob_total} ({cost_prob_wins / cost_prob_total * 100:.1f}%)\n")
                if exp_prob_total > 0:
                    f.write(
                        f"  Expanded Win Rate: {exp_prob_wins}/{exp_prob_total} ({exp_prob_wins / exp_prob_total * 100:.1f}%)\n")
                f.write("\n")

        # Domain conclusion
        f.write("\nCONCLUSION\n")
        f.write("-" * 15 + "\n")
        # Cost assessment
        if total_cost_decided > 0:
            cost_win_rate = cost_wins_mdp / total_cost_decided
            if cost_win_rate > 0.6:
                cost_conclusion = f"MDP-H significantly outperforms PDB-H in cost for {domain_name}"
            elif cost_win_rate > 0.4:
                cost_conclusion = f"MDP-H and PDB-H show competitive performance in cost for {domain_name}"
            else:
                cost_conclusion = f"PDB-H outperforms MDP-H in cost for {domain_name}"
        else:
            cost_conclusion = f"No decisive cost comparison available for {domain_name}"

        # Expanded states assessment
        if total_exp_decided > 0:
            exp_win_rate = exp_wins_mdp / total_exp_decided
            if exp_win_rate > 0.6:
                exp_conclusion = f"MDP-H significantly outperforms PDB-H in expanded states for {domain_name}"
            elif exp_win_rate > 0.4:
                exp_conclusion = f"MDP-H and PDB-H show competitive performance in expanded states for {domain_name}"
            else:
                exp_conclusion = f"PDB-H outperforms MDP-H in expanded states for {domain_name}"
        else:
            exp_conclusion = f"No decisive expanded states comparison available for {domain_name}"

        f.write(f"Cost: {cost_conclusion}\n")
        f.write(f"Search Efficiency: {exp_conclusion}\n\n")

        if both_solved > 0:
            f.write(f"Average cost improvement when both solve: {avg_cost_improvement:+.2f}%\n")
        if len(both_expanded_df) > 0:
            f.write(f"Average expanded states improvement when both valid: {avg_exp_improvement:+.2f}%\n")


def create_pattern_selection_tables(df: pd.DataFrame, output_dir: Path):
    if df.empty or 'pattern_type_clean' not in df.columns or df['pattern_type_clean'].nunique() == 0:
        print("Skipping focused pattern strategy table: No valid pattern types or data.")
        return

    summary_stats = []

    # Pre-calculate '# Domains Best For (Cost Win Rate)'
    best_strategy_counts_cost_wr = {}
    for domain_name_iter in sorted(df['domain'].unique()):
        domain_df_iter = df[df['domain'] == domain_name_iter]
        if domain_df_iter.empty: continue

        domain_pattern_perf = []
        for p_type_iter in sorted(domain_df_iter['pattern_type_clean'].unique()):
            p_type_domain_df = domain_df_iter[domain_df_iter['pattern_type_clean'] == p_type_iter]
            if p_type_domain_df.empty: continue

            cost_wins_mdp = (p_type_domain_df['cost_winner'] == 'mdp').sum()
            cost_decided = (p_type_domain_df['cost_winner'] != 'tie').sum()
            cost_win_rate_domain_ptype = cost_wins_mdp / cost_decided * 100 if cost_decided > 0 else 0.0
            domain_pattern_perf.append({'pattern_type_clean': p_type_iter, 'metric_value': cost_win_rate_domain_ptype})

        if domain_pattern_perf:
            best_for_domain_df = pd.DataFrame(domain_pattern_perf)
            if not best_for_domain_df.empty:
                max_win_rate_in_domain = best_for_domain_df['metric_value'].max()
                if max_win_rate_in_domain >= 0:  # Count even if win rate is 0 but it's the max (e.g. all are 0)
                    # or change to > 0 if only positive "best" counts
                    best_strategies_for_this_domain = best_for_domain_df[
                        best_for_domain_df['metric_value'] == max_win_rate_in_domain
                        ]['pattern_type_clean'].tolist()
                    for best_strat in best_strategies_for_this_domain:
                        best_strategy_counts_cost_wr[best_strat] = best_strategy_counts_cost_wr.get(best_strat, 0) + 1

    for p_type in sorted(df['pattern_type_clean'].unique()):
        if pd.isna(p_type): continue  # Should not happen if pattern_type_clean is well-managed

        pattern_df_iter = df[df['pattern_type_clean'] == p_type]
        if pattern_df_iter.empty: continue

        total_configs = len(pattern_df_iter)

        # Cost MDP Win %
        cost_wins_mdp = (pattern_df_iter['cost_winner'] == 'mdp').sum()
        cost_decided = (pattern_df_iter['cost_winner'] != 'tie').sum()  # Exclude ties from denominator
        overall_cost_win_rate = cost_wins_mdp / cost_decided * 100 if cost_decided > 0 else 0.0

        # Avg. Cost Improvement %
        both_solved_cost = pattern_df_iter[pattern_df_iter['pdb_solved'] & pattern_df_iter['mdp_solved']]
        avg_cost_imp = both_solved_cost['cost_improvement'].mean()
        if pd.isna(avg_cost_imp): avg_cost_imp = 0.0

        # Exp. MDP Win %
        exp_wins_mdp = (pattern_df_iter['expanded_winner'] == 'mdp').sum()
        exp_decided = (pattern_df_iter['expanded_winner'] != 'tie').sum()  # Exclude ties
        overall_exp_win_rate = exp_wins_mdp / exp_decided * 100 if exp_decided > 0 else 0.0

        # Avg. Exp. Improvement %
        both_valid_exp = pattern_df_iter[pattern_df_iter['pdb_expanded_valid'] & pattern_df_iter['mdp_expanded_valid']]
        avg_exp_imp = both_valid_exp['expanded_improvement'].mean()
        if pd.isna(avg_exp_imp): avg_exp_imp = 0.0

        # Net Solves (MDP - PDB)
        mdp_only_solved = (pattern_df_iter['mdp_solved'] & ~pattern_df_iter['pdb_solved']).sum()
        pdb_only_solved = (pattern_df_iter['pdb_solved'] & ~pattern_df_iter['mdp_solved']).sum()
        net_solves_mdp_vs_pdb = mdp_only_solved - pdb_only_solved

        summary_stats.append({
            'Pattern Strategy': str(p_type),
            'Total Configs': total_configs,
            'Cost MDP Win %': overall_cost_win_rate,
            'Avg Cost Imprv %': avg_cost_imp,
            'Exp MDP Win %': overall_exp_win_rate,
            'Avg Exp Imprv %': avg_exp_imp,
            'Net Solves (MDP-PDB)': net_solves_mdp_vs_pdb,
            '#Dom Best (Cost)': best_strategy_counts_cost_wr.get(p_type, 0)
        })

    if not summary_stats:
        print("No summary statistics generated for the focused pattern strategy table.")
        return

    summary_df = pd.DataFrame(summary_stats)
    # Ensure desired column order
    column_order = [
        'Pattern Strategy', 'Total Configs',
        'Cost MDP Win %', 'Avg Cost Imprv %',
        'Exp MDP Win %', 'Avg Exp Imprv %',
        'Net Solves (MDP-PDB)', '#Dom Best (Cost)'
    ]
    summary_df = summary_df[column_order]

    with open(output_dir / 'focused_pattern_strategy_comparison.txt', 'w') as f:
        f.write("FOCUSED COMPARISON OF PATTERN SELECTION STRATEGIES\n")
        # Adjust title length based on actual columns
        title_line_length = sum([len(col) + 3 for col in summary_df.columns]) - 3  # Rough estimate
        f.write("=" * min(120, title_line_length) + "\n")  # Cap at 120

        # Use a more compact to_string format or manually format
        # For better alignment, manual formatting or tabulate library might be needed for complex headers
        # For now, using pandas default with float_format
        f.write(summary_df.to_string(index=False, float_format='{:.1f}'.format))

        f.write("\n\nNotes:\n")
        f.write("- Win Rates (%) are for MDP-H vs PDB-H, calculated on non-tie cases.\n")
        f.write(
            "- Avg. Imprv. (%) is (PDB_Metric - MDP_Metric) / PDB_Metric * 100, for cases where both provided valid results.\n")
        f.write("- Net Solves (MDP-PDB): (MDP Only Solves) - (PDB Only Solves). Positive favors MDP-H coverage.\n")
        f.write("- #Dom Best (Cost): Number of domains where this strategy had the highest MDP Cost Win Rate.\n")

    print(
        f"Created focused pattern selection strategy comparison table: {output_dir / 'focused_pattern_strategy_comparison.txt'}")


def create_pattern_cardinality_table(df: pd.DataFrame, output_dir: Path):
    """Creates a comprehensive table showing the effects of pattern size and amount on performance."""
    if df.empty:
        return

    # Open the output file
    with open(output_dir / 'pattern_cardinality_table.txt', 'w') as f:
        # ===== PATTERN SIZE ANALYSIS =====
        f.write("EFFECT OF PATTERN SIZE ON PERFORMANCE\n")
        f.write("=" * 120 + "\n\n")

        # Table header for pattern size
        f.write(
            "{:<10} | {:>8} | {:>10} {:>10} | {:>9} {:>9} {:>9} | {:>10} {:>10} | {:>9} {:>9} {:>9} | {:>10} {:>10}\n".format(
                "Size", "Count",
                "MDP Solv%", "PDB Solv%",
                "MDP Wins", "PDB Wins", "Ties", "Win Rate%", "Avg Impr%",
                "MDP Wins", "PDB Wins", "Ties", "Win Rate%", "Avg Impr%"
            ))

        # Section headers for cost and expansion
        f.write("{:<10} | {:>8} | {:>10} {:>10} | {:>43} | {:>43}\n".format(
            "", "", "", "", "------------------ COST ------------------", "--------------- EXPANSIONS ---------------"
        ))

        f.write("-" * 120 + "\n")

        # Process each unique pattern size
        for size in sorted(df['pattern_size'].unique()):
            size_df = df[df['pattern_size'] == size]

            total_instances = len(size_df)

            # Calculate solve rates
            mdp_solve_rate = size_df['mdp_solved'].sum() / total_instances * 100
            pdb_solve_rate = size_df['pdb_solved'].sum() / total_instances * 100

            # Cost performance
            cost_wins_mdp = (size_df['cost_winner'] == 'mdp').sum()
            cost_wins_pdb = (size_df['cost_winner'] == 'pdb').sum()
            cost_ties = (size_df['cost_winner'] == 'tie').sum()

            # Win rate and improvement
            total_decided = cost_wins_mdp + cost_wins_pdb
            cost_win_rate = cost_wins_mdp / total_decided * 100 if total_decided > 0 else 0

            both_cost_df = size_df[size_df['pdb_solved'] & size_df['mdp_solved']]
            avg_cost_improv = both_cost_df['cost_improvement'].mean() if not both_cost_df.empty else 0

            # Expanded states
            exp_wins_mdp = (size_df['expanded_winner'] == 'mdp').sum()
            exp_wins_pdb = (size_df['expanded_winner'] == 'pdb').sum()
            exp_ties = (size_df['expanded_winner'] == 'tie').sum()

            exp_total_decided = exp_wins_mdp + exp_wins_pdb
            exp_win_rate = exp_wins_mdp / exp_total_decided * 100 if exp_total_decided > 0 else 0

            both_exp_df = size_df[size_df['pdb_expanded_valid'] & size_df['mdp_expanded_valid']]
            avg_exp_improv = both_exp_df['expanded_improvement'].mean() if not both_exp_df.empty else 0

            # Write size row
            f.write(
                "{:<10} | {:>8} | {:>10.2f} {:>10.2f} | {:>9} {:>9} {:>9} | {:>10.2f} {:>10.2f} | {:>9} {:>9} {:>9} | {:>10.2f} {:>10.2f}\n".format(
                    size, total_instances,
                    mdp_solve_rate, pdb_solve_rate,
                    cost_wins_mdp, cost_wins_pdb, cost_ties, cost_win_rate, avg_cost_improv,
                    exp_wins_mdp, exp_wins_pdb, exp_ties, exp_win_rate, avg_exp_improv
                ))

        # ===== PATTERN AMOUNT ANALYSIS =====
        f.write("\n\nEFFECT OF PATTERN AMOUNT ON PERFORMANCE\n")
        f.write("=" * 120 + "\n\n")

        # Table header for pattern amount
        f.write(
            "{:<10} | {:>8} | {:>10} {:>10} | {:>9} {:>9} {:>9} | {:>10} {:>10} | {:>9} {:>9} {:>9} | {:>10} {:>10}\n".format(
                "Amount", "Count",
                "MDP Solv%", "PDB Solv%",
                "MDP Wins", "PDB Wins", "Ties", "Win Rate%", "Avg Impr%",
                "MDP Wins", "PDB Wins", "Ties", "Win Rate%", "Avg Impr%"
            ))

        # Section headers for cost and expansion
        f.write("{:<10} | {:>8} | {:>10} {:>10} | {:>43} | {:>43}\n".format(
            "", "", "", "", "------------------ COST ------------------", "--------------- EXPANSIONS ---------------"
        ))

        f.write("-" * 120 + "\n")

        # Process each unique pattern amount
        for amount in sorted(df['pattern_amount'].unique()):
            amount_df = df[df['pattern_amount'] == amount]

            total_instances = len(amount_df)

            # Calculate solve rates
            mdp_solve_rate = amount_df['mdp_solved'].sum() / total_instances * 100
            pdb_solve_rate = amount_df['pdb_solved'].sum() / total_instances * 100

            # Cost performance
            cost_wins_mdp = (amount_df['cost_winner'] == 'mdp').sum()
            cost_wins_pdb = (amount_df['cost_winner'] == 'pdb').sum()
            cost_ties = (amount_df['cost_winner'] == 'tie').sum()

            # Win rate and improvement
            total_decided = cost_wins_mdp + cost_wins_pdb
            cost_win_rate = cost_wins_mdp / total_decided * 100 if total_decided > 0 else 0

            both_cost_df = amount_df[amount_df['pdb_solved'] & amount_df['mdp_solved']]
            avg_cost_improv = both_cost_df['cost_improvement'].mean() if not both_cost_df.empty else 0

            # Expanded states
            exp_wins_mdp = (amount_df['expanded_winner'] == 'mdp').sum()
            exp_wins_pdb = (amount_df['expanded_winner'] == 'pdb').sum()
            exp_ties = (amount_df['expanded_winner'] == 'tie').sum()

            exp_total_decided = exp_wins_mdp + exp_wins_pdb
            exp_win_rate = exp_wins_mdp / exp_total_decided * 100 if exp_total_decided > 0 else 0

            both_exp_df = amount_df[amount_df['pdb_expanded_valid'] & amount_df['mdp_expanded_valid']]
            avg_exp_improv = both_exp_df['expanded_improvement'].mean() if not both_exp_df.empty else 0

            # Write amount row
            f.write(
                "{:<10} | {:>8} | {:>10.2f} {:>10.2f} | {:>9} {:>9} {:>9} | {:>10.2f} {:>10.2f} | {:>9} {:>9} {:>9} | {:>10.2f} {:>10.2f}\n".format(
                    amount, total_instances,
                    mdp_solve_rate, pdb_solve_rate,
                    cost_wins_mdp, cost_wins_pdb, cost_ties, cost_win_rate, avg_cost_improv,
                    exp_wins_mdp, exp_wins_pdb, exp_ties, exp_win_rate, avg_exp_improv
                ))

        # ===== COMBINED SIZE × AMOUNT HEATMAP-LIKE TABLE =====
        f.write("\n\nCOMBINED EFFECT OF PATTERN SIZE AND AMOUNT (WIN RATES)\n")
        f.write("=" * 120 + "\n\n")

        # Get unique sizes and amounts
        sizes = sorted(df['pattern_size'].unique())
        amounts = sorted(df['pattern_amount'].unique())

        # Create header row with amounts
        f.write("{:<10} |".format("Size\\Amount"))
        for amount in amounts:
            f.write(" {:^14} |".format(amount))
        f.write("\n")
        f.write("-" * 120 + "\n")

        # For each size, create a row with win rates for each amount
        for size in sizes:
            f.write("{:<10} |".format(size))
            for amount in amounts:
                # Get data for this size-amount combination
                combo_df = df[(df['pattern_size'] == size) & (df['pattern_amount'] == amount)]

                if len(combo_df) > 0:
                    # Calculate win rates
                    cost_wins_mdp = (combo_df['cost_winner'] == 'mdp').sum()
                    cost_wins_pdb = (combo_df['cost_winner'] == 'pdb').sum()
                    total_decided = cost_wins_mdp + cost_wins_pdb
                    cost_win_rate = cost_wins_mdp / total_decided * 100 if total_decided > 0 else 0

                    exp_wins_mdp = (combo_df['expanded_winner'] == 'mdp').sum()
                    exp_wins_pdb = (combo_df['expanded_winner'] == 'pdb').sum()
                    exp_total_decided = exp_wins_mdp + exp_wins_pdb
                    exp_win_rate = exp_wins_mdp / exp_total_decided * 100 if exp_total_decided > 0 else 0

                    # Write the win rates in the cell
                    f.write(" C:{:5.1f}% E:{:5.1f}% |".format(cost_win_rate, exp_win_rate))
                else:
                    f.write(" {:^14} |".format("N/A"))
            f.write("\n")

        # ===== COMBINED SIZE × AMOUNT HEATMAP-LIKE TABLE FOR IMPROVEMENT =====
        f.write("\n\nCOMBINED EFFECT OF PATTERN SIZE AND AMOUNT (IMPROVEMENT %)\n")
        f.write("=" * 120 + "\n\n")

        # Create header row with amounts
        f.write("{:<10} |".format("Size\\Amount"))
        for amount in amounts:
            f.write(" {:^14} |".format(amount))
        f.write("\n")
        f.write("-" * 120 + "\n")

        # For each size, create a row with improvement percentages for each amount
        for size in sizes:
            f.write("{:<10} |".format(size))
            for amount in amounts:
                # Get data for this size-amount combination
                combo_df = df[(df['pattern_size'] == size) & (df['pattern_amount'] == amount)]

                if len(combo_df) > 0:
                    # Calculate average improvements
                    both_cost_df = combo_df[combo_df['pdb_solved'] & combo_df['mdp_solved']]
                    avg_cost_improv = both_cost_df['cost_improvement'].mean() if not both_cost_df.empty else 0

                    both_exp_df = combo_df[combo_df['pdb_expanded_valid'] & combo_df['mdp_expanded_valid']]
                    avg_exp_improv = both_exp_df['expanded_improvement'].mean() if not both_exp_df.empty else 0

                    # Write the improvement percentages in the cell
                    f.write(" C:{:+5.1f}% E:{:+5.1f}% |".format(avg_cost_improv, avg_exp_improv))
                else:
                    f.write(" {:^14} |".format("N/A"))
            f.write("\n")

        # Add a final section explaining the interpretation
        f.write("\n\nINTERPRETATION GUIDE:\n")
        f.write("-" * 60 + "\n")
        f.write("Win Rate%: Percentage of non-tie instances where MDP-H outperforms PDB-H\n")
        f.write("Avg Impr%: Average percentage improvement from MDP-H over PDB-H\n")
        f.write("  - For cost: Positive values mean MDP-H finds better (lower cost) solutions\n")
        f.write("  - For expansions: Positive values mean MDP-H expands fewer states\n")
        f.write("  - Negative values mean PDB-H performs better\n")
        f.write("\nIn the combined tables:\n")
        f.write("  - C: Cost performance\n")
        f.write("  - E: Expansion (search efficiency) performance\n")

    print(f"Created pattern cardinality table in {output_dir}/pattern_cardinality_table.txt")
    return

def create_comprehensive_summary(df: pd.DataFrame, output_dir: Path):
    """Creates a comprehensive text summary of all results."""
    if df.empty:
        return

    # Overall statistics
    total_configs = len(df)
    domains = df['domain'].nunique()
    problems = df['problem'].nunique()

    # Solution statistics
    mdp_solved = df['mdp_solved'].sum()
    pdb_solved = df['pdb_solved'].sum()
    both_solved = (df['pdb_solved'] & df['mdp_solved']).sum()

    # Winner statistics
    cost_wins_mdp = (df['cost_winner'] == 'mdp').sum()
    cost_wins_pdb = (df['cost_winner'] == 'pdb').sum()
    cost_ties = (df['cost_winner'] == 'tie').sum()

    # Improvement statistics
    both_solved_df = df[df['pdb_solved'] & df['mdp_solved']]
    if not both_solved_df.empty:
        avg_cost_improvement = both_solved_df['cost_improvement'].mean()
        median_cost_improvement = both_solved_df['cost_improvement'].median()
        positive_improvements = (both_solved_df['cost_improvement'] > 0).sum()
    else:
        avg_cost_improvement = median_cost_improvement = positive_improvements = 0

    with open(output_dir / 'comprehensive_summary.txt', 'w') as f:
        f.write("MDP-H vs PDB-H: Comprehensive Performance Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write("EXPERIMENTAL SETUP\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total configurations analyzed: {total_configs}\n")
        f.write(f"Domains: {domains}\n")
        f.write(f"Problems: {problems}\n")
        f.write(f"Pattern strategies: {', '.join(df['pattern_type_clean'].unique())}\n")
        f.write(f"Pattern sizes: {', '.join(map(str, sorted(df['pattern_size'].unique())))}\n")
        f.write(f"Pattern amounts: {', '.join(map(str, sorted(df['pattern_amount'].unique())))}\n\n")

        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        f.write(f"MDP-H solved instances: {mdp_solved} ({mdp_solved / total_configs * 100:.1f}%)\n")
        f.write(f"PDB-H solved instances: {pdb_solved} ({pdb_solved / total_configs * 100:.1f}%)\n")
        f.write(f"Both solved: {both_solved} ({both_solved / total_configs * 100:.1f}%)\n\n")

        f.write("COST PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        f.write(f"MDP-H wins: {cost_wins_mdp} ({cost_wins_mdp / total_configs * 100:.1f}%)\n")
        f.write(f"PDB-H wins: {cost_wins_pdb} ({cost_wins_pdb / total_configs * 100:.1f}%)\n")
        f.write(f"Ties: {cost_ties} ({cost_ties / total_configs * 100:.1f}%)\n")
        if both_solved > 0:
            f.write(f"Average cost improvement: {avg_cost_improvement:.2f}%\n")
            f.write(f"Median cost improvement: {median_cost_improvement:.2f}%\n")
            f.write(
                f"Instances with improvement: {positive_improvements}/{both_solved} ({positive_improvements / both_solved * 100:.1f}%)\n\n")

        # Domain-specific analysis
        f.write("DOMAIN-SPECIFIC PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        for domain in sorted(df['domain'].unique()):
            domain_df = df[df['domain'] == domain]
            domain_wins = (domain_df['cost_winner'] == 'mdp').sum()
            domain_total = len(domain_df[domain_df['cost_winner'] != 'tie'])
            if domain_total > 0:
                f.write(f"{domain}: {domain_wins}/{domain_total} wins ({domain_wins / domain_total * 100:.1f}%)\n")

    print("Saved comprehensive summary.")


# --- Main Analysis Function ---
def analyze_planning_results(json_filepath: str, output_dir: str):
    """Main function to perform complete analysis."""
    print(f"Starting analysis of {json_filepath}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    df = load_and_preprocess_data(json_filepath)
    if df is None or df.empty:
        print("No data available for analysis.")
        return

    # Generate overall analysis
    print("\nGenerating overall analysis...")
    df_cost = df[(df['mdp_cost'] < 90) & (df['pdb_cost'] < 90)]  # Filter out extreme values
    plot_overall_performance_summary(df_cost, output_path)
    plot_pattern_strategy_impact(df, output_path)
    plot_pattern_cardinality_effects(df, output_path)

    # Generate pattern selection tables
    print("\nGenerating pattern selection tables...")
    create_pattern_selection_tables(df, output_path)

    print("\nGenerating pattern cardinality table...")
    create_pattern_cardinality_table(df, output_path)

    # Generate domain comparison overview
    print("\nGenerating domain comparison overview...")
    plot_domain_comparison_overview(df, output_path)

    # Generate per-domain analysis
    print("\nGenerating per-domain analyses...")
    domains = sorted(df['domain'].unique())
    for domain in domains:
        print(f"  Analyzing domain: {domain}")
        analyze_single_domain(df, domain, output_path)

    # Create comprehensive summary
    create_comprehensive_summary(df, output_path)

    # Create domains overview table
    print("\nCreating domain overview table...")
    create_single_consolidated_table(df, output_path)

    print(f"\nAnalysis complete! Results saved to {output_path}")
    print(f"Generated analysis for {len(domains)} domains: {', '.join(domains)}")
    return df


# --- Main execution ---
if __name__ == '__main__':
    # Configuration
    input_file = 'comparison_FINAL.json'
    output_folder = 'thesis_analysis_results'

    # Run analysis
    if Path(input_file).exists():
        df = analyze_planning_results(input_file, output_folder)

        # Print quick summary
        if df is not None:
            print("\n" + "=" * 50)
            print("QUICK SUMMARY")
            print("=" * 50)
            print(f"Total instances: {len(df)}")
            print(f"MDP-H wins: {(df['cost_winner'] == 'mdp').sum()}")
            print(f"PDB-H wins: {(df['cost_winner'] == 'pdb').sum()}")
            print(f"Ties: {(df['cost_winner'] == 'tie').sum()}")

            both_solved = df[df['pdb_solved'] & df['mdp_solved']]
            if not both_solved.empty:
                avg_improvement = both_solved['cost_improvement'].mean()
                print(f"Average cost improvement: {avg_improvement:.2f}%")
    else:
        print(f"Input file '{input_file}' not found.")