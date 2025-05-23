import networkx as nx
from collections import deque
from itertools import combinations


def build_causal_graph(variables, actions):
    graph = nx.DiGraph()  # Directed graph

    # Add all variables as nodes
    for v in variables:
        graph.add_node(v)

    # Add edges based on Definition 5.3.1
    for action in actions:
        pre_vars = action.preconditions.keys()
        eff_vars = action.effects.keys()

        # Rule 1: edges from precondition to effect variables
        for u in pre_vars:
            for v in eff_vars:
                if u != v:
                    graph.add_edge(u, v)

        # Rule 2: edges between effect variables
        for u, v in combinations(eff_vars, 2):
            graph.add_edge(u, v)
            graph.add_edge(v, u)  # Both directions as they form a clique

    return graph


def find_causally_relevant_variables(graph, goal_vars):
    relevant = set()

    # Start from goal variables and traverse backwards
    queue = deque(goal_vars)
    while queue:
        v = queue.popleft()
        if v not in relevant:
            relevant.add(v)
            # Add all predecessors (using incoming edges)
            for u in graph.predecessors(v):
                queue.append(u)

    return relevant


def is_weakly_connected(graph, variables):
    if not variables:
        return True

    # Start BFS from any variable
    start = next(iter(variables))
    visited = {start}
    queue = deque([start])

    # Convert directed graph to undirected for weak connectivity
    while queue:
        v = queue.popleft()
        # Check both outgoing and incoming edges
        neighbors = set(graph.successors(v)) | set(graph.predecessors(v))
        for u in neighbors:
            if u in variables and u not in visited:
                visited.add(u)
                queue.append(u)

    # Check if all variables were reached
    return len(visited) == len(variables)


def is_interesting_pattern(pattern, graph, goal_vars):
    # Condition 1: Check weak connectivity
    if not is_weakly_connected(graph, pattern):
        return False

    # Get projected task's causal graph
    projected_graph = graph.subgraph(pattern)

    # Condition 2: Check causal relevance
    projected_goal_vars = {v for v in pattern if v in goal_vars}
    if not projected_goal_vars:
        return False

    causally_relevant = find_causally_relevant_variables(projected_graph, projected_goal_vars)
    return pattern.issubset(causally_relevant)


def find_interesting_patterns(variables, actions, goal_vars, max_size):
    causal_graph = build_causal_graph(variables, actions)
    patterns = []

    # Start with singleton patterns containing goal variables
    candidates = [{v} for v in goal_vars]

    while candidates:
        pattern = candidates.pop(0)

        if is_interesting_pattern(pattern, causal_graph, goal_vars):
            patterns.append(pattern)

            # Generate larger patterns if size limit not reached
            if len(pattern) < max_size:
                # Find potential variables to add
                neighbors = set()
                for v in pattern:
                    neighbors.update(causal_graph.successors(v))
                    neighbors.update(causal_graph.predecessors(v))

                # Create new candidate patterns
                for v in neighbors - pattern:
                    new_pattern = pattern | {v}
                    if new_pattern not in candidates:
                        candidates.append(new_pattern)

    return patterns

def get_grown_patterns(previous_MDP_patterns, pattern_size, pattern_amount, sorted_size_two_patterns):
    # If size is 2, then solve all
    # Otherwise grow the pattern from the previous ones
    # Here we have two methods, either use the conjunction from previous
    # Or add a variable, and then check if its an interesting pattern
    # and then the same way solve from all results IP's and pick the best

    # Convert to sets
    new_interesting_patterns = set()
    previous_patterns_set = [set(pattern) for pattern in previous_MDP_patterns]

    print(f"Growing patterns from size {pattern_size - 1} to {pattern_size}")
    # -- First method
    # Conjunction of previous patterns
    for i, one_set in enumerate(previous_patterns_set):
        for second_set in previous_patterns_set[i + 1:]:
            # Check if second_set has any elements in common with one_set
            if one_set != second_set and len(one_set.intersection(second_set)) == pattern_size - 2:
                # Create a new set with the union of both sets
                new_set = one_set.union(second_set)
                # Check if the new set is interesting
                new_interesting_patterns.add(tuple(new_set))

    # If not enough patterns, then add some random ones, from interesting patterns, that have at
    # least one variable in common with the previous ones
    # FORFEIT prev comment, instead try to grow from initial sorted 2 size patterns
    if len(new_interesting_patterns) < pattern_amount[-1]:
        print("Not enough patterns, joining with 2 size patterns")
        # Get the intersection of the previous patterns and sorted size two patterns
        for one_set in previous_patterns_set:
            for second_set in [set(ptrn) for ptrn in sorted_size_two_patterns]:
                # Check if second_set has any elements in common with one_set
                if one_set != second_set and len(one_set.intersection(second_set)) == 1:
                    # Create a new set with the union of both sets
                    new_set = one_set.union(second_set)
                    # Check if the new set is interesting
                    new_interesting_patterns.add(tuple(new_set))

                    # Check if enough patterns
                    if len(new_interesting_patterns) >= pattern_amount[-1]:
                        break
            if len(new_interesting_patterns) >= pattern_amount[-1]:
                print("Enough patterns found, breaking")
                print(new_interesting_patterns)
                break

    all_patterns = [list(ptrn) for ptrn in new_interesting_patterns]
    return all_patterns


def select_best_patterns_with_goal_coverage_optimized(pattern_heuristic_pairs, num_patterns, goal_states):
    if not pattern_heuristic_pairs:
        return []

    goal_set = set(goal_states)
    all_pattern_sets = [set(p_tuple[0]) for p_tuple in pattern_heuristic_pairs]

    selected_pairs = list(pattern_heuristic_pairs[:num_patterns])
    if not selected_pairs:
        return []

    covered_goals = set()
    # Calculate initial coverage based on the first num_patterns
    # Assumes selected_pairs[i] corresponds to pattern_heuristic_pairs[i] initially
    for i in range(len(selected_pairs)):
        pattern_index_in_all = pattern_heuristic_pairs.index(selected_pairs[i])
        covered_goals.update(all_pattern_sets[pattern_index_in_all].intersection(goal_set))

    uncovered_goals = goal_set - covered_goals

    if uncovered_goals:
        for i in range(len(selected_pairs) - 1, -1, -1):  # Iterate backwards through selected_pairs
            if not uncovered_goals:
                break

            # Try to find a replacement from the remaining patterns
            for j in range(num_patterns, len(pattern_heuristic_pairs)):
                candidate_original_tuple = pattern_heuristic_pairs[j]
                candidate_pattern_set = all_pattern_sets[j]

                newly_covered_by_candidate = candidate_pattern_set.intersection(uncovered_goals)

                if newly_covered_by_candidate:
                    selected_pairs[i] = candidate_original_tuple
                    uncovered_goals -= newly_covered_by_candidate
                    break  # Move to the next pattern to potentially replace in selected_pairs

    return selected_pairs
