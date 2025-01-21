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