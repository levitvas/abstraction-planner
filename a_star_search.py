import heapq
import math
import random
import time
from collections import defaultdict

from sas_parser.action import OperatorSas


def initialize(facts, operators):
    counter = [[] for _ in range(len(operators))]
    preconditions_of = {fact: [] for fact in facts}
    first_visit = [True] * len(operators)

    for i, op in enumerate(operators):
        for num, fact in op.preconditions.items():
            preconditions_of[(num, fact)].append(i)

    return counter, preconditions_of, first_visit


def generate_applicable_operators(state, counter, preconditions_of, actions):
    applicable_ops = []
    first_visit = [True] * len(actions)
    for num, fact in enumerate(state):
        for op_idx in preconditions_of[(num, fact)]:
            if first_visit[op_idx]:
                counter[op_idx] = len(actions[op_idx].preconditions)
                first_visit[op_idx] = False
            counter[op_idx] -= 1

    for a in range(len(actions)):
        if counter[a] == 0 and not first_visit[a]:
            applicable_ops.append(a)

    return applicable_ops


def is_goal(state, goal):
    return goal.issubset(state)


def check_goal(end_state, current_state):
    for key, val in end_state.items():
        if current_state[key] != val:
            return False
    return True


def reconstruct_path(node, parents):
    path = []
    cost = 0
    nodes = set()
    nodes.add(tuple(node))
    while parents[tuple(node)] is not None:
        path.append((parents[node][0], parents[node][1]))
        cost += parents[node][2]
        node = tuple(parents[node][0])
        nodes.add(node)
    return cost, path[::-1]


def fdr_to_strips(state):
    st = set()
    for num, val in enumerate(state):
        st.add((num, val))
    return st


def a_star(facts, init_state, actions, goal_state, heuristics, var_len, parser):
    print("Doing A*")
    g_score = dict()
    parent = dict()
    parent[tuple(init_state)] = None
    g_score[tuple(init_state)] = 0
    order = 0
    expanded_states = 0

    goal_strips = set()
    for num, val in goal_state.items():
        goal_strips.add((num, val))

    counter, preconditions_of, first_visit = initialize(facts, actions)

    open_sets = [[] for _ in range(len(heuristics))]
    current_heuristic = 0
    for idx, (name, heuristic) in enumerate(heuristics):
        if name == "abs":
            predicted_cost = heuristic(init_state) * -1
            heapq.heappush(open_sets[idx], (predicted_cost, order, init_state))
        else:
            heapq.heappush(open_sets[idx], (
                heuristic(facts, fdr_to_strips(init_state), actions, goal_strips, var_len, preconditions_of),
                order, init_state))


    order += 1
    closed_set = set()
    closed_set.add(tuple(init_state))
    while any(open_set for open_set in open_sets):

        while not open_sets[current_heuristic]:
            current_heuristic = (current_heuristic + 1) % len(open_sets)
            if all(not open_set for open_set in open_sets):
                return None

        score, count, current_state = heapq.heappop(open_sets[current_heuristic])

        if check_goal(goal_state, current_state):
            return reconstruct_path(tuple(current_state), parent), expanded_states

        applicable_ops = generate_applicable_operators(current_state, counter, preconditions_of, actions)

        # TODO: just add one or check if existing state
        if not tuple(current_state) in closed_set:
            closed_set.add(tuple(current_state))
            expanded_states += 1

        for action in applicable_ops:
            action: OperatorSas = actions[action]
            new_g_score = g_score[tuple(current_state)] + action.cost
            child_state, _sh_state = action.apply(current_state)
            child_state = child_state.variables

            if new_g_score < g_score.get(tuple(child_state), math.inf):
                parent[tuple(child_state)] = (current_state, action, action.cost)
                g_score[tuple(child_state)] = new_g_score

                # if check_goal(goal_state, child_state):
                #     heur = 0
                for idx, (name, heuristic) in enumerate(heuristics):
                    if name == "abs":
                        predicted_cost = heuristic(child_state) * -1
                        heapq.heappush(open_sets[idx], (predicted_cost, order, child_state))
                    else:
                        heapq.heappush(open_sets[idx], (
                            heuristic(facts, fdr_to_strips(child_state), actions, goal_strips, var_len,
                                      preconditions_of),
                            order, child_state))
                order += 1

        current_heuristic = (current_heuristic + 1) % len(open_sets)
    return None


def gbfs(facts, init_state, actions, goal_state, heuristics, var_len, tie_breaking, time_limit):
    # print("Doing GBFS")
    time_start = time.time()
    parent = dict()
    parent[tuple(init_state)] = None
    order = 0
    expanded_states = 0

    goal_strips = set()
    for num, val in goal_state.items():
        goal_strips.add((num, val))

    counter, preconditions_of, first_visit = initialize(facts, actions)

    open_sets = [[] for _ in range(len(heuristics))]
    current_heuristic = 0

    for idx, (name, heuristic) in enumerate(heuristics):
        if name == "abs":
            predicted_cost = heuristic(init_state)
            heapq.heappush(open_sets[idx], (predicted_cost, order, init_state))
        else:
            heapq.heappush(open_sets[idx], (
                heuristic(facts, fdr_to_strips(init_state), actions, goal_strips, var_len, preconditions_of),
                order, init_state))

    # get random from 0 to 1, to get randomized tie-breaking
    order += 1
    order = random.random()
    closed_set = set()

    # Create separate closed sets for each heuristic
    closed_sets = [set() for _ in range(len(heuristics))]
    best_h_values = [{} for _ in range(len(heuristics))]

    path_costs = {tuple(init_state): 0}

    revisited_states = 0
    # print("start")

    while any(open_set for open_set in open_sets):
        if time.time() - time_start > time_limit:
            print(f"- Time limit reached {expanded_states}")
            return (-1, None), -1
        # print(f"left!! {[len(x) for x in open_sets]}")

        while not open_sets[current_heuristic]:
            current_heuristic = (current_heuristic + 1) % len(open_sets)
            if all(not open_set for open_set in open_sets):
                return None

        h_score, count, current_state = heapq.heappop(open_sets[current_heuristic])

        if check_goal(goal_state, current_state):
            # print(f"popped!! {[len(x) for x in open_sets]}")
            # print(f"Revisted states: {revisited_states}")
            return reconstruct_path(tuple(current_state), parent), expanded_states

        applicable_ops = generate_applicable_operators(current_state, counter, preconditions_of, actions)

        # if tuple(current_state) in closed_set:
        if tuple(current_state) in closed_sets[current_heuristic]:
            # print(f"popped!! {[len(x) for x in open_sets]}")
            revisited_states += 1
            current_heuristic = (current_heuristic + 1) % len(open_sets)
            continue

        if tuple(current_state) not in closed_sets[current_heuristic]:
        # if tuple(current_state) not in closed_set:
        #     closed_set.add(tuple(current_state))
            closed_sets[current_heuristic].add(tuple(current_state))
            # print("popped")
            expanded_states += 1



        for action in applicable_ops:
            if time.time() - time_start > time_limit:
                print(f"+ Time limit reached {expanded_states}")
                return (-1, None), -1
            action: OperatorSas = actions[action]
            child_state, _sh_state = action.apply(current_state)
            child_state = child_state.variables

            child_tuple = tuple(child_state)
            new_cost = path_costs[tuple(current_state)] + action.cost
            if tuple(child_state) not in closed_sets[current_heuristic]:
            # if child_tuple not in closed_set:
                if child_tuple not in path_costs or new_cost < path_costs[child_tuple]:
                    path_costs[child_tuple] = new_cost
                    parent[child_tuple] = (current_state, action, action.cost)

                    # Random tie-breaking
                    order = random.random()

                    # Take average of other heuristics
                    heuristic_values = []
                    heuristic_sum = 0
                    for idx, (name, heuristic) in enumerate(heuristics):
                        if name == "abs":
                            predicted_cost = heuristic(child_state)
                            heuristic_sum += predicted_cost
                            heuristic_values.append((idx, predicted_cost))
                            if tie_breaking == 'random':
                                if child_tuple not in best_h_values[idx] or predicted_cost < best_h_values[idx][child_tuple]:
                                    best_h_values[idx][child_tuple] = predicted_cost
                                    heapq.heappush(open_sets[idx], (predicted_cost, order, child_state))
                        else:
                            heapq.heappush(open_sets[idx], (
                                heuristic(facts, fdr_to_strips(child_state), actions, goal_strips, var_len,
                                          preconditions_of),
                                order, child_state))

                    if tie_breaking == 'average':
                        for idx, value in heuristic_values:
                            if child_tuple not in best_h_values[idx] or value < best_h_values[idx][child_tuple]:
                                best_h_values[idx][child_tuple] = value
                                heapq.heappush(open_sets[idx], (value, heuristic_sum / len(heuristic_values), child_state))
        current_heuristic = (current_heuristic + 1) % len(open_sets)

    return None
