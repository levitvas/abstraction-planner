import heapq
import math
import time
from collections import defaultdict

from parser.action import OperatorSas


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


def a_star(facts, init_state, actions, goal_state, heuristic, var_len, parser, gamma=None, projection=None):
    open_set = []

    g_score = dict()
    parent = dict()
    parent[tuple(init_state)] = None

    g_score[tuple(init_state)] = 0

    order = 0

    goal_strips = set()
    for num, val in goal_state.items():
        goal_strips.add((num, val))

    counter, preconditions_of, first_visit = initialize(facts, actions)

    if projection is not None:
        predicted_cost = heuristic(init_state, parser.end_state.variables, parser, gamma, projection)
        heapq.heappush(open_set, (
            predicted_cost,
            order, init_state))
    else:
        heapq.heappush(open_set, (
            heuristic(facts, fdr_to_strips(init_state), actions, goal_strips, var_len, preconditions_of),
            order, init_state))

    order += 1
    start = time.time()


    while open_set:
        # print(len(open_set))
        # elapsed = time.time()
        # if elapsed - start > 25:  # Limit to see in BRUTE
        #     return None

        score, count, current_state = heapq.heappop(open_set)
        if check_goal(goal_state, current_state):
            return reconstruct_path(tuple(current_state), parent)

        applicable_ops = generate_applicable_operators(current_state, counter, preconditions_of, actions)

        for action in applicable_ops:
            action: OperatorSas = actions[action]
            new_g_score = g_score[tuple(current_state)] + action.cost
            child_state, _sh_state = action.apply(current_state)
            child_state = child_state.variables

            if new_g_score < g_score.get(tuple(child_state), math.inf):
                parent[tuple(child_state)] = (current_state, action, action.cost)
                g_score[tuple(child_state)] = new_g_score

                # print("---")
                # print(current_state)
                # print()
                # print(action)
                # print()
                # print(child_state)

                if check_goal(goal_state, child_state):
                    heur = 0
                elif projection is not None:
                    heur = heuristic(child_state, parser.end_state.variables, parser, gamma,
                                               projection)
                else:
                    heur = heuristic(facts, fdr_to_strips(child_state), actions, goal_strips, var_len, preconditions_of)

                heapq.heappush(open_set, (
                    new_g_score + heur, order,
                    child_state))
                order += 1

    return None
