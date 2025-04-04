import random
import sys

from a_star_search import a_star, gbfs
from heuristics.abstract_heuristic import abstract_heuristic
from heuristics.hmax_heuristic import hmax
from heuristics.lmcut_heuristic import lm_cut
from heuristics.pdb_heuristic import pdb_heuristic
from sas_parser.parser import Parser
from utils.interesting_patterns import find_interesting_patterns


def solve_sas(parser: Parser, heuristics, tie_breaking, time_limit):
    bfs_sum = 0
    F = []
    g = set()
    for num, v in enumerate(parser.variables):
        for d in range(len(v.variables)):
            F.append((num, d))

    s_0 = set()
    for num, var in enumerate(parser.begin_state.variables):
        s_0.add((num, var))
    for var, val in parser.end_variables.items():
        g.add(F.index((var, val)))

    ret, expanded_states = gbfs(F, parser.begin_state.variables, parser.operators, parser.end_variables, heuristics,
                                len(parser.variables), tie_breaking, time_limit)
    if not ret:
        return -1, -1
    cost, path = ret[0], ret[1]
    # for (node, act) in path:
    #     print(act.name)
    # print(f"Plan cost: {cost}")
    # print(f"Expanded states: {expanded_states}")
    return cost, expanded_states


def create_plan(sas_file):
    # input_f = sys.argv[1]
    # heuristic = sys.argv[2]

    with open(sas_file) as f:
        lines = f.read().split('\n')
    parser = Parser(lines)

    heuristic = 'alternation'
    gamma = 0.9

    projections = [[3, 6, 8], [1, 4, 5], [2, 7, 8], [0, 1, 5]]  # Adding [2, 3, 5] makes it much worse
    # projections = [[0, 2, 4], [0, 4, 5], [0, 4, 6], [0, 4, 7], [0, 8, 4], [1, 4, 5], [2, 4, 5], [4, 5, 6], [4, 5, 7], [8, 4, 5], [4, 6, 7], [8, 4, 6]]
    goal_states = [pos for pos, variable in enumerate(parser.end_state.variables) if variable != -1]

    max_pattern_size = 3
    interesting_patterns = find_interesting_patterns([range(0, len(parser.variables), 1)], parser.operators,
                                                     goal_states, max_pattern_size)
    print(len(interesting_patterns))

    # print(parser.end_state.variables)
    # print(goal_states)
    # print([list(pat) for pat in interesting_patterns])
    # get 8 random patterns with max size from interesting patterns
    projections = [list(pat) for pat in interesting_patterns if len(pat) == max_pattern_size]
    # projections = random.sample(projections, 8)
    projections = [[1, 2, 3]]
    print(projections)
    # exit(0)
    # projections = [[0], [1]]
    # projections = [[8, 9, 3], [8, 3, 4], [2, 10, 4]]
    projections = [[1, 6], [3, 5], [2, 10], [8, 2], [2, 7], [1, 4], [2, 6], [10, 3], [3, 7], [8, 3], [9, 2], [3, 4],
                   [9, 3], [0, 6], [3, 6], [2, 5], [0, 4], [2, 4], [1, 5], [0, 5]]
    # projections = [[3, 4], [0, 4], [9, 2]]

    if heuristic == 'hmax':
        predicted_cost = solve_sas(parser, [("hmax", hmax)], "average", 120)
    elif heuristic == 'abs':
        predicted_cost = solve_sas(parser, [("abs", abstract_heuristic(parser.begin_state.variables,
                                                                       parser.end_state.variables, parser, gamma,
                                                                       projections[0]))])
    elif heuristic == "alternation":
        heuristics = [
            ("abs", abstract_heuristic(parser.begin_state.variables, parser.end_state.variables, parser, gamma, x)) for
            x in projections]
        predicted_cost = solve_sas(parser, heuristics, "average", 120)
        print(predicted_cost[0], predicted_cost[1])
    elif heuristic == "pdb":
        heuristics = [
            ("abs", pdb_heuristic(parser.begin_state.variables, parser.end_state.variables, parser, gamma, x)) for
            x in projections]
        predicted_cost = solve_sas(parser, heuristics)
    else:
        predicted_cost = solve_sas(parser, lm_cut)


if __name__ == '__main__':
    create_plan("problems/elevator/elevator-11.sas")
