import cProfile
import pstats
import random
import sys

from a_star_search import a_star, gbfs
from heuristics.abstract_heuristic import abstract_heuristic
from heuristics.hmax_heuristic import hmax
from heuristics.lmcut_heuristic import lm_cut
from heuristics.hff_heuristic import hff
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
    return cost, expanded_states


def create_plan(sas_file):
    with open(sas_file) as f:
        lines = f.read().split('\n')
    parser = Parser(lines)

    heuristic = 'hff'
    gamma = 0.9

    goal_states = [pos for pos, variable in enumerate(parser.end_state.variables) if variable != -1]

    max_pattern_size = 3
    interesting_patterns = find_interesting_patterns([range(0, len(parser.variables), 1)], parser.operators,
                                                     goal_states, max_pattern_size)
    print(len(interesting_patterns))

    projections = [list(pat) for pat in interesting_patterns if len(pat) == max_pattern_size]
    projections = [[11,4,13],[13,11,5],[11,3,13],[11,4,5],[5,3,13]]

    if heuristic == 'hmax':
        predicted_cost = solve_sas(parser, [("hmax", hmax)], "average", 120)
        print(predicted_cost)
    elif heuristic == 'hff':
        predicted_cost = solve_sas(parser, [("hff", hff)], "average", 120)
        print(predicted_cost)
    elif heuristic == 'abs':
        predicted_cost = solve_sas(parser, [("abs", abstract_heuristic(parser.begin_state.variables,
                                                                       parser.end_state.variables, parser, gamma,
                                                                       projections[0]))])
    elif heuristic == "alternation":
        heuristics = [
            ("abs", abstract_heuristic(parser.begin_state.variables, parser.end_state.variables, parser, gamma, x)) for
            x in projections]
        print("starting alt")
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
    # create_plan("problems/driverlog/driverlog-6.sas")driverlog/driverlog-10.sas"
    cProfile.run('create_plan("problems/barman-opt11/barman-opt11-1.sas")', 'benchmark_stats')

    # Analyze results
    p = pstats.Stats('benchmark_stats')
    p.strip_dirs().sort_stats('cumulative').print_stats(30)
