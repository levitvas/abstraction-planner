import sys

from a_star_search import a_star
from heuristics.abstract_heuristic import abstract_h
from heuristics.hmax_heuristic import hmax
from heuristics.lmcut_heuristic import lm_cut
from parser.parser import Parser


def solve_sas(parser: Parser, heuristic, gamma=None, projection=None):
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

    # print(parser.begin_state)
    # print(parser.end_variables)
    # print(parser.end_variables)
    # print(F)

    ret = a_star(F, parser.begin_state.variables, parser.operators, parser.end_variables, heuristic, len(parser.variables), parser, gamma, projection)
    if not ret:
        return 0
    cost, path = ret[0], ret[1]
    for (node, act) in path:
        print(act.name)
    print(f"Plan cost: {cost}")
    return bfs_sum


if __name__ == '__main__':
    # input_f = sys.argv[1]
    # heuristic = sys.argv[2]
    input_f = 'blocks.sas'
    heuristic = 'abs'
    projection = [1, 2]
    gamma = 0.9

    with open(input_f) as f:
        lines = f.read().split('\n')
    parser = Parser(lines)

    if heuristic == 'hmax':
        predicted_cost = solve_sas(parser, hmax)
    elif heuristic == 'abs':
        predicted_cost = solve_sas(parser, abstract_h, gamma, projection)
    else:
        predicted_cost = solve_sas(parser, lm_cut)

