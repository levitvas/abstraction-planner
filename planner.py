import sys

from lmcut import lm_cut
from hmax import hmax
from parser import Parser
from a_star import a_star, initialize, generate_applicable_operators


def solve_sas(parser: Parser, heuristic):
    bfs_sum = 0
    F = []
    g = set()
    for num, v in enumerate(parser.variables):
        for d in range(len(v.variables)):
            F.append((num, d))

    s_0 = set()
    for num, var in enumerate(parser.begin_state.variables):
        s_0.add((num, var))
    for num, var in enumerate(parser.end_variables):
        g.add(F.index((var, parser.end_state.variables[num])))

    # print(parser.begin_state)
    # print(parser.end_variables)
    # print(parser.end_variables)
    # print(F)

    ret = a_star(F, parser.begin_state.variables, parser.operators, parser.end_variables, heuristic, len(parser.variables))
    if not ret:
        return 0
    cost, path = ret[0], ret[1]
    for (node, act) in path:
        print(act.name)
    print(f"Plan cost: {cost}")
    return bfs_sum


if __name__ == '__main__':
    input_f = sys.argv[1]
    heuristic = sys.argv[2]

    with open(input_f) as f:
        lines = f.read().split('\n')
    parser = Parser(lines)

    if heuristic == 'hmax':
        predicted_cost = solve_sas(parser, hmax)
    else:
        predicted_cost = solve_sas(parser, lm_cut)

