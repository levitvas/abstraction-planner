import sys

from a_star_search import a_star, gbfs
from heuristics.abstract_heuristic import abstract_heuristic
from heuristics.hmax_heuristic import hmax
from heuristics.lmcut_heuristic import lm_cut
from sas_parser.parser import Parser


def solve_sas(parser: Parser, heuristics):
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


    ret, expanded_states = gbfs(F, parser.begin_state.variables, parser.operators, parser.end_variables, heuristics, len(parser.variables), parser)
    if not ret:
        return 0
    cost, path = ret[0], ret[1]
    for (node, act) in path:
        print(act.name)
    print(f"Plan cost: {cost}")
    print(f"Expanded states: {expanded_states}")
    return bfs_sum

def create_plan():
    # input_f = sys.argv[1]
    # heuristic = sys.argv[2]
    input_f = 'driverlog-3.sas'

    with open(input_f) as f:
        lines = f.read().split('\n')
    parser = Parser(lines)

    heuristic = 'alternation'
    gamma = 0.9
    # Change tie-braking to random - Done
    # For tie-braking try the average of other heuristics - Done
    # Try pure pattern DB, without shadow states, just solve the projection, that is the heuristic value
    # print the heuristic used for the path


    projections = [[3, 6, 8], [1, 4, 5], [2, 7, 8], [0, 1, 5], [2, 3, 5], [1, 3, 5], [4, 5, 6] , [0, 1, 7]] # Adding [2, 3, 5] makes it much worse
    # projections = [[0], [1]]
    # projections = [[1, 6, 7, 8], [2, 3, 4, 5], [3, 4, 5, 6], [0, 2, 3, 8]]

    if heuristic == 'hmax':
        predicted_cost = solve_sas(parser, [("hmax", hmax)])
    elif heuristic == 'abs':
        predicted_cost = solve_sas(parser, [("abs", abstract_heuristic(parser.begin_state.variables, parser.end_state.variables, parser, gamma, projections[0]))])
    elif heuristic == "alternation":
        heuristics = [
            ("abs", abstract_heuristic(parser.begin_state.variables, parser.end_state.variables, parser, gamma, x)) for
            x in projections]
        predicted_cost = solve_sas(parser, heuristics)
    else:
        predicted_cost = solve_sas(parser, lm_cut)

if __name__ == '__main__':
    create_plan()
