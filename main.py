from heuristics.abstract_heuristic import abstract_h
from sas_parser.parser import Parser


def mutex_legal(state, mutex_groups, variables):
    for group in mutex_groups:
        mutex = False
        for var, value in group:
            if variables[var].variables[value] in state.variables:
                if mutex:
                    return False
                mutex = True

    return True

if __name__ == '__main__':
    gamma = 0.9
    file_single = 'driverlog-1.sas'
    projection = [1, 2, 6]

    with open(file_single) as f:
        lines = f.read().split('\n')
    parser = Parser(lines)

    print("Solving SAS file: {}".format(file_single))

    predicted_cost = abstract_h(parser.begin_state.variables, parser.end_state.variables, parser, gamma, projection)
    print(predicted_cost)
