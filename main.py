from heuristics.abstract_heuristic import abstract_heuristic
from heuristics.pdb_heuristic import pdb_heuristic
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
    file_single = 'transport_example.sas'
    projection = [1, 2]

    with open(file_single) as f:
        lines = f.read().split('\n')
    parser = Parser(lines)

    print("Solving SAS file: {}".format(file_single))

    print(parser.begin_state)
    abstract_h = abstract_heuristic(parser.begin_state.variables, parser.end_state.variables, parser, gamma, projection)
    abstract_pdb = pdb_heuristic(parser.begin_state.variables, parser.end_state.variables, parser, gamma, projection)

    predicted_cost = abstract_h(parser.begin_state.variables)
    print(predicted_cost)

    predicted_cost = abstract_pdb(parser.begin_state.variables)
    print(predicted_cost)
