from parser.parser import Parser


def issubset(subset: frozenset[tuple[int, int]], superset: frozenset[tuple[int, int]]) -> bool:
    for (pre, eff) in subset:
        if (pre, eff) not in superset:
            return False

    return True

def check_goal(end_state, current_state):
    for (var, atom) in enumerate(end_state.variables):
        if atom == -1 or current_state.variables[var] == atom:
            continue
        else:
            return False
    return True
