import sys
import math

from a_star_search import initialize
from sas_parser.parser import Parser


def hmax(facts, state, actions, goal, var_len, preconditions_of):
    sigma = {fact: math.inf for fact in facts}
    for fact in state:
        sigma[fact] = 0

    for action in actions:
        if not action.preconditions:
            for key, fact in action.effects.items():
                sigma[fact] = min(sigma[(key, fact)], action.cost)

    U = [len(action.preconditions) for num, action in enumerate(actions)]
    C = set()

    while not goal.issubset(C):
        min_q_v = math.inf
        q = None
        # for fact in facts - C:
        for fact, val in sigma.items():
            if fact not in C:
            # val = sigma[fact]
                if val < min_q_v:
                    q = fact
                    min_q_v = val

        # q = min([fact for fact in sigma.keys()], key=lambda x: sigma[x])
        if q is None:
            # If q is None, it means the goal is unreachable
            return math.inf

        C.add(q)
        op_idxs = preconditions_of[q]
        for num in op_idxs:
            action = actions[num]
            U[num] -= 1
            if U[num] == 0:
                for key, fact in action.effects.items():
                    v = action.cost + sigma[q]
                    if v < sigma[(key, fact)]:
                        sigma[(key, fact)] = v

    return max([sigma[fact] for fact in goal])


if __name__ == '__main__':
    input_f = sys.argv[1]

    with open(input_f) as f:
        lines = f.read().split('\n')
    parser = Parser(lines)
    F = []
    s_0 = set()
    g = set()

    for num, v in enumerate(parser.variables):
        for d in range(len(v.variables)):
            F.append((num, d))

    s_0 = set()
    for num, var in enumerate(parser.begin_state.variables):
        s_0.add((num, var))
    for num, var in parser.end_variables.items():
        g.add((num, var))

    # for num, val in goal_state.items():
    #     goal_strips.add((num, val))
    counter, preconditions_of, first_visit = initialize(F, parser.operators)
    print(hmax(F, s_0, parser.operators, g, len(parser.variables), preconditions_of))
