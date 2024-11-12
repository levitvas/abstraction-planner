import math
import sys
import copy

from a_star import initialize
from parser.action import OperatorSas
from parser.parser import Parser


def lm_cut(facts, initial_state, ops, goal, var_len, preconditions_of):
    actions = list(ops)
    facts = list(facts)
    preconditions_of = copy.deepcopy(preconditions_of)

    def build_justification_graph(state, hmaxes):
        graph = {}
        backwards = {}
        for fact in facts:
            graph[fact] = []
            backwards[fact] = []
        for num, action in enumerate(actions):
            m_l = [hmaxes[v] for v in action.preconditions.items()]
            precondition = max(action.preconditions.items(), key=lambda v: (hmaxes[v], v[0], v[1]))
            # print(m_l)
            for var, effect in action.effects.items():
                graph[precondition].append((num, (var, effect)))
                backwards[(var, effect)].append((num, precondition))
        return graph, backwards

    def compute_hmax(state, cost_function):
        sigma = {fact: math.inf for fact in facts}
        for fact in state:
            sigma[fact] = 0
        U = [len(action.preconditions) for action in actions]
        C = set()

        while not goal.issubset(C):
            min_q_v = math.inf
            q = None
            for fact, val in sigma.items():
                if fact not in C:
                    if val < min_q_v:
                        q = fact
                        min_q_v = val

            if q is None:
                return math.inf, None

            C.add(q)
            op_idxs = preconditions_of[q]
            for num in op_idxs:
                action = actions[num]
                U[num] -= 1
                if U[num] == 0:
                    for key, fact in action.effects.items():
                        v = cost_function[num] + sigma[q]
                        if v < sigma[(key, fact)]:
                            sigma[(key, fact)] = v

        return max([sigma[fact] for fact in goal]), sigma

    act_len = len(actions)
    in_st = {}
    for var, fact in initial_state:
        in_st[var] = fact
    actions.append(OperatorSas({var_len: '⊥'}, in_st, 0, "from ⊥"))
    g_dict = {}
    for var, fact in goal:
        preconditions_of[(var, fact)].append(act_len+1)
        g_dict[var] = fact
    actions.append(OperatorSas(g_dict, {var_len + 1: '>'}, 0, "to >"))
    preconditions_of[(var_len, '⊥')] = [act_len]
    preconditions_of[(var_len + 1, '>')] = []
    initial_state = {(var_len, '⊥')}
    goal = {(var_len + 1, '>')}
    # print(goal)
    facts.append((var_len, '⊥'))
    facts.append((var_len + 1, '>'))
    cost_function = [a.cost for a in actions]

    h_lmcut = 0
    hmax, hmaxes = compute_hmax(initial_state, cost_function)
    if hmax == math.inf:
        return math.inf

    order = 0

    while hmax != 0:
        # Build justification graph
        graph, backwards = build_justification_graph(initial_state, hmaxes)
        # Find reachable facts to the goal with zero cost
        V_greater = set()
        open_list = list(goal)
        while open_list:
            fact = open_list.pop()
            V_greater.add(fact)
            for action, precondition in backwards[fact]:
                if cost_function[action] == 0 and precondition not in V_greater:
                    open_list.append(precondition)

        # Find the cut
        U_perp = set()
        open_list = list(initial_state)
        while open_list:
            fact = open_list.pop()
            U_perp.add(fact)
            for action, effect in graph[fact]:
                if effect not in U_perp and effect not in V_greater:
                    open_list.append(effect)

        # Extract landmark and update cost function
        landmark = set()
        for fact in U_perp:
            for action, effect in graph[fact]:
                if effect in V_greater:
                    landmark.add(action)


        m = min(cost_function[action] for action in landmark)
        h_lmcut += m
        for action in landmark:
            cost_function[action] -= m

        order += 1
        hmax, hmaxes = compute_hmax(initial_state, cost_function)

    return h_lmcut


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
    print(lm_cut(F, s_0, parser.operators, g, len(parser.variables), preconditions_of))
