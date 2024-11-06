import copy
import numpy as np

from hiive.mdptoolbox.mdp import ValueIteration
from action import OperatorSas
from parser import Parser
from scipy.sparse import csr_matrix
from state import State


def check_goal(end_state, current_state):
    for atom in end_state.variables:
        if atom in current_state.variables:
            continue
        else:
            return False

    return True


def bfs_solve(operators, begin_state, end_state):
    visited = [begin_state]
    stack = [(begin_state, [])]
    while stack:
        cur_state, path = stack.pop(0)
        for action in operators:
            if action.applicable(cur_state):
                new_state = action.apply(cur_state)
                if new_state not in visited:
                    # Check if goal
                    if check_goal(end_state, new_state):
                        return path + [action]
                    visited.append(new_state)
                    stack.append((new_state, path + [action]))

    return []


def mutex_legal(state, mutex_groups, variables):
    for group in mutex_groups:
        mutex = False
        for var, value in group:
            if variables[var].variables[value] in state.variables:
                if mutex:
                    return False
                mutex = True

    return True


def create_state_space(operators, parser):
    visited = [parser.begin_state]
    stack = [parser.begin_state]

    position = 0
    while stack:
        cur_state = stack.pop(0)
        for idx, action in enumerate(operators):
            if action.applicable(cur_state):
                new_state = action.apply(cur_state)
                if not mutex_legal(new_state, parser.mutex_groups, parser.variables):
                    continue
                cur_state.action_state[copy.copy(action)] = new_state
                if new_state not in visited:
                    position += 1
                    new_state.change_pos(position)
                    visited.append(new_state)
                    stack.append(new_state)
                else:
                    for i in reversed(visited):
                        if i == new_state:
                            new_state.change_pos(i.position)

    return visited


def add_shadow_states(states):
    position = states[-1].position
    for st in states:
        if st.variables[-1] == 'shadow':
            continue
        local_shadows = {}
        for idx, actione in enumerate(list(st.action_state)):
            if actione.probability() != 1 and not actione.shadow:
                # Create a shadow state and shadow action for unsuccessful action
                shadow_action = OperatorSas(None, None, actione)
                st_tmp = State(st.variables.copy(), shadow_action)
                position += 1
                st_tmp.change_pos(position)
                st_tmp.variables.append('shadow')
                st.action_state[shadow_action] = st_tmp
                local_shadows[actione] = (st_tmp, shadow_action)
        for idx, (shadow, orig_action) in local_shadows.items():
            list_without = st.action_state.copy()
            list_without.pop(idx)
            list_without.pop(orig_action)
            shadow.action_state.update(list_without)
        states.extend(i[0] for i in local_shadows.values())


def solve_sas(sas_file, parser, gamma, projection):
    # ------------ Abstraction implementation
    projection_vars = projection
    abs_pos = []
    for (i, v) in enumerate(parser.variables):
        if i in projection_vars:
            continue
        abs_pos.append(i)
    abstraction_var = [parser.variables[i].variables for i in abs_pos]

    new_operators = []
    # Abstract beginning state
    for idx, atom in enumerate(parser.begin_state.variables):
        for var in abstraction_var:
            if atom in var:
                parser.begin_state.variables[idx] = 'X'

    # Abstract ending state
    for idx, atom in enumerate(parser.end_state.variables):
        for var in abstraction_var:
            if atom in var:
                parser.end_state.variables[idx] = 'X'

    # Abstract all actions
    for action in parser.operators:
        action.abstract(parser.variables, abs_pos)
        new_operators.append(action)

    pos = 0
    # check two actions if they have the same preconditions(values too), if so, keep the one with max probability
    while True:
        action = new_operators[pos]
        pos_other = 0
        while True:
            if pos_other >= len(new_operators):
                break
            other = new_operators[pos_other]
            if pos == pos_other:
                pos_other += 1
                continue
            if action.get_max(other):
                if action.probability() < other.probability():
                    new_operators.pop(pos)
                    pos -= 1
                    break
                else:
                    new_operators.pop(pos_other)
                    continue
            pos_other += 1
        pos += 1
        if pos >= len(new_operators):
            break

    n_operators = []
    for action in new_operators:
        if action in n_operators:
            # same precon var -> sum
            # same precon(var and value) -> max
            n_operators[n_operators.index(action)].numerator += action.numerator
        else:
            n_operators.append(action)

    bfs = create_state_space(n_operators, parser)  # Starts from beginning state and generates all possible
    add_shadow_states(bfs)  # Adds shadow states to the existing state space

    goal_idx = []
    actions = []
    for state in bfs:
        for act in state.action_state.keys():
            if check_goal(parser.end_state, state) and state.variables[-1] != 'shadow':
                goal_idx.append(bfs.index(state))
            if act not in actions and not act.shadow:
                actions.append(act)

    if len(actions) * len(bfs) >= 2000000:  # Change number to make a limit. The higher the number the longer it runs
        print("Too big")
        return 0

    if len(actions) == 0:
        print("no actions")
        return 0

    transition_ar = []
    reward_ar = []

    # Start filling up and creating the transition and reward sparse matrices
    for idx, operator in enumerate(actions):
        data = []
        row = []
        col = []
        data_r = []
        row_r = []
        col_r = []
        for pos, state in enumerate(bfs):
            if operator.denominator == 0:
                exit(0)
            if operator not in state.action_state:
                data.append(1.0)
                row.append(pos)
                col.append(pos)

                data_r.append(-1000.0)
                row_r.append(pos)
                col_r.append(pos)
                continue
            if pos in goal_idx:
                data.append(1.0)
                row.append(pos)
                col.append(pos)

                data_r.append(0.0)
                row_r.append(pos)
                col_r.append(pos)
                continue
            next_state = state.action_state[operator]

            pos_n = next_state.position
            data.append(round(operator.probability(), 4))
            row.append(pos)
            col.append(pos_n)

            data_r.append(-operator.cost)
            row_r.append(pos)
            col_r.append(pos_n)
            if operator.probability() != 1:
                shadow_op = OperatorSas(None, None, operator)
                sh_state = state.action_state[shadow_op]

                data.append(round(shadow_op.probability(), 4))
                row.append(pos)
                sh_index = sh_state.position
                col.append(sh_index)

                data_r.append(0.0)
                row_r.append(pos)
                col_r.append(sh_index)

        if len(bfs) - 1 not in row or len(bfs) - 1 not in col:
            data.append(0.0)
            row.append(len(bfs) - 1)
            col.append(len(bfs) - 1)

        if len(bfs) - 1 not in row_r or len(bfs) - 1 not in col_r:
            data_r.append(0.0)
            row_r.append(len(bfs) - 1)
            col_r.append(len(bfs) - 1)
        a = csr_matrix((np.array(data), (np.array(row), np.array(col))))
        b = csr_matrix((np.array(data_r), (np.array(row_r), np.array(col_r))))
        transition_ar.append(a)
        reward_ar.append(b)

    bfs_sum = 0
    try:
        a = np.array(transition_ar)
        b = np.array(reward_ar)
        vi = ValueIteration(a, b, gamma)
    except OverflowError:
        print("MDP library error")
        return 0.0
    except RuntimeWarning:
        print("MDP library error")
        return 0.0
    vi.run()
    # print("V is {}".format(vi.V)) #  Uncomment to print the value function for each state

    bfs_solved = bfs_solve(n_operators, parser.begin_state, parser.end_state)
    if bfs_solved:
        for a in bfs_solved:
            # print(a) #  Prints the path in the raw abstraction
            bfs_sum -= a.cost

    return vi.V[0]


if __name__ == '__main__':
    gamma = 0.9
    file_single = 'transport_example.sas'
    projection = [1, 2]

    with open(file_single) as f:
        lines = f.read().split('\n')
    parser = Parser(lines)

    predicted_cost = solve_sas(file_single, parser, gamma, projection)
    print(predicted_cost)
