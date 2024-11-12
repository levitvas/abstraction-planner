import copy
import numpy as np

from hiive.mdptoolbox.mdp import ValueIteration
from parser.action import OperatorSas
from scipy.sparse import csr_matrix
from parser.parser import Parser
from parser.state import State
from utils.abstraction import abstract_all, action_reduction, create_state_space_with_shadow_states


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

def solve_sas(sas_file, parser: Parser, gamma, projection: list[int]):
    # ------------ Abstraction implementation

    new_start, new_end, new_operators = abstract_all(parser, projection)
    # TODO: Change all states to use new states
    # TODO: Change all stuff to use integers instead of strings

    print(new_operators)
    final_operators = action_reduction(new_operators)
    print(final_operators)
    print("% ------ %")

    print("Creating state space")
    bfs_states: list[State]
    bfs_states, shadow_num = create_state_space_with_shadow_states(final_operators, new_start)  # Starts from beginning state and generates all possible states
    print("% State space size: {}, shadow states {}".format(len(bfs_states) - shadow_num, shadow_num))
    print([op.action_result.items() for op in bfs_states])
    print("% ------ %")

    exit(0)

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

    print("Solving SAS file: {}".format(file_single))

    predicted_cost = solve_sas(file_single, parser, gamma, projection)
    print(predicted_cost)
