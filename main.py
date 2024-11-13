import numpy as np

from hiive.mdptoolbox.mdp import ValueIteration
from parser.action import OperatorSas
from scipy.sparse import csr_matrix
from parser.parser import Parser
from parser.state import State
from utils.abstraction import abstract_all, action_reduction, create_state_space_with_shadow_states
from utils.help_functions import check_goal

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


    #  In theory can find the goal ids when state spacing
    print(new_end)
    goal_idx = []
    for (idx, state) in enumerate(bfs_states):
        if check_goal(new_end, state) and state.shadow_state is False:
            goal_idx.append(idx)

    if len(final_operators) * len(bfs_states) >= 2000000:  # Change number to make a limit. The higher the number the longer it runs
        print("! Excessive state space")
        return 0

    if len(final_operators) == 0:
        print("! No actions")
        return 0

    transition_ar = []
    reward_ar = []

    # Start filling up and creating the transition and reward sparse matrices
    for idx, operator in enumerate(final_operators):
        data = []
        row = []
        col = []
        data_r = []
        row_r = []
        col_r = []
        state: State
        for pos, state in enumerate(bfs_states):
            if operator.probability == 0:
                print("! FATAL ERROR: Operator probability is 0")
                exit(0)

            if idx not in state.action_result.keys():
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

            next_state = state.action_result[idx][0]
            data.append(round(operator.probability, 4))
            row.append(pos)
            col.append(next_state)

            data_r.append(-operator.cost)
            row_r.append(pos)
            col_r.append(next_state)

            if operator.probability != 1:
                sh_state = state.action_result[idx][1]
                data.append(round(1 - operator.probability, 4))
                row.append(pos)
                sh_index = sh_state
                col.append(sh_index)

                data_r.append(0.0)
                row_r.append(pos)
                col_r.append(sh_index)

        bfs_len = len(bfs_states)
        if bfs_len - 1 not in row or bfs_len - 1 not in col:
            data.append(0.0)
            row.append(bfs_len - 1)
            col.append(bfs_len - 1)

        if bfs_len - 1 not in row_r or bfs_len - 1 not in col_r:
            data_r.append(0.0)
            row_r.append(bfs_len - 1)
            col_r.append(bfs_len - 1)

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

    # bfs_solved = bfs_solve(n_operators, parser.begin_state, parser.end_state)
    # if bfs_solved:
    #     for a in bfs_solved:
            # print(a) #  Prints the path in the raw abstraction
            # bfs_sum -= a.cost

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
