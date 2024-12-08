import numpy as np
import logging

from hiive.mdptoolbox.mdp import ValueIteration
from scipy.sparse import csr_matrix

from sas_parser.parser import Parser
from sas_parser.state import State
from utils.abstraction import abstract_all, create_state_space_with_shadow_states, action_reduction
from utils.help_functions import check_goal
from value_iter import optimized_value_iteration


def value_iteration_optimized(transition_ar, reward_ar, gamma=0.99, epsilon=1e-6, max_iter=1000):
    n_states = transition_ar[0].shape[0]
    V = np.zeros(n_states)

    for _ in range(max_iter):
        V_old = V.copy()
        Q = np.array([reward_ar[a].dot(np.ones(n_states)) + gamma * transition_ar[a].dot(V)
                      for a in range(len(transition_ar))])
        V = Q.max(axis=0)
        if np.max(np.abs(V - V_old)) < epsilon:
            break

    return V[0]

def abstract_h(begin_state, end_state, parser: Parser, gamma, projection: list[int]):
    # ------------ Abstraction implementation
    # logging.basicConfig(level=logging.DEBUG)
    logging.debug("%% Abstracting")
    logging.debug(begin_state)
    new_start, new_end, new_operators = abstract_all(begin_state.copy(), end_state.copy(), parser, projection)
    logging.debug(new_start)
    logging.debug(new_end)
    # TODO: Change all stuff to use integers instead of strings

    logging.debug(new_operators)
    final_operators = action_reduction(new_operators)
    # logging.debug(final_operators)
    logging.debug("% ------ %")

    logging.debug("Creating state space")
    bfs_states: list[State]
    bfs_states, shadow_num = create_state_space_with_shadow_states(final_operators, new_start)  # Starts from beginning state and generates all possible states
    logging.debug("% State space size: {}, shadow states {}".format(len(bfs_states) - shadow_num, shadow_num))
    # logging.debug([op.action_result.items() for op in bfs_states])
    logging.debug("% ------ %")


    #  In theory can find the goal ids when state spacing
    goal_idx = []
    for (idx, state) in enumerate(bfs_states):
        if check_goal(new_end, state) and state.shadow_state is False:
            goal_idx.append(idx)

    if len(final_operators) * len(bfs_states) >= 2000000:  # Change number to make a limit. The higher the number the longer it runs
        logging.error("! Excessive state space")
        return 0

    if len(final_operators) == 0:
        logging.error("! No actions")
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
                logging.error("! FATAL ERROR: Operator probability is 0")
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

            if operator.probability > 1:
                logging.error("! FATAL ERROR: Operator probability > 1")
                exit(0)

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
        # vi = ValueIteration(a, b, gamma)
    except OverflowError:
        logging.error("MDP library error")
        return 0.0
    except RuntimeWarning:
        logging.error("MDP library error")
        return 0.0
    vi.run()

    print(vi.V)
    return vi.V[0]