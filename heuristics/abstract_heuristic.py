import numpy as np
import logging

from scipy.sparse import csr_matrix

from sas_parser.parser import Parser
from sas_parser.state import State
from utils.abstraction import abstract_all, create_state_space_with_shadow_states, action_reduction
from utils.gpu_value import FastValueIteration
from utils.help_functions import check_goal
from value_iter import optimized_value_iteration


class abstract_heuristic:
    def __init__(self, begin_state, end_state, parser: Parser, gamma, projection: list[int]):
        # ------------ Abstraction implementation
        # logging.basicConfig(level=logging.DEBUG)
        logging.debug("%% Abstracting")
        logging.debug(begin_state)
        abs_pos: list[int] = []
        for (i, v) in enumerate(parser.variables):
            if i in projection:
                continue
            abs_pos.append(i)
        self.abs_pos = abs_pos
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
        bfs_states, shadow_num, state_positions = create_state_space_with_shadow_states(final_operators,
                                                                       new_start)  # Starts from beginning state and generates all possible states
        self.state_positions = state_positions
        logging.debug("% State space size: {}, shadow states {}".format(len(bfs_states) - shadow_num, shadow_num))
        # logging.debug([op.action_result.items() for op in bfs_states])
        logging.debug("% ------ %")

        #  In theory can find the goal ids when state spacing
        goal_idx = []
        for (idx, state) in enumerate(bfs_states):
            if check_goal(new_end, state) and state.shadow_state is False:
                goal_idx.append(idx)

        if len(final_operators) * len(
                bfs_states) >= 2000000:  # Change number to make a limit. The higher the number the longer it runs
            logging.error("! Excessive state space")

        if len(final_operators) == 0:
            logging.error("! No actions")

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
                data.append(round(operator.probability, 5))
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
                    data.append(round(1 - operator.probability, 5))
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
        # print("Starting MDP")
        try:
            a = np.array(transition_ar)
            b = np.array(reward_ar)
            vi_f = FastValueIteration(transition_ar, reward_ar, gamma)
            # vi = ValueIteration(a, b, gamma)
        except OverflowError:
            logging.error("MDP library error")
        except RuntimeWarning:
            logging.error("MDP library error")
        # vi.run()
        vi_f.run()

        # self.values = vi.V
        # print(vi.time)
        self.values = vi_f.V.cpu().numpy() * -1
        # print(self.values)
        # print(vi_f.V)
        self.bfs_states = bfs_states
        # print("Finished solving MDP")

    def __call__(self, state):
        abstracted_state: State = State(state.copy())
        for (idx, atom) in enumerate(abstracted_state):
            if idx in self.abs_pos:
                abstracted_state.variables[idx] = -1

        # return self.values[self.bfs_states.index(abstracted_state)]
        return self.values[self.state_positions[abstracted_state]]
