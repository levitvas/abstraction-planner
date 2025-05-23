import logging

from sas_parser.parser import Parser
from sas_parser.state import State
from utils.abstraction import abstract_all, create_state_space_with_shadow_states, action_reduction
from utils.help_functions import check_goal


def calculate_state_values_bfs(state_space, goal_states):
    values = {state: float('inf') for state in state_space if not state.shadow_state}
    queue = []

    # Initialize goal states with value 0 and add to queue
    for goal in goal_states:
        if goal in values:
            values[goal] = 0
            queue.append(goal)

    while queue:
        state = queue.pop(0)
        if state.shadow_state:
            continue
        current_value = values[state]

        # Find all predecessors of current state
        for pred_state in state_space:
            if pred_state.shadow_state:
                continue
            for action_results in pred_state.action_result.values():
                for successor_pos in action_results:
                    if state_space[successor_pos] == state:
                        new_value = current_value + 1
                        if new_value < values[pred_state]:
                            values[pred_state] = new_value
                            queue.append(pred_state)

    return values

class pdb_heuristic:
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

        logging.debug(new_operators)
        final_operators = action_reduction(new_operators)
        logging.debug("% ------ %")

        logging.debug("Creating state space")
        bfs_states: list[State]
        bfs_states, shadow_num, state_positions = create_state_space_with_shadow_states(final_operators,
                                                                       new_start)  # Starts from beginning state and generates all possible states
        self.state_positions = state_positions
        logging.debug("% State space size: {}, shadow states {}".format(len(bfs_states) - shadow_num, shadow_num))
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

        self.values = calculate_state_values_bfs(bfs_states, [bfs_states[idx] for idx in goal_idx])
        self.bfs_states = bfs_states

    def __call__(self, state):
        abstracted_state: State = State(state.copy())
        for (idx, atom) in enumerate(abstracted_state):
            if idx in self.abs_pos:
                abstracted_state.variables[idx] = -1

        return self.values[abstracted_state]
