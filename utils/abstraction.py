from parser.action import OperatorSas
from parser.parser import Parser
from parser.state import State


def abstract_all(parser: Parser, projection: list[int]):
    projection_vars = projection
    abs_pos: list[int] = []
    for (i, v) in enumerate(parser.variables):
        if i in projection_vars:
            continue
        abs_pos.append(i)
    # abstraction_var: list[State] = [parser.variables[i] for i in abs_pos]

    # Abstract beginning state
    new_begin_state: State = parser.begin_state.copy()
    for (idx, atom) in enumerate(new_begin_state):
        if idx in abs_pos:
            new_begin_state.variables[idx] = -1

    # Abstract ending state
    new_end_state: State = parser.end_state.copy()
    for (idx, atom) in enumerate(new_end_state):
        if idx in abs_pos:
            new_begin_state.variables[idx] = -1


    # Abstract all actions
    new_operators = []
    for action in parser.operators:
        abs_op = action.abstract(parser.variables, abs_pos)
        new_operators.append(abs_op)

    return new_begin_state, new_end_state, new_operators

def action_reduction(new_operators: list[OperatorSas]):
    # 1. Group abstracted operators with same preconditions and effects
    # 2. Check for domination, i.e. if one operator has subset of preconditions, same effects but lower probability
    # 3. If dominated, remove from list
    # 4. If not dominated, sum the probabilities of the groups

    print("Starting operator reduction")
    print("% Operators before reduction: {}".format(len(new_operators)))

    # Step 1
    groups = {}
    for op in new_operators:
        abstract_pre, abstract_eff = op.preconditions, op.effects
        key = (frozenset(abstract_pre.items()), frozenset(abstract_eff.items()))
        if key not in groups:
            groups[key] = []
        groups[key].append(op)

    # Step 2
    for (pre, eff), ops in groups.items():
        if len(pre) == 0:
            continue
        for (pre2, eff2), ops2 in groups.items():
            if (pre, eff) == (pre2, eff2):
                continue
            if pre.issubset(pre2) and eff == eff2:
                for op in ops:
                    for op2 in ops2:
                        if op.probability > op2.probability:
                            # new_operators.remove(op)
                            groups[(pre2, eff2)].remove(op)
                            break
    print("% Operators after removing dominated operators: {}".format(sum(len(v) for v in groups.values())))

    # Step 3
    final_operators = []
    for (pre, eff), ops in groups.items():
        new_probability = sum(op.probability for op in ops)
        # TODO: Revise the naming
        final_operators.append(
            OperatorSas(ops[0].preconditions, ops[0].effects, ops[0].cost, ops[0].name, new_probability))

    print("% Operators after summing probabilities: {}".format(len(final_operators)))
    return final_operators

def create_state_space_with_shadow_states(operators, start_state):
    visited = [start_state]
    stack = [start_state]
    shadow_state_number = 0

    position = 0
    while stack:
        cur_state = stack.pop(0)
        # TODO: Change to existing get_applicable in a_star, if it will add performance. Check it out later
        # print("Current state: ", cur_state.variables)
        for idx, action in enumerate(operators):
            # print("  Try action: ", action.name)
            # Create the shadow states here directly
            if action.applicable(cur_state, idx):
                new_state, shadow_state = action.apply(cur_state, idx)
                # print("    Passed action: ",action.preconditions, action.effects, cur_state.variables, new_state.variables)

                # TODO: Recheck mutex groups
                # if not mutex_legal(new_state, parser.mutex_groups, parser.variables):
                #     continue

                if new_state not in visited:
                    position += 1
                    new_state.change_pos(position)
                    visited.append(new_state)
                    stack.append(new_state)
                else:
                    # for i in reversed(visited):
                    #     if i == new_state:
                    #         new_state.change_pos(i.position)
                    # TODO: Change back if slower
                    new_state.change_pos(visited[visited.index(new_state)].position)
                # cur_state.action_state[copy.copy(action)] = new_state
                cur_state.action_result[idx].append(new_state.position)

                if shadow_state is not None:
                    if shadow_state not in visited:
                        shadow_state_number += 1
                        position += 1
                        shadow_state.change_pos(position)
                        visited.append(shadow_state)
                        stack.append(shadow_state)
                    else:
                        shadow_state.change_pos(visited[visited.index(shadow_state)].position)
                    cur_state.action_result[idx].append(shadow_state.position)

    return visited, shadow_state_number
