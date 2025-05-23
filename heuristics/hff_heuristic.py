import math
import sys

def hff(all_problem_facts, initial_state_facts, actions_list, goal_facts,
        var_len_unused, preconditions_of_fact_map_unused):

    # Calculate s* (h_max-like fact costs) and bs (best supporters)
    s_star_costs = {fact: math.inf for fact in all_problem_facts}
    bs_map = {fact: None for fact in all_problem_facts}  # Stores action

    for fact in initial_state_facts:
        s_star_costs[fact] = 0

    # Iteratively update costs until a fixed point (h_max style relaxation)
    updated_in_pass = True
    while updated_in_pass:
        updated_in_pass = False
        for action in actions_list:
            max_prec_cost = 0
            preconditions_met = True
            if action.preconditions:
                for var_idx, req_val in action.preconditions.items():
                    prec_fact = (var_idx, req_val)
                    cost = s_star_costs.get(prec_fact, math.inf)
                    if cost == math.inf:
                        preconditions_met = False
                        break
                    max_prec_cost = max(max_prec_cost, cost)

            if preconditions_met:
                cost_to_achieve_effs = action.cost + max_prec_cost
                for eff_var, eff_val in action.effects.items():
                    effect_fact = (eff_var, eff_val)
                    if cost_to_achieve_effs < s_star_costs.get(effect_fact, math.inf):
                        s_star_costs[effect_fact] = cost_to_achieve_effs
                        bs_map[effect_fact] = action
                        updated_in_pass = True

    # Initial Goal Reachability Check
    for g_fact in goal_facts:
        if s_star_costs.get(g_fact, math.inf) == math.inf:
            return math.inf  # Goal unreachable in relaxed problem

    # Extract relaxed plan and sum costs
    relaxed_plan_actions = set()  # Store action objects
    current_subgoals = set(goal_facts)

    while not current_subgoals.issubset(initial_state_facts):
        achievable_subgoals = current_subgoals - initial_state_facts
        if not achievable_subgoals: break  # All subgoals are in initial state

        # Select p: subgoal in achievable_subgoals with max s_star_cost
        chosen_p = None
        max_p_cost = -1.0
        for sub_g in achievable_subgoals:
            cost_g = s_star_costs.get(sub_g, math.inf)
            if cost_g == math.inf: return math.inf
            if cost_g > max_p_cost:
                max_p_cost = cost_g
                chosen_p = sub_g

        if chosen_p is None: break  # Should not happen if achievable_subgoals is not empty

        supporter_action = bs_map.get(chosen_p)
        if supporter_action is None:  # Non-initial fact with no supporter
            if s_star_costs.get(chosen_p, math.inf) == 0:  # it was an initial fact after all
                current_subgoals.discard(chosen_p)
                continue
            return math.inf  # Error condition

        relaxed_plan_actions.add(supporter_action)

        current_subgoals.discard(chosen_p)  # p is achieved
        for eff_var, eff_val in supporter_action.effects.items():
            current_subgoals.discard((eff_var, eff_val))  # Remove other effects
        if supporter_action.preconditions:
            for pre_var, pre_val in supporter_action.preconditions.items():
                current_subgoals.add((pre_var, pre_val))  # Add new subgoals

    plan_cost = sum(action.cost for action in relaxed_plan_actions)
    return plan_cost


if __name__ == '__main__':
    from sas_parser.parser import Parser
    input_f = sys.argv[1]
    with open(input_f) as f: lines = f.read().split('\n')
    parser = Parser(lines)

    F_all_facts_hff = []
    for var_idx, var_obj in enumerate(parser.variables):
        domain_size = len(var_obj.variables)
        for domain_val_idx in range(domain_size):
            F_all_facts_hff.append((var_idx, domain_val_idx))

    s_0_hff_facts = set()
    for var_idx, val_idx in enumerate(parser.begin_state.variables):
        s_0_hff_facts.add((var_idx, val_idx))

    g_hff_facts = set()
    for var_idx, val_idx in parser.end_variables.items():
        g_hff_facts.add((var_idx, val_idx))

    heuristic_val = hff(
        F_all_facts_hff,
        s_0_hff_facts,
        parser.operators,
        g_hff_facts,
        len(parser.variables),
        {}
    )
    print(f"FF Heuristic value (condensed): {heuristic_val}")