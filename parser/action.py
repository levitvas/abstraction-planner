import copy

from parser.state import State


class OperatorSas:
    def __init__(self, precond, eff, cost, name, probability=1, original_action=None, copy=None):
        if copy is not None:
            self.name = copy.name
            self.preconditions = copy.preconditions.copy()
            self.variables = list(self.preconditions.copy().keys())
            self.values = list(self.preconditions.copy().values())
            self.effects = copy.effects.copy()
            self.shadow = copy.shadow
            self.numerator = copy.numerator
            self.denominator = copy.denominator
            self.cost = copy.cost

        elif original_action is not None:
            self.name = original_action.name
            self.preconditions = original_action.preconditions
            self.variables = list(self.preconditions.copy().keys())
            self.values = list(self.preconditions.copy().values())
            self.effects = original_action.effects
            self.cost = original_action.cost

            self.shadow = True

            self.numerator = original_action.denominator - original_action.numerator
            self.denominator = original_action.denominator
        else:
            self.name: str = name
            self.cost: int = cost

            self.preconditions: dict[int, int] = precond
            self.effects: dict[int, int] = eff

            self.variables: list[int] = list(self.preconditions.keys())
            self.values: list[int] = list(self.preconditions.values())

            self.probability: float = probability

            self.shadow: bool = False

    def __str__(self):
        ret = " Name: {}, Shadow: {}, {}".format(self.name, self.shadow, self.probability)
        return ret

    def __eq__(self, other):
        return (self.preconditions == other.preconditions) and (self.effects == other.effects) and \
            (self.shadow == other.shadow)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return OperatorSas(None, None, None, self)

    def __hash__(self):
        return hash(self.name)

    def applicable(self, state: State, action_id: int):
        if state.action_id == action_id:
            return False
        for var, atom in self.preconditions.items():
            if state.variables[var] != atom:
                return False
        return True

    def apply(self, state: State, action_id: int):
        new_state = State(state.variables.copy())
        for var, atom in self.effects.items():
            new_state.variables[var] = atom
        if self.probability != 1:
            shadow_state = State(state.variables.copy(), action_id, self.name)
            shadow_state.shadow_state = True
            return new_state, shadow_state
        return new_state, None

    def abstract(self, variables, positions):
        joined = self.preconditions.items()
        abstracted_op = OperatorSas(self.preconditions, self.effects, self.cost, self.name, self.probability)
        for (var, atom) in joined:
            if var in positions:
                # if values == 0:
                #     values = len(variables[var].variables)
                # else:
                #     values *= len(variables[var].variables)
                abstracted_op.probability *= 1./len(variables[var].variables)
        for pos in positions:
            abstracted_op.preconditions.pop(pos, None)
            abstracted_op.effects.pop(pos, None)

        return abstracted_op

    def get_max(self, other):
        return (self.preconditions == other.preconditions) and (self.variables == other.variables) and (
                    self.values == other.values) and (self.effects == other.effects)
