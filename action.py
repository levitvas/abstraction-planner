import copy

from state import State


class OperatorSas:
    def __init__(self, precond, eff, cost, name, original_action=None, copy=None):
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
            self.shadow = True
            self.numerator = original_action.denominator - original_action.numerator
            self.denominator = original_action.denominator
            self.cost = original_action.cost
        else:
            self.preconditions = precond
            self.cost = cost
            self.effects = eff
            self.name = name
            self.variables = list(self.preconditions.copy().keys())
            self.values = list(self.preconditions.copy().values())

            self.shadow = False  # Action is a shadow action(normal -> shadow state)

    def __str__(self):
        ret = " Name: {}, Shadow: {}, {}".format(self.name, self.shadow, (self.numerator / self.denominator))
        return ret

    def __eq__(self, other):
        return (self.preconditions == other.preconditions) and (self.effects == other.effects) and (
                self.variables == other.variables) and (self.shadow == other.shadow)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return OperatorSas(None, None, None, self)

    def __hash__(self):
        return hash(self.name)

    def applicable(self, state: State):
        for var, atom in self.preconditions.items():
            if state.variables[var] != atom:
                return False
        return True

    def apply(self, state):
        state_copy = state.variables.copy()
        if self.shadow and 'shadow' not in state_copy:
            state_copy.append('shadow')
            new_act = copy.copy(self)
            return State(state_copy, new_act)
        elif self.shadow and 'shadow' in state_copy:
            new_act = copy.copy(self)
            return State(state_copy, new_act)
        else:
            if not self.shadow and 'shadow' in state_copy:
                state_copy.pop(-1)
            for var, atom in self.effects.items():
                state_copy[var] = atom
        return State(state_copy)

    def abstract(self, variables, positions):
        joined = self.preconditions.items()
        values = 0
        for atom in joined:
            if atom[0] in positions:
                if values == 0:
                    values = len(variables[atom[0]].variables)
                else:
                    values *= len(variables[atom[0]].variables)
                self.denominator = values
        for pos in positions:
            self.preconditions.pop(pos, None)
            self.effects.pop(pos, None)

        return True

    def get_max(self, other):
        return (self.preconditions == other.preconditions) and (self.variables == other.variables) and (
                    self.values == other.values) and (self.effects == other.effects)

    def probability(self):
        return self.numerator / self.denominator
