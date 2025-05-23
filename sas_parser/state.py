from collections import defaultdict
from typing import Iterator


class State:
    __slots__ = ('variables', 'name', 'action_result', 'action_id', 'shadow_state', 'position')

    def __init__(self, variables, original_state=-1, name=None):
        # Currently variables are strings
        self.variables: list[int] = variables
        self.name: str = name
        self.action_result: dict[int, list] = defaultdict(list)  # first one is normal state, second is shadow
        self.action_id = original_state  # Will contain the pos of the original state
        self.shadow_state = True if original_state != -1 else False
        self.position: int = 0

    def __str__(self):
        return '{} {} {} {}'.format(self.variables, self.action_id, self.shadow_state, self.position)

    def change_pos(self, pos):
        self.position = pos

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, State):  # Type check
            return False
        if self.action_id != other.action_id or self.shadow_state != other.shadow_state:
            return False
        return self.variables == other.variables

    def __hash__(self):
        h = hash(tuple(self.variables))
        h = h * 31 + hash(self.shadow_state)
        h = h * 31 + hash(self.action_id)
        return h

    def copy(self):
        return State(self.variables.copy(), self.action_id, self.name)

    def __iter__(self):
        return iter(self.variables)
