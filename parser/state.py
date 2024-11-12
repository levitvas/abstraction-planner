from collections import defaultdict
from typing import Iterator


class State:
    def __init__(self, variables, original_state=-1, name=None):
        # Currently variables are strings
        # TODO: potentially change to integers
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
        a = self.variables == other.variables
        b = self.action_id == other.action_id
        return a and b

    def copy(self):
        return State(self.variables.copy(), self.action_id, self.name)

    def __iter__(self) -> Iterator:
        return iter(self.variables)
