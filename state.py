class State:
    def __init__(self, variables, shadow_action=None):
        self.variables = variables
        self.action_state = {}
        self.shadow_action = shadow_action
        self.position = 0

    def __str__(self):
        return '{} {} {}'.format(self.variables, self.shadow_action, self.position)

    def is_there_action(self):
        return self.shadow_action is not None

    def change_pos(self, pos):
        self.position = pos

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        a = self.variables == other.variables
        b = True
        if self.shadow_action is not None and other.shadow_action is not None:
            b = self.shadow_action == other.shadow_action
        return a and b
