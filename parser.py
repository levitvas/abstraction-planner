from action import OperatorSas
from state import State


class Parser:
    def __init__(self, lines):
        self.lines = lines
        # Parse unimportant information
        a = lines.pop(0)
        while a != 'begin_metric':
            a = lines.pop(0)
        use_metric = int(lines.pop(0)) == 1
        lines.pop(0)

        # Init var list
        self.variables = []
        self.begin_state = []
        self.end_state = []
        self.end_variables = []
        self.operators = []
        self.mutex_groups = []

        while lines:
            line = lines.pop(0)
            if line == 'begin_variable':
                # Create variable and add it to the list of vars
                self.variables.append(State(make_variable(lines)))
            elif line == 'begin_state':
                # Store begin states from vars
                self.begin_state = State(make_start_state(lines, self.variables))
            elif line == 'begin_goal':
                # Store end goal as states from vars
                self.end_state = State(self.make_end_state(lines, self.variables))
            elif line == 'begin_operator':
                # Create an Action class and add it the actions list
                self.operators.append(OperatorSas(lines, self.variables, None, None, use_metric))
            elif line == 'begin_mutex_group':
                self.mutex_groups.append(make_mutex_group(lines))
            else:
                pass
                # Undefined line, break
                # print("Unknown line, please recheck SAS integrity: {}".format(line))

    def make_end_state(self, lines, variables):
        end_state = []
        num_of_var = lines.pop(0)
        while line := lines.pop(0):
            if line == 'end_goal':
                break
            l = line.split()
            var, atom = int(l[0]), int(l[1])
            self.end_variables.append(var)
            end_state.append(variables[var].variables[atom])

        return end_state

        # ------------ Successful parser


def make_mutex_group(lines):
    amount = lines.pop(0)
    tuples = []
    while line := lines.pop(0):
        if line == 'end_mutex_group':
            break
        split = line.split(' ')
        var_tuple = (int(split[0]), int(split[1]))
        tuples.append(var_tuple)

    return tuples


def make_variable(lines):
    var_name = lines.pop(0)
    mutex = lines.pop(0)
    amount = lines.pop(0)
    atoms = []
    while line := lines.pop(0):
        if line == 'end_variable':
            break
        if line == '<none of those>':
            atoms.append('none')
            continue
        atom = line.split('Atom ')[1]
        atoms.append(atom)

    return atoms


def make_start_state(lines, variables):
    start_state = []
    idx = 0
    while line := lines.pop(0):
        if line == 'end_state':
            break
        start_state.append(variables[idx].variables[int(line)])
        idx += 1

    return start_state
