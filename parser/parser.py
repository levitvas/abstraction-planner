from numpy.core.defchararray import isdigit

from parser.action import OperatorSas
from parser.state import State

class Parser:
    def __init__(self, lines):
        self.lines = lines
        # Parse unimportant information
        a = lines.pop(0)
        while a != 'begin_metric':
            a = lines.pop(0)
        use_metric = int(lines.pop(0)) == 1
        lines.pop(0)
        _var_amount = lines.pop(0)

        # Init var list
        self.variables: list[State] = []
        self.begin_state: State
        self.end_state: State
        self.end_variables: dict[int, int] = {}
        self.operators: list[OperatorSas] = []
        self.mutex_groups: list[list[tuple[int, int]]] = []

        while lines:
            line = lines.pop(0)
            if line == 'begin_variable':
                # Create variable and add it to the list of vars
                atoms, name = self.make_variable(lines)
                self.variables.append(State(atoms, None, name))
            elif line == 'begin_state':
                # Store begin states from vars
                # Contains integers only
                # Ex: [0, 1 , 1]
                self.begin_state = State(self.make_start_state(lines))
            elif line == 'begin_goal':
                # Store end goal as states from vars
                # Contains integers only, -1 if not present
                # Ex: [-1, -1 , 1]
                self.end_state = State(self.make_end_state(lines))
            elif line == 'begin_operator':
                # Creates the operator here directly and adds to list
                self.operators.append(self.make_operator(lines, use_metric))
            elif line == 'begin_mutex_group':
                # Currently unused I think?
                self.mutex_groups.append(self.make_mutex_group(lines))
            elif isdigit(line):
                # Skip line
                continue
            else:
                # Undefined line, break
                print("Unknown line, please recheck SAS integrity: {}".format(line))

        print("~-~ Parsed SAS file successfully ~-~")

    def make_end_state(self, lines) -> list:
        end_state = [-1] * len(self.variables)
        _num_of_vars = lines.pop(0)
        while line := lines.pop(0):
            if line == 'end_goal':
                break
            l = line.split()
            var, atom = int(l[0]), int(l[1])
            self.end_variables[var] = atom
            end_state[var] = atom

        return end_state

    def make_operator(self, lines, costs) -> OperatorSas:
        name = lines.pop(0)
        preconditions = {}
        effects = {}

        prevail_conds = int(lines.pop(0))
        for i in range(prevail_conds):
            line = lines.pop(0)
            l = line.split()
            var, atom = int(l[0]), int(l[1])
            if atom != -1:
                preconditions[var] = atom

        # TODO: Check if value is -1 and if effect is not 0, then add one more precondition
        precond_conds = int(lines.pop(0))
        for i in range(precond_conds):
            line = lines.pop(0)
            l = line.split()
            var, pre, eff = int(l[1]), int(l[2]), int(l[3])
            if pre != -1:
                preconditions[var] = pre
            effects[var] = eff

        cost = int(lines.pop(0))
        if not costs:
            cost = 1
        lines.pop(0)
        return OperatorSas(preconditions, effects, cost, name)

    def make_mutex_group(self, lines):
        _amount = lines.pop(0)
        tuples = []
        while line := lines.pop(0):
            if line == 'end_mutex_group':
                break
            split = line.split(' ')
            var_tuple = (int(split[0]), int(split[1]))
            tuples.append(var_tuple)

            return tuples

    def make_variable(self, lines) -> (list, str):
        var_name = lines.pop(0)
        _mutex = lines.pop(0)
        _amount = lines.pop(0)
        atoms = []
        while line := lines.pop(0):
            if line == 'end_variable':
                break
            if line == '<none of those>':
                atoms.append('none')
                continue
            if "Atom " in line:
                atom = line.split('Atom ')[1]
            if "NegatedAtom " in line:
                atom = "not " + line.split('Atom ')[1]
            atoms.append(atom)

        return atoms, var_name


    def make_start_state(self, lines):
        start_state = []
        while line := lines.pop(0):
            if line == 'end_state':
                break
            start_state.append(int(line))

        return start_state
