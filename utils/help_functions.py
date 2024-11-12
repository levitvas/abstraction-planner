from parser.parser import Parser


def issubset(subset: frozenset[tuple[int, int]], superset: frozenset[tuple[int, int]]) -> bool:
    for (pre, eff) in subset:
        if (pre, eff) not in superset:
            return False

    return True
