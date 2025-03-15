import math


def null(p1, p2):
    """
    Calculate the null heuristic.

    Args:
        p1 (tuple): The coordinates (x, y) of the first point.
        p2 (tuple): The coordinates (x, y) of the second point.

    Returns:
        int: The null heuristic.

    """
    return 0


def manhattan(p1, p2):
    """
    Calculate the Manhattan distance between two points.

    Args:
        p1 (tuple): The coordinates (x, y) of the first point.
        p2 (tuple): The coordinates (x, y) of the second point.

    Returns:
        int: The Manhattan distance between the two points.

    """
    x1, y1 = p1
    x2, y2 = p2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    return dx + dy


def euclidean(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (tuple): The coordinates (x, y) of the first point.
        p2 (tuple): The coordinates (x, y) of the second point.

    Returns:
        float: The Euclidean distance between the two points.

    """
    x1, y1 = p1
    x2, y2 = p2

    dx = (x2 - x1) ** 2
    dy = (y2 - y1) ** 2

    return math.sqrt(dx + dy)


def chebyshev(p1, p2):
    """
    Calculate the Chebyshev distance between two points.

    Args:
        p1 (tuple): The coordinates (x, y) of the first point.
        p2 (tuple): The coordinates (x, y) of the second point.

    Returns:
        int: The Chebyshev distance between the two points.

    """
    x1, y1 = p1
    x2, y2 = p2

    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    return max(dx, dy)


class Heuristic:
    """
    A class to represent various heuristic functions.

    Attributes:
        functions_map (dict of function): A mapping of heuristic function names to their implementations.

    """

    functions_map = {
        "null":      null,
        "manhattan": manhattan,
        "euclidean": euclidean,
        "chebyshev": chebyshev
    }
