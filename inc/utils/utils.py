import numpy as np

from inc.constants.output import TITLE


def rolling(array, window):
    """
    Compute the rolling average of a given array with a specified window size.

    Args:
        array (numpy.ndarray): The input array for which the rolling average is to be computed.
        window (int): The size of the window over which the average is computed.

    Returns:
        numpy.ndarray: An array containing the rolling averages.

    """
    return np.convolve(array, np.ones(window), mode = "valid") / window


def build_path(node):
    """
    Constructs the path from the given node to the root node.

    This function traverses from the given node up to the root node,
    collecting the states of each node along the way. The path is
    returned in the order from the root node to the given node.

    Args:
        node (Node): The starting node from which to build the path. It is assumed that each node has a `parent`
            attribute pointing to its parent node and a `state` attribute representing the state of the node.

    Returns:
        tuple: A tuple containing the states from the root node to the given node in order.

    """
    path = []

    while node.parent is not None:
        path.append(node.state)

        node = node.parent

    return tuple(reversed(path))


def print_title(title, frame = "#", size = 220):
    """
    Prints a formatted title centered within a line of separators.

    The title will be centered within a line of `size` characters, padded with spaces, and surrounded
    by lines of `frame` characters.

    Args:
        title (str): The title to be printed.
        frame (str, optional): The formatted character to be used as the frame. Defaults to "#".
        size (int, optional): The size of the terminal window. Defaults to 220.

    """
    spaces        = " " * ((size - len(title)) // 2)
    separators    = frame * size

    title = separators + "\n{}\n".format(spaces + title + spaces) + separators + "\n"

    print(TITLE.substitute(msg = title))


def matrix_to_string(matrix, indent = 0):
    """
    Converts a 2D matrix into a formatted string representation with optional indentation.

    Args:
        matrix (list of list): The 2D matrix to be converted to a string.
        indent (int, optional): The amount tab indentations to apply to the entire matrix. Defaults to 0.

    Returns:
        str: A string representation of the matrix with each element aligned and rows separated by newlines.

    """
    string = ""
    tab    = "\t" * indent

    for row in matrix:
        string += "\t" + tab + " ".join(f"{elem:<4}" for elem in row) + "\n"

    return tab + "[\n" + string + tab + "]\n"


def solution_to_string(env, sol):
    """
    Converts a solution tuple to a list of positions using the environment `state_to_position` method.

    Args:
        env (gym.core.Env): An environment object that has a `state_to_position` method.
        sol (tuple): A tuple representing the solution, where each element is a state.

    Returns:
        list of tuple: A list of positions corresponding to the states in the solution tuple if sol is not `None`
            and is a tuple instance. Otherwise, returns the original solution.

    """
    if (sol is not None) and (isinstance(sol, tuple)):
        return [env.state_to_position(state) for state in sol]

    return sol
