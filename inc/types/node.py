class Node:
    """
    A class representing a node in a tree o graph.

    Attributes:
        state (signedinteger): The state represented by the node.
        parent (Node): The parent node.
        path_cost (int): The cost to reach this node from the start node.
        depth_cost (int): The depth of the node in the tree or graph.
        value (int or float): The value associated with the node.
        removed (bool): A flag indicating whether the node has been removed.

    """

    def __init__(self, state, parent = None, path_cost = 0, value = 0):
        """
        Initialize a new node.

        Args:
            state (signedinteger): The state represented by the node.
            parent (Node, optional): The parent node. Defaults to `None`.
            path_cost (int, optional): The cost to reach this node from the start node. Defaults to 0.
            value (int or float, optional): The value associated with the node. Defaults to 0.

        """
        if parent is None:
            self.depth_cost = 0
        else:
            self.depth_cost = parent.depth_cost + 1

        self.state     = state
        self.parent    = parent
        self.path_cost = path_cost
        self.value     = value
        self.removed   = False


    def __hash__(self):
        """
        Calculate the hash value of the node based on its state.

        The hash value is computed by converting the node state to an integer.

        Returns:
            int: The hash value of the node.

        """
        return int(self.state)


    def __lt__(self, other):
        """
        Compare this node with another node based on their values.

        Args:
            other (Node): The other node to compare with.

        Returns:
            bool: `True` if this node value is less than the other node value, `False` otherwise.

        """
        return self.value < other.value
