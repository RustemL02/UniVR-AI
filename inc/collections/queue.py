from collections import deque


class NodeQueue:
    """
    A queue implementation for nodes using deque and a dictionary for fast access.

    Attributes:
        queue (deque): A deque to store the nodes in the queue.
        node_dict (dict): A dictionary to store nodes with their state as the key for quick access.
        length (signedinteger): The current length of the queue.

    """

    def __init__(self):
        """
        Initializes a new instance of the queue.

        """
        self.queue     = deque()
        self.node_dict = {}
        self.length    = 0


    def is_empty(self):
        """
        Check if the queue is empty.

        Returns:
            bool: `True` if the queue is empty, `False` otherwise.

        """
        return self.length == 0


    def add(self, node):
        """
        Adds a node to the queue.

        Args:
            node (Node): The node to be added to the queue. It is expected to have a `state` attribute.

        """
        self.queue.append(node)

        self.node_dict[node.state]  = node
        self.length                += 1


    def remove(self):
        """
        Remove and return the first node from the queue.

        Returns:
            Node: The node that was removed from the queue.

        """
        node = self.queue.popleft()

        if node in self.node_dict:
            del self.node_dict[node.state]

        self.length -= 1

        return node


    def __len__(self):
        """
        Return the amount nodes in the queue.

        Returns:
            int: The amount nodes in the queue.

        """
        return self.length


    def __contains__(self, node):
        """
        Check if a node is in the queue.

        Args:
            node (Node): The node to check for membership in the queue.

        Returns:
            bool: `True` if the node is in the queue, `False` otherwise.

        """
        return node in self.node_dict


    def __getitem__(self, node):
        """
        Retrieve the value associated with the given node from the queue.

        Args:
            node (Node): The key for which the value needs to be retrieved.

        Returns:
            Node: The node associated with the given key.

        Raises:
            KeyError: If the node is not found in the queue.

        """
        return self.node_dict[node]
