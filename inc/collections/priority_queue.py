import heapq


class NodePriorityQueue:
    """
    A priority queue implementation for nodes, using a heap and a dictionary for fast access.

    Attributes:
        queue (list of Node): The heap-based priority queue.
        node_dict (dict): A dictionary to store nodes with their state as the key for quick access.
        length (signedinteger): The current length of the priority queue.

    """

    def __init__(self):
        """
        Initializes a new instance of the priority queue.

        """
        self.queue     = []
        self.node_dict = {}
        self.length    = 0


    def is_empty(self):
        """
        Check if the priority queue is empty.

        Returns:
            bool: `True` if the priority queue is empty, `False` otherwise.

        """
        return self.length == 0


    def add(self, node):
        """
        Add a node to the priority queue.

        Args:
            node (Node): The node to be added to the priority queue. It is assumed that node has a `state` attribute.

        """
        heapq.heappush(self.queue, node)

        self.node_dict[node.state]  = node
        self.length                += 1


    def remove(self):
        """
        Removes and returns the highest priority node from the priority queue.

        Returns:
            Node: The highest priority node that has not been marked as removed.

        Raises:
            IndexError: If the priority queue is empty.

        """
        while True:
            node = heapq.heappop(self.queue)

            if not node.removed:
                if node.state in self.node_dict:
                    del self.node_dict[node.state]

                self.length -= 1

                return node


    def replace(self, node):
        """
        Replace an existing node in the priority queue with a new node.

        Args:
            node (Node): The new node to replace the existing node with. The node should have a `state` attribute.

        """
        self.node_dict[node.state].removed = True
        self.node_dict[node.state]         = node

        self.length -= 1

        self.add(node)


    def __len__(self):
        """
        Return the amount nodes in the priority queue.

        Returns:
            int: The amount nodes in the priority queue.

        """
        return self.length


    def __contains__(self, node):
        """
        Check if a node is in the priority queue.

        Args:
            node (Node): The node to check for membership in the priority queue.

        Returns:
            bool: `True` if the node is in the priority queue, `False` otherwise.

        """
        return node in self.node_dict


    def __getitem__(self, node):
        """
        Retrieve the value associated with the given node from the priority queue.

        Args:
            node (Node): The key for which the value needs to be retrieved.

        Returns:
            Node: The value associated with the given key.

        Raises:
            KeyError: If the node is not present in the priority queue.

        """
        return self.node_dict[node]
