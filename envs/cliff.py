from envs.collections.grid import GridEnv


class CliffEnv(GridEnv):

    def __init__(self):
        grid = [
            ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E"],
            ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E"],
            ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E"],
            ["S", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "G"]
        ]

        rewards = {
            "S": -1.0,
            "E": -1.0,
            "C": -100.0,
            "G":  1.0,
        }

        actions = {
            0: "L",
            1: "R",
            2: "U",
            3: "D"
        }

        dynamics = {
            0: {0: 1.0},
            1: {1: 1.0},
            2: {2: 1.0},
            3: {3: 1.0}
        }

        terminals = [
            "G"
        ]

        super().__init__(grid, rewards, actions, dynamics, terminals)


    def step(self, action):
        """
        Perform a single step in the environment based on the given action.

        Args:
            action (signedinteger): The intex of the action to be taken.

        Returns:
            tuple: A tuple containing:
                - int: The next state after taking the action;
                - float: The reward received after taking the action;
                - bool: Whether the episode has terminated;
                - bool: Whether the episode has been truncated (always `False` in this implementation);
                - dict: Additional information (empty dictionary in this implementation).

        """
        if self.terminated:
            return None

        next_state = self.sample(self.curr_state, action)
        reward     = self.R[self.curr_state, action, next_state]

        if self.grid[next_state] == "C":
            next_state = self.strt_state

        self.curr_state = next_state

        terminated = self.is_terminal(self.curr_state)
        truncated  = False

        self.terminated = terminated

        return next_state, reward, terminated, truncated, {}
