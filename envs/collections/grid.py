import io
import sys
import numpy as np

from typing import Optional

from gym       import Env, spaces
from gym.utils import seeding

from inc.utils.utils import matrix_to_string


class GridEnv(Env):
    """
    GridEnv is a custom environment for a grid-based reinforcement learning problem.

    The environment is represented as a grid with different cell types, possible actions, and transition dynamics.
    It supports various rendering modes and provides methods to interact with the environment, such as resetting,
    stepping, and converting between states and grid positions.

    The environment is configured with a start state and a goal state, and transitions between states are
    determined by precomputed transition probabilities.

    Attributes:
        metadata (dict): Metadata for rendering modes.
        grid (numpy.ndarray): The grid layout represented as a flattened array.
        rewards (dict): A dictionary mapping grid cell types to their corresponding rewards.
        actions (dict): A dictionary mapping action indices to their corresponding action names.
        terminals (list of str): A list of terminal states represented as grid cell types.
        shape (tuple): The dimensions of the grid (rows, columns).
        rows (signedinteger): The number of rows in the grid.
        cols (signedinteger): The number of columns in the grid.
        strt_state (signedinteger): The index of the starting state.
        goal_state (signedinteger): The index of the goal state.
        action_space (spaces.Discrete): The action space of the environment.
        observation_space (spaces.Discrete): The observation space of the environment.
        T (numpy.ndarray): The transition probability function `T(s, a, s')`.
        R (numpy.ndarray): The rewards function `R(s, a, s')`.
        RS (numpy.ndarray): The reward state function `RS(s)`.
        states_range (list of int): The range of possible states.
        rewards_range (tuple): The range of possible rewards (min, max).
        np_random (numpy.random.Generator): The random number generator.
        curr_state (signedinteger): The index of the current state.
        terminated (bool): Whether the episode has terminated.

    """

    metadata = {
        "render_modes": ["human", "rgb_array"]
    }


    def __init__(self, grid, rewards, actions, dynamics, terminals):
        """
        Initialize the grid environment.

        Args:
            grid (list of list of str): The grid layout represented as a list of lists of strings in 2D.
            rewards (dict): A dictionary mapping grid cell types to their corresponding rewards.
            actions (dict): A dictionary mapping action indices to their corresponding action names.
            dynamics (dict of int and float): A dictionary mapping action indices to a dictionary of transition probabilities.
            terminals (list): A list of terminal states.

        """
        self.grid      = np.asarray(grid).flatten()
        self.rewards   = rewards
        self.actions   = actions
        self.terminals = terminals

        # Define grid dimensions.
        self.shape = (np.shape(grid))
        self.rows  = self.shape[0]
        self.cols  = self.shape[1]

        # Define start and goal states.
        self.strt_state = np.argwhere(self.grid == "S")[0, 0]
        self.goal_state = np.argwhere(self.grid == "G")[0, 0]

        # Define action and observation space.
        self.action_space      = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(len(self.grid))

        functions_spaces = (self.observation_space.n, self.action_space.n, self.observation_space.n)

        # Precompute transition probability function `T`, rewards function `R` and reward state function `RS`.
        self.T  = np.zeros(functions_spaces)
        self.R  = np.zeros(functions_spaces)
        self.RS = np.zeros(self.observation_space.n)

        for state in range(self.observation_space.n):
            if self.grid[state] == "W":
                continue

            if self.is_terminal(state):
                self.T[state, :, state] = 1.0
                continue

            curr_x, curr_y = self.state_to_position(state)

            for action in range(self.action_space.n):
                for prob in dynamics[action]:
                    cases = {
                        0:
                            lambda x, y: (x, max(0, y - 1)),
                        1:
                            lambda x, y: (x, min(self.cols - 1, y + 1)),
                        2:
                            lambda x, y: (max(0, x - 1), y),
                        3:
                            lambda x, y: (min(self.rows - 1, x + 1), y)
                    }

                    next_x, next_y = cases[prob](curr_x, curr_y)
                    next_state     = self.position_to_state(next_x, next_y)

                    if self.grid[next_state] == "W":
                        next_state = state

                    self.T[state, action, next_state] += dynamics[action][prob]
                    self.R[state, action, next_state]  = rewards[self.grid[next_state]]
                    self.RS[state]                     = rewards[self.grid[state]]

                # Normalize probability values over the whole state space.
                self.T[state, action, :] /= np.sum(self.T[state, action, :])

        for state in range(self.observation_space.n):
            if self.grid[state] == "P": self.RS[state] = rewards["P"]
            if self.grid[state] == "G": self.RS[state] = rewards["G"]

        self.states_range  = range(self.observation_space.n)
        self.rewards_range = self.R.min(), self.R.max()
        self.np_random     = None
        self.curr_state    = None
        self.terminated    = False
        self.seed()
        self.reset()


    def is_terminal(self, state):
        """
        Check if the given state is a terminal state.

        Args:
            state (signedinteger): The index of the state to check.

        Returns:
            bool: `True` if the state is a terminal state, `False` otherwise.

        """
        return self.grid[state] in self.terminals


    def seed(self, seed = None):
        """
        Set the seed for the random number generator.

        Args:
            seed (int, optional): The seed value to initialize the random number generator.
                If `None`, a random seed is generated. Defaults to `None`.

        Returns:
            list of int: A list containing the seed value used to initialize the random number generator.

        """
        self.np_random, seed = seeding.np_random(seed)

        return [seed]


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to the starting state.

        Args:
            seed (int, optional): An optional random seed for reproducibility. Defaults to `None`.
            options (dict, dict): Additional options for resetting the environment. Default to `None`.

        Returns:
            tuple: A tuple containing:
                - int: The initial state of the environment;
                - dict: Additional information (empty dictionary in this implementation).

        """
        self.curr_state = self.strt_state
        self.terminated = False

        self.seed(seed)

        return self.curr_state, {}


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

        self.curr_state = next_state

        terminated = self.is_terminal(self.curr_state)
        truncated  = False

        self.terminated = terminated

        return next_state, reward, terminated, truncated, {}


    def state_to_position(self, state):
        """
        Convert a state index to a grid position.

        Args:
            state (signedinteger): The index of the state to convert.

        Returns:
            tuple: A tuple containing:
                - int: representing the row position of the state in the grid;
                - int: representing the column position of the state in the grid.

        """
        row, col = divmod(state, self.cols)

        return int(row), int(col)


    def position_to_state(self, row, col):
        """
        Convert a grid position to a state index.

        Args:
            row (signedinteger): The row index of the position.
            col (signedinteger): The column index of the position.

        Returns:
            int: The corresponding state of the position in the grid.

        """
        return row * self.cols + col


    def sample(self, state, action):
        """
        Sample the next state based on the current state and action.

        Args:
            state (signedinteger): The index of the current state.
            action (signedinteger): The intex of the action to be taken from the current state.

        Returns:
            int: The next state sampled according to the transition probabilities.

        """
        return self.np_random.choice(self.states_range, p = self.T[state, action])


    def render(self, mode = "human"):
        """
        Renders the grid in the specified mode.

        Args:
            mode (str, optional): The rendering mode can be "human" or "ansi". Defaults to "human".

        Returns:
            `None`.

        """
        outfile = io.StringIO() if mode == "ansi" else sys.stdout

        outfile.write(np.array_str(self.grid.reshape(self.rows, self.cols)) + "\n")


    def stats_to_string(self, indent = 0):
        """
        Converts the environment statistics to a formatted string.

        Args:
            indent (int, optional): The number of tabs to indent the output. Defaults to 0.

        Returns:
            str: A formatted string representing the environment statistics, including: the grid layout,
                transition probabilities for each action, and rewards for each state type.

        """
        result = ""
        tab    = "\t" * indent

        curr_state = self.strt_state
        next_state = self.strt_state + 1
        curr_pos   = self.state_to_position(curr_state)
        next_pos   = self.state_to_position(next_state)

        grid = np.array(self.grid).reshape(self.rows, self.cols)

        result += tab + "Grid: \n{}\n".format(matrix_to_string(grid.tolist(), indent + 1))

        for action in range(self.action_space.n):
            result += tab + "Probabilities from {} to {} with action {}: {}\n".format(
                curr_pos, next_pos, self.actions[action], self.T[curr_state, action, next_state]
            )

        result += "\n"

        for reward in self.rewards:
            result += tab + "Reward for states type {}: {}\n".format(reward, self.rewards[reward])

        return result
