import numpy as np


def q_value(env, utility, state, action, gamma = 0.9):
    """
    Calculate the expected utility of taking a given action in a given state.

    Args:
        env (gym.core.Env): The environment contains the transition probability function (`T`),
            reward state function (`RS`), and `grid` information.
        utility (list of float): A list of utility values for each state.
        state (signedinteger): The index of the current state.
        action (signedinteger): The intex of the action to be taken from the current state.
        gamma (float, optional): The discount factor for future rewards. Defaults to 0.9.

    Returns:
        float: The expected utility value for the given state and action.

    """
    if env.grid[state] in ("P", "G"):
        return env.RS[state]

    return env.RS[state] + gamma * np.dot(env.T[state, action], utility)


def epsilon_greedy(q_table, state, epsilon = 0.1):
    """
    Selects an action based on the epsilon-greedy policy.

    Args:
        q_table (numpy.ndarray): The Q-table containing the estimated value of each action in each state.
        state (signedinteger): The index of the current state.
        epsilon (float): Probability of exploring (between 0 and 1). Defaults to 0.1.

    Returns:
        int: Index of the selected action.

    """
    if np.random.random() < epsilon:
        return np.random.choice(q_table.shape[1])

    return q_table[state].argmax()


def softmax(q_table, state, temp = 1.0):
    """
    Selects an action based on the softmax policy.

    Args:
        q_table (numpy.ndarray): The Q-table containing the estimated value of each action in each state.
        state (signedinteger): The index of the current state.
        temp (float): Parameter controlling the randomness in action selection. Higher values encourage
            exploration, lower values encourage exploitation. Defaults to 1.0.

    Returns:
        int: Index of the selected action.

    """
    prob  = np.exp(q_table[state] / temp)
    prob /= np.sum(prob)

    return np.random.choice(q_table.shape[1], p = prob)


def values_to_policy(env, values):
    """
    Convert a value function to a policy for a given environment.

    Calculates the expected value of taking each action. The action with the highest expected value is
    selected as the optimal action for that state.

    Args:
        env (gym.core.Env): The environment for which the policy is being generated.
            It should have `observation_space` and `action_space` attributes.
        values (list of float): The value function.

    Returns:
        numpy.ndarray: The policy derived from the value function, where each element represents
            the best action to take in the corresponding state.

    """
    values = np.asarray(values)
    policy = [0 for _ in range(env.observation_space.n)]

    for state in range(env.observation_space.n):
        expected = np.zeros(env.action_space.n)

        for action in range(env.action_space.n):
            for next_state in range(env.observation_space.n):
                expected[action] += env.T[state, action, next_state] * values[next_state]

        expected      = np.round(expected, 6)
        winners       = np.argwhere(expected == np.amax(expected)).flatten()
        policy[state] = int(winners[0])

    return np.asarray(policy)


def convert_policy(env, solution):
    """
    Converts a given policy solution into a format suitable for the environment.

    Args:
        env (gym.core.Env): The environment contains the `actions` and `shape` attributes.
        solution (tuple): A tuple containing:
            - policy (numpy.ndarray): The policy to convert;
            - stats (tuple): The episode statistics;
            - time (float): The time taken to generate the policy.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The policy converted into a format suitable for the environment;
            - tuple: The episode statistics;
            - time: The time taken to generate the policy.

    """
    policy, stats, time = solution

    actions = np.vectorize(env.actions.get)
    policy  = np.reshape(policy, env.shape)

    return np.asarray(actions(policy)), stats, time


def run_episode(env, policy, limit = 1000):
    """
    Run a single episode of the environment using the given policy.

    Simulates a single episode in a reinforcement learning environment, following a given policy.
    The episode terminates when either the environment reaches a terminal state or the step limit is reached.

    Args:
        env (gym.core.Env): The environment to run the episode. It should have `reset` and `step` methods.
        policy (numpy.ndarray): The policy to follow during the episode.
        limit (signedinteger): The maximum amount steps to take in the episode. Defaults to 1000.

    Returns:
        tuple: A tuple containing:
            - bool: Whether the episode terminated before the step limit was reached;
            - float: The total reward accumulated during the episode.

    """
    state, _   = env.reset()
    reward     = 0
    count      = 0
    terminated = False

    while (not terminated) and (count < limit):
        state, next_reward, terminated, _, _ = env.step(policy[state])

        reward += next_reward
        count  += 1

    return terminated, reward
