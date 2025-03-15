import gym
import numpy as np

from envs import *

from inc.ai.reinforcement_learning import *
from inc.graphics.plot             import *
from inc.utils.utils               import *

from src.reinforcement_learning.q_lrn import q_lrn
from src.reinforcement_learning.sarsa import sarsa


def print_solution_stats(env, sol):
    policy, _, time = convert_policy(env, sol)
    env_string      = env.stats_to_string(1)

    statistics = [
        "Environment: \n{}".format(env_string),
        "Policy: \n{}".format(matrix_to_string(policy, 1)),
        "Execution time: {}".format(time)
    ]

    for statistic in statistics:
        print(statistic)
    print("")


def evaluation(env, function, expl_funct, expl_param, window = 100):
    policy, result, time = function(env, expl_funct, expl_param)

    print_solution_stats(env, (policy, result, time))

    y_rewards = rolling(result[0], window)
    y_lengths = rolling(result[1], window)
    x_rewards = np.arange(1, len(y_rewards) + 1)
    x_lengths = np.arange(1, len(y_lengths) + 1)

    return (x_rewards, y_rewards), (x_lengths, y_lengths)


class Plot_comparison:

    def __init__(self, env, expl_funct, expl_param):
        self.env        = env
        self.expl_funct = expl_funct
        self.expl_param = expl_param


    @staticmethod
    def plot(env, title, q_lrn_title, sarsa_title, expl_funct, expl_param, window = 100):
        print_title(title)

        series_rew = []
        series_len = []

        q_lrn_reward, q_lrn_length = evaluation(env, q_lrn, expl_funct, expl_param, window)
        sarsa_reward, sarsa_length = evaluation(env, sarsa, expl_funct, expl_param, window)

        series_rew.append({"x": q_lrn_reward[0], "y": q_lrn_reward[1], "label": q_lrn_title})
        series_len.append({"x": q_lrn_length[0], "y": q_lrn_length[1], "label": q_lrn_title})
        series_rew.append({"x": sarsa_reward[0], "y": sarsa_reward[1], "label": sarsa_title})
        series_len.append({"x": sarsa_length[0], "y": sarsa_length[1], "label": sarsa_title})

        plot(series_rew, "Rewards per episodes", "Episodes", "Rewards")
        plot(series_len, "Lengths per episodes", "Episodes", "Lengths")


    def plot_comparison(self, window = 100):
        title       = "Comparison between Q-Learning and Sarsa"
        q_lrn_title = "Q-Learning"
        sarsa_title = "Sarsa"

        Plot_comparison.plot(
            self.env, title, q_lrn_title, sarsa_title, self.expl_funct, self.expl_param, window
        )


class Main:
    if __name__ == "__main__":
        explorations = [
            [
                epsilon_greedy,
                0.1
            ],
            [
                softmax,
                1.0
            ],
        ]

        for expl in explorations:
            env = gym.make(CLIFF)

            results = Plot_comparison(env, expl[0], expl[1])
            results.plot_comparison()
