import gym
import numpy as np

from envs import *

from inc.ai.reinforcement_learning import *
from inc.constants.output          import *
from inc.utils.utils               import *

from src.reinforcement_learning.sarsa import sarsa


def print_solution_stats(env, expl_name, sol):
    policy, _, time = sol
    env_string      = env.stats_to_string(1)

    statistics = [
        "Exploration function: {}\n".format(expl_name),
        "Environment: \n{}".format(env_string),
        "Policy: \n{}".format(matrix_to_string(policy, 1)),
        "Execution time: {}".format(time)
    ]

    for statistic in statistics:
        print(statistic)
    print("")


class CheckResult_SARSA:

    def __init__(self, env, solution, terminated, index):
        self.env        = env
        self.solution   = convert_policy(env, solution)
        self.terminated = terminated
        self.index      = index


    @staticmethod
    def check_solution(env, title, expl_name, solution, correct_policy, terminated):
        print_title(title)
        print_solution_stats(env, expl_name, solution)

        policy         = solution[0]
        correspondence = True

        for row in range(len(policy)):
            for col in range(len(policy[row])):
                if (correct_policy[row][col] != "-") and (policy[row][col] != correct_policy[row][col]):
                    correspondence = False
                    break

        if (not correspondence) and (not terminated):
            solution_string = matrix_to_string(correct_policy, 1)

            print(PolicyMessages.NOT_CORRECT_POLICY.format(solution_string))
        else:
            print(GeneralMessages.CORRECT)
        print("")


    def check_solution_env(self):
        title       = "SARSA"
        expl_name   = [
            "Epsilon-Greedy",
            "Softmax",
        ]
        policy_corr = [
            ["R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "D"],
            ["U", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "D"],
            ["U", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "D"],
            ["U", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
        ]

        CheckResult_SARSA.check_solution(
            self.env, title, expl_name[self.index], self.solution, policy_corr, self.terminated
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

            solution   = sarsa(env, expl[0], expl[1])
            terminated = run_episode(env, solution[0])[0]
            index      = explorations.index(expl)

            results = CheckResult_SARSA(env, solution, terminated, index)
            results.check_solution_env()
