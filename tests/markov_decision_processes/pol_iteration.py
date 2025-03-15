import gym
import numpy as np

from envs import *

from inc.ai.reinforcement_learning import *
from inc.constants.output          import *
from inc.utils.utils               import *

from src.markov_decision_processes.pol_iteration import pol_iteration


def print_solution_stats(env, env_name, sol):
    policy, _, time = sol
    env_string      = env.stats_to_string(1)

    statistics = [
        "Environment: {}\n{}".format(env_name, env_string),
        "Policy: \n{}".format(matrix_to_string(policy, 1)),
        "Execution time: {}".format(time)
    ]

    for statistic in statistics:
        print(statistic)
    print("")


class CheckResult_pol_iteration:

    def __init__(self, env, solution, env_name, index):
        self.env      = env
        self.solution = convert_policy(env, solution)
        self.env_name = env_name
        self.index    = index


    @staticmethod
    def check_solution(env, title, env_name, solution, correct_policy):
        print_title(title)
        print_solution_stats(env, env_name, solution)

        policy = solution[0]

        if not np.all(policy == correct_policy):
            solution_string = matrix_to_string(correct_policy, 1)

            print(PolicyMessages.NOT_CORRECT_POLICY.format(solution_string))
        else:
            print(GeneralMessages.CORRECT)
        print("")


    def check_solution_env(self):
        title         = "Policy Iteration"
        policies_corr = [
            [
                ["D", "L", "L", "U"],
                ["D", "L", "L", "L"],
                ["R", "R", "R", "L"]
            ],
            [
                ["R", "R", "R", "D"],
                ["D", "L", "R", "L"],
                ["R", "R", "R", "L"]
            ],
            [
                ["D", "L", "L", "U"],
                ["D", "L", "L", "L"],
                ["R", "R", "L", "L"]
            ],
            [
                ["D", "D", "L", "L", "L", "U", "R", "D", "L", "L"],
                ["D", "D", "L", "L", "L", "L", "R", "D", "L", "L"],
                ["D", "L", "U", "L", "U", "D", "L", "D", "U", "D"],
                ["D", "L", "L", "L", "U", "L", "L", "D", "L", "D"],
                ["D", "L", "D", "L", "U", "L", "L", "D", "D", "L"],
                ["D", "L", "U", "L", "L", "L", "L", "D", "D", "U"],
                ["D", "L", "L", "R", "D", "R", "R", "D", "L", "L"],
                ["R", "D", "D", "D", "L", "L", "R", "D", "D", "D"],
                ["U", "R", "R", "R", "D", "D", "R", "R", "D", "D"],
                ["L", "R", "R", "R", "R", "R", "R", "R", "R", "L"]
            ]
        ]

        CheckResult_pol_iteration.check_solution(
            self.env, title, self.env_name, self.solution, policies_corr[self.index]
        )


class Main:
    if __name__ == "__main__":
        env_names = [
            LAVA_FLOOR,
            VERY_BAD_LAVA_FLOOR,
            NICE_LAVA_FLOOR,
            HUGE_LAVA_FLOOR
        ]

        for name in env_names:
            env        = gym.make(name)
            name_index = env_names.index(name)

            solution = pol_iteration(env)

            results = CheckResult_pol_iteration(env, solution, name, name_index)
            results.check_solution_env()
