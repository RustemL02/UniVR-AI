import gym

from envs import *

from inc.ai.reinforcement_learning import *
from inc.graphics.plot             import *
from inc.utils.utils               import *

from src.markov_decision_processes.pol_iteration import pol_iteration
from src.markov_decision_processes.val_iteration import val_iteration


def print_env_stats(env, env_name):
    env_string = env.stats_to_string(1)

    statistics = [
        "Environment: {}\n{}".format(env_name, env_string)
    ]

    for statistic in statistics:
        print(statistic)
    print("")


def evaluation(env, function, steps = 100, reps = 10, limit = 100):
    x_values = [0.0 for _ in range(limit)]
    y_values = [0.0 for _ in range(limit)]

    for i in range(limit):
        reward = 0
        policy = function(env, limit = i)[0]

        for _ in range(reps):
            reward += run_episode(env, policy, steps)[1]

        x_values[i] = i
        y_values[i] = reward / reps

    return x_values, y_values


class Plot_comparison:

    def __init__(self, env, env_name, index):
        self.env      = env
        self.env_name = env_name
        self.index    = index


    @staticmethod
    def plot(env, title, val_title, pol_title, env_name, steps = 100, reps = 10, limit = 100):
        print_title(title)
        print_env_stats(env, env_name)

        series = []

        val_reward = evaluation(env, val_iteration, steps, reps, limit)
        pol_reward = evaluation(env, pol_iteration, steps, reps, limit)

        series.append({"x": val_reward[0], "y": val_reward[1], "label": val_title})
        series.append({"x": pol_reward[0], "y": pol_reward[1], "label": pol_title})

        plot(series, "Learning rate for {}".format(env_name), "Iterations", "Reward")


    def plot_comparison(self, steps = 100, reps = 10, limit = 100):
        title     = "Comparison between value iteration and policy iteration"
        val_title = "Value iteration"
        pol_title = "Policy iteration"

        Plot_comparison.plot(
            self.env, title, val_title, pol_title, self.env_name, steps, reps, limit
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

            results = Plot_comparison(env, name, name_index)
            results.plot_comparison()
