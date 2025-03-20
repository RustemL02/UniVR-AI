import gym

from envs import *

from inc.constants.output import *
from inc.utils.utils      import *

from src.search.uninformed.ids import ids_ts, ids_gs


def print_solution_stats(env, sol):
    path, time_cost, space_cost, iterations = sol

    statistics = [
        "Solution: {}".format(solution_to_string(env, path)),
        "N° of nodes explored: {}".format(time_cost),
        "Max n° of nodes in memory: {}".format(space_cost),
        "Iterations: {}".format(iterations)
    ]

    for statistic in statistics:
        print(statistic)
    print("")


class CheckResult_IDS:

    def __init__(self, env, solution_ts, solution_gs):
        self.env         = env
        self.solution_ts = solution_ts
        self.solution_gs = solution_gs


    @staticmethod
    def check_solution(env, title, solution, correct_values):
        print_title(title)
        print_solution_stats(env, solution)

        path,      time_cost, space_cost, iterations      = solution
        path_corr, time_corr, space_corr, iterations_corr = correct_values
        path                                              = solution_to_string(env, path)

        checks = [
            (path,       path_corr,       SearchMessages.NOT_CORRECT_SOLUTION),
            (time_cost,  time_corr,       SearchMessages.NOT_CORRECT_TIME_COST),
            (space_cost, space_corr,      SearchMessages.NOT_CORRECT_SPACE_COST),
            (iterations, iterations_corr, SearchMessages.NOT_CORRECT_ITERATIONS)
        ]

        for value, value_corr, message in checks:
            if value != value_corr:
                print(message.format(value_corr))
                break
        else:
            print(GeneralMessages.CORRECT)
        print("\n")


    def check_solution_ts(self):
        title           = "Iterative Deepening Depth First Search (tree search)"
        path_corr       = [(0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]
        time_corr       = 138298
        space_corr      = 9
        iterations_corr = 9

        CheckResult_IDS.check_solution(
            self.env, title, self.solution_ts, (path_corr, time_corr, space_corr, iterations_corr)
        )


    def check_solution_gs(self):
        title           = "Iterative Deepening Depth First Search (graph search)"
        path_corr       = [(0, 1), (0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]
        time_corr       = 132
        space_corr      = 11
        iterations_corr = 11

        CheckResult_IDS.check_solution(
            self.env, title, self.solution_gs, (path_corr, time_corr, space_corr, iterations_corr)
        )


class Main:
    if __name__ == "__main__":
        env = gym.make(SMALL_MAZE)

        solution_ts = ids_ts(env)
        solution_gs = ids_gs(env)

        results = CheckResult_IDS(env, solution_ts, solution_gs)
        results.check_solution_ts()
        results.check_solution_gs()
