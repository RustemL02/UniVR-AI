import gym

from envs import *

from inc.constants.output import *
from inc.utils.heuristic  import *
from inc.utils.utils      import *

from src.search.informed.astar import astar_ts, astar_gs


def print_solution_stats(env, sol):
    path, time_cost, space_cost, heuristic = sol

    statistics = [
        "Solution: {}".format(solution_to_string(env, path)),
        "N° of nodes explored: {}".format(time_cost),
        "Max n° of nodes in memory: {}".format(space_cost),
        "Heuristic: {}".format(heuristic)
    ]

    for statistic in statistics:
        print(statistic)
    print("")


class CheckResult_AStar:

    def __init__(self, env, solution_ts, solution_gs, heuristic):
        self.env         = env
        self.solution_ts = solution_ts
        self.solution_gs = solution_gs
        self.heuristic   = heuristic


    @staticmethod
    def check_solution(env, title, solution, correct_values):
        print_title(title)
        print_solution_stats(env, solution)

        path,      time_cost, space_cost, heuristic = solution
        path_corr, time_corr, space_corr            = correct_values
        path                                        = solution_to_string(env, path)

        checks = [
            (path,       path_corr,  SearchMessages.NOT_CORRECT_SOLUTION),
            (time_cost,  time_corr,  SearchMessages.NOT_CORRECT_TIME_COST),
            (space_cost, space_corr, SearchMessages.NOT_CORRECT_SPACE_COST)
        ]

        for value, value_corr, message in checks:
            if value != value_corr:
                print(message.format(value_corr))
                break
        else:
            print(GeneralMessages.CORRECT)
        print("\n")


    def check_solution_ts(self):
        title      = "A* Search (tree search)"
        path_corr  = [(0, 1), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]
        time_corr  = [337, 285, 197, 285]
        space_corr = 16

        index = list(Heuristic.functions_map.keys()).index(self.heuristic)

        CheckResult_AStar.check_solution(
            self.env, title, self.solution_ts, (path_corr, time_corr[index], space_corr)
        )


    def check_solution_gs(self):
        title      = "A* Search (graph search)"
        path_corr  = [
            [(0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)],
            [(0, 1), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)],
            [(0, 1), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)],
            [(0, 1), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]
        ]
        time_corr  = 61
        space_corr = 16

        index = list(Heuristic.functions_map.keys()).index(self.heuristic)

        CheckResult_AStar.check_solution(
            self.env, title, self.solution_gs, (path_corr[index], time_corr, space_corr)
        )


class Main:
    if __name__ == "__main__":
        env = gym.make(SMALL_MAZE)

        for heuristic in Heuristic.functions_map.keys():
            solution_ts = astar_ts(env, Heuristic.functions_map[heuristic]) + (heuristic,)
            solution_gs = astar_gs(env, Heuristic.functions_map[heuristic]) + (heuristic,)

            results = CheckResult_AStar(env, solution_ts, solution_gs, heuristic)
            results.check_solution_ts()
            results.check_solution_gs()
