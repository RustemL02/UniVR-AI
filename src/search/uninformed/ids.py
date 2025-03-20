from inc.types.node  import Node

from src.search.uninformed.dls import dls_ts, dls_gs


def ids_ts(env):
    iterations       = 0
    total_time_cost  = 0
    total_space_cost = 1

    for iterations in range(0, len(env.grid)):
        solution, time_cost, space_cost = dls_ts(env, Node(env.strt_state, None), iterations)

        total_time_cost  += time_cost
        total_space_cost  = space_cost

        if (solution is None) or (len(solution) > 0):
            return solution, total_time_cost, total_space_cost, iterations

    return [], total_time_cost, total_space_cost, iterations


def ids_gs(env):
    iterations       = 0
    total_time_cost  = 0
    total_space_cost = 1

    for iterations in range(0, len(env.grid)):
        solution, time_cost, space_cost = dls_gs(env, set(), Node(env.strt_state, None), iterations)

        total_time_cost  += time_cost
        total_space_cost  = space_cost

        if (solution is None) or (len(solution) > 0):
            return solution, total_time_cost, total_space_cost, iterations

    return [], total_time_cost, total_space_cost, iterations
