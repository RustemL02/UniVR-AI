from inc.types.node  import Node
from inc.utils.utils import build_path


def dls_ts(env, node, limit = 1000000):
    total_time_cost  = 1
    total_space_cost = node.depth_cost
    finished         = False

    if node.state == env.goal_state:
        return build_path(node), total_time_cost, total_space_cost

    if limit == 0:
        return [], total_time_cost, total_space_cost

    for action in range(env.action_space.n):
        child                           = Node(env.sample(node.state, action), node)
        solution, time_cost, space_cost = dls_ts(env, child, limit - 1)

        total_time_cost  += time_cost
        total_space_cost  = max(total_space_cost, space_cost)

        if solution is not None:
            finished = True if len(solution) == 0 else False

            if len(solution) > 0:
                return solution, total_time_cost, total_space_cost

    if finished:
        return [], total_time_cost, total_space_cost

    return None, total_time_cost, total_space_cost


def dls_gs(env, explored, node, limit = 1000000):
    total_time_cost  = 1
    total_space_cost = node.depth_cost
    finished         = False

    if node.state == env.goal_state:
        return build_path(node), total_time_cost, total_space_cost

    if limit == 0:
        return [], total_time_cost, total_space_cost

    explored.add(node.state)

    for action in range(env.action_space.n):
        child = Node(env.sample(node.state, action), node)

        if child.state not in explored:
            solution, time_cost, space_cost = dls_gs(env, explored, child, limit - 1)

            total_time_cost  += time_cost
            total_space_cost  = max(total_space_cost, space_cost)

            if solution is not None:
                finished = True if len(solution) == 0 else False

                if len(solution) > 0:
                    return solution, total_time_cost, total_space_cost

    if finished:
        return [], total_time_cost, total_space_cost

    return None, total_time_cost, total_space_cost
