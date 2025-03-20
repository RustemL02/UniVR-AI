from inc.collections.priority_queue import NodePriorityQueue
from inc.types.node                 import Node
from inc.utils.utils                import build_path


def ucs_ts(env):
    node       = Node(env.strt_state, None)
    time_cost  = 1
    space_cost = 1

    if node.state == env.goal_state:
        return build_path(node), time_cost, space_cost

    queue = NodePriorityQueue()
    queue.add(node)

    while not queue.is_empty():
        node      = queue.remove()
        path_cost = node.path_cost + 1

        if node.state == env.goal_state:
            return build_path(node), time_cost, space_cost

        for action in range(env.action_space.n):
            child      = Node(env.sample(node.state, action), node, path_cost, path_cost)
            time_cost += 1

            if child.state not in queue:
                queue.add(child)

            if (child.state in queue) and (queue[child.state].path_cost > child.path_cost):
                queue.replace(child)

        space_cost = max(space_cost, len(queue))

    return None, time_cost, space_cost


def ucs_gs(env):
    node       = Node(env.strt_state, None)
    time_cost  = 1
    space_cost = 1

    if node.state == env.goal_state:
        return build_path(node), time_cost, space_cost

    queue    = NodePriorityQueue()
    explored = set()

    queue.add(node)

    while not queue.is_empty():
        node      = queue.remove()
        path_cost = node.path_cost + 1

        if node.state == env.goal_state:
            return build_path(node), time_cost, space_cost

        explored.add(node.state)

        for action in range(env.action_space.n):
            child      = Node(env.sample(node.state, action), node, path_cost, path_cost)
            time_cost += 1

            if (child.state not in explored) and (child.state not in queue):
                queue.add(child)

            if (child.state in queue) and (queue[child.state].path_cost > child.path_cost):
                queue.replace(child)

        space_cost = max(space_cost, len(queue) + len(explored))

    return None, time_cost, space_cost
