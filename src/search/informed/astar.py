from inc.collections.priority_queue import NodePriorityQueue
from inc.types.node                 import Node
from inc.utils.utils                import build_path


def astar_ts(env, heuristic, limit = 1000000):
    strt_pos = env.state_to_position(env.strt_state)
    goal_pos = env.state_to_position(env.goal_state)

    node       = Node(env.strt_state, None, 0, heuristic(strt_pos, goal_pos))
    time_cost  = 1
    space_cost = 1

    if node.state == env.goal_state:
        return build_path(node), time_cost, space_cost

    queue = NodePriorityQueue()
    queue.add(node)

    while not queue.is_empty():
        if time_cost >= limit:
            return [], time_cost, space_cost

        node      = queue.remove()
        path_cost = node.path_cost + 1

        if node.state == env.goal_state:
            return build_path(node), time_cost, space_cost

        for action in range(env.action_space.n):
            state    = env.sample(node.state, action)
            position = env.state_to_position(state)

            child      = Node(state, node, path_cost, path_cost + heuristic(position, goal_pos))
            time_cost += 1

            if child.state not in queue:
                queue.add(child)

            if (child.state in queue) and (queue[child.state].value > child.value):
                queue.replace(child)

        space_cost = max(space_cost, len(queue))

    return None, time_cost, space_cost


def astar_gs(env, heuristic):
    strt_pos = env.state_to_position(env.strt_state)
    goal_pos = env.state_to_position(env.goal_state)

    node       = Node(env.strt_state, None, 0, heuristic(strt_pos, goal_pos))
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
            state    = env.sample(node.state, action)
            position = env.state_to_position(state)

            child      = Node(state, node, path_cost, path_cost + heuristic(position, goal_pos))
            time_cost += 1

            if (child.state not in explored) and (child.state not in queue):
                queue.add(child)

            if (child.state in queue) and (queue[child.state].value > child.value):
                queue.replace(child)

        space_cost = max(space_cost, len(queue) + len(explored))

    return None, time_cost, space_cost
