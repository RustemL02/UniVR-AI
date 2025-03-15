from inc.collections.queue import NodeQueue
from inc.types.node        import Node
from inc.utils.utils       import build_path


def bfs_ts(env):
    node       = Node(env.strt_state, None)
    time_cost  = 1
    space_cost = 1

    if node.state == env.goal_state:
        return build_path(node), time_cost, space_cost

    queue = NodeQueue()
    queue.add(node)

    while not queue.is_empty():
        node = queue.remove()

        for action in range(env.action_space.n):
            child      = Node(env.sample(node.state, action), node)
            time_cost += 1

            if child.state == env.goal_state:
                return build_path(child), time_cost, space_cost

            queue.add(child)

        space_cost = max(space_cost, len(queue))

    return None, time_cost, space_cost


def bfs_gs(env):
    node       = Node(env.strt_state, None)
    time_cost  = 1
    space_cost = 1

    if node.state == env.goal_state:
        return build_path(node), time_cost, space_cost

    queue    = NodeQueue()
    explored = set()

    queue.add(node)

    while not queue.is_empty():
        node = queue.remove()

        explored.add(node.state)

        for action in range(env.action_space.n):
            child      = Node(env.sample(node.state, action), node)
            time_cost += 1

            if (child.state not in explored) and (child.state not in queue):
                if child.state == env.goal_state:
                    return build_path(child), time_cost, space_cost

                queue.add(child)

        space_cost = max(space_cost, len(queue) + len(explored))

    return None, time_cost, space_cost
