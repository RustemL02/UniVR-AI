import numpy as np

from timeit import default_timer as timer

from inc.ai.reinforcement_learning import q_value


def pol_evaluation(env, U, policy, gamma = 0.9, max_error = 1e-3, limit = 1000):
    for _ in range(limit):
        U_i   = U.copy()
        delta = 0

        for state in range(env.observation_space.n):
            U[state] = q_value(env, U_i, state, policy[state], gamma)
            delta    = max(delta, abs(U[state] - U_i[state]))

        if delta < (max_error * (1 - gamma) / gamma):
            break

    return U


def pol_iteration(env, gamma = 0.9, max_error = 1e-3, limit = 1000):
    start_time = timer()
    U          = [0.0 for _ in range(env.observation_space.n)]
    policy     = [0   for _ in range(env.observation_space.n)]

    for _ in range(limit):
        U         = pol_evaluation(env, U, policy, gamma, max_error, limit)
        unchanged = True

        for state in range(env.observation_space.n):
            best_action = int(np.argmax(
                [q_value(env, U, state, action, gamma) for action in range(env.action_space.n)]
            ))

            best_utility = q_value(env, U, state, best_action, gamma)
            curr_utility = q_value(env, U, state, policy[state], gamma)

            if best_utility > curr_utility:
                policy[state] = best_action
                unchanged     = False

        if unchanged:
            break

    final_time = timer() - start_time

    return np.asarray(policy), (), final_time
