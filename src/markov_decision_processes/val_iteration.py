from timeit import default_timer as timer

from inc.ai.reinforcement_learning import q_value, values_to_policy


def val_iteration(env, gamma = 0.9, max_error = 1e-3, limit = 1000):
    start_time = timer()
    U          = [0.0 for _ in range(env.observation_space.n)]
    U_1        = [0.0 for _ in range(env.observation_space.n)]

    for _ in range(limit):
        U     = U_1.copy()
        delta = 0

        for state in range(env.observation_space.n):
            expected_utilities = [q_value(env, U, state, action, gamma) for action in range(env.action_space.n)]
            U_1[state]         = max(expected_utilities)
            delta              = max(delta, abs(U[state] - U_1[state]))

        if delta < (max_error * (1 - gamma) / gamma):
            break

    final_time = timer() - start_time

    return values_to_policy(env, U), (), final_time
