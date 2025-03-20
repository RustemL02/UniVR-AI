import numpy as np

from timeit import default_timer as timer


def q_lrn(env, expl_funct, expl_param, alpha = 0.3, gamma = 0.999, episodes = 1000):
    start_time = timer()
    Q          = np.zeros((env.observation_space.n, env.action_space.n))
    rewards    = [0.0 for _ in range(episodes)]
    lengths    = [0   for _ in range(episodes)]

    for episode in range(episodes):
        state      = env.reset()[0]
        terminated = False

        while not terminated:
            action                               = expl_funct(Q, state, expl_param)
            next_state, reward, terminated, _, _ = env.step(action)

            Q[state, action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state, action])

            state = next_state

            rewards[episode] += reward
            lengths[episode] += 1

    policy = Q.argmax(axis = 1)

    final_time = timer() - start_time

    return np.asarray(policy), (rewards, lengths), final_time
