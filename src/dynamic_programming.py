#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang (zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada (hyperkentakun@gmail.com)                         #
# 2024 Rishikesh                                                      #
#                                                                     #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from .environment import GridWorldEnv


def get_uniform_policy(env: GridWorldEnv) -> np.ndarray:
    """Return a uniform random policy array of shape (WORLD_SIZE, WORLD_SIZE, NUM_ACTIONS)."""
    prob = 1.0 / env.NUM_ACTIONS
    return np.full((env.WORLD_SIZE, env.WORLD_SIZE, env.NUM_ACTIONS), prob)


def get_greedy_actions(env: GridWorldEnv, values: np.ndarray) -> dict:
    """
    Find best actions for each state with tie-breaking.

    Args:
        env: GridWorldEnv instance
        values: (WORLD_SIZE, WORLD_SIZE) array

    Returns:
        dict mapping (i, j) -> list of optimal action indices
    """
    greedy_actions = {}
    for i in range(env.WORLD_SIZE):
        for j in range(env.WORLD_SIZE):
            q_values = []
            for a_idx in range(env.NUM_ACTIONS):
                P_mat, reward = env.get_transitions([i, j], a_idx)
                expected_future = np.sum(P_mat * values)
                q = reward + env.DISCOUNT * expected_future
                q_values.append(q)
            q_values = np.array(q_values)
            # Find indices matching max q-value (within numerical tolerance)
            max_q = np.max(q_values)
            best = np.where(np.abs(q_values - max_q) < 1e-8)[0].tolist()
            greedy_actions[(i, j)] = best
    return greedy_actions


def get_epsilon_greedy_policy(
    env: GridWorldEnv, values: np.ndarray, epsilon: float = 0.0
) -> np.ndarray:
    """
    Compute epsilon-greedy policy with respect to state-value vector V.

    For state s:
        If a in A*(s) (optimal actions):
            pi(a|s) = (1 - epsilon) / |A*| + epsilon / |A|
        Otherwise:
            pi(a|s) = epsilon / |A|

    Args:
        env: GridWorldEnv instance
        values: (WORLD_SIZE, WORLD_SIZE) array
        epsilon: exploration factor in [0.0, 1.0]

    Returns:
        policy: (WORLD_SIZE, WORLD_SIZE, NUM_ACTIONS) array
    """
    policy = np.zeros((env.WORLD_SIZE, env.WORLD_SIZE, env.NUM_ACTIONS))
    greedy_actions = get_greedy_actions(env, values)

    for i in range(env.WORLD_SIZE):
        for j in range(env.WORLD_SIZE):
            best = greedy_actions[(i, j)]
            num_best = len(best)
            for a in range(env.NUM_ACTIONS):
                if a in best:
                    policy[i, j, a] = (
                        (1.0 - epsilon) / num_best
                    ) + epsilon / env.NUM_ACTIONS
                else:
                    policy[i, j, a] = epsilon / env.NUM_ACTIONS
    return policy


def evaluate_policy_linear(
    env: GridWorldEnv, policy: np.ndarray = None
) -> np.ndarray:
    """
    Exact closed-form policy evaluation via linear system of equations:
    (I - gamma * P_pi) * V = R_pi

    Args:
        env: GridWorldEnv instance
        policy: (WORLD_SIZE, WORLD_SIZE, NUM_ACTIONS) array. If None, uniform policy is used.

    Returns:
        values: (WORLD_SIZE, WORLD_SIZE) exact state-value matrix
    """
    if policy is None:
        policy = get_uniform_policy(env)

    num_states = env.WORLD_SIZE * env.WORLD_SIZE
    A = -1.0 * np.eye(num_states)
    b = np.zeros(num_states)

    for i in range(env.WORLD_SIZE):
        for j in range(env.WORLD_SIZE):
            s = [i, j]
            idx_s = env.state_to_index(s)
            for a_idx in range(env.NUM_ACTIONS):
                action_prob = policy[i, j, a_idx]
                if action_prob <= 0:
                    continue
                P_mat, reward = env.get_transitions(s, a_idx)
                for (next_i, next_j), prob in np.ndenumerate(P_mat):
                    if prob > 0:
                        idx_next = env.state_to_index([next_i, next_j])
                        A[idx_s, idx_next] += (
                            action_prob * prob * env.DISCOUNT
                        )
                b[idx_s] -= action_prob * reward

    values = np.linalg.solve(A, b)
    return values.reshape(env.WORLD_SIZE, env.WORLD_SIZE)


def evaluate_policy_iterative(
    env: GridWorldEnv,
    policy: np.ndarray = None,
    tol: float = 1e-4,
    max_iter: int = 10000,
) -> np.ndarray:
    """
    Iterative policy evaluation using Bellman expectation backup:
    V_{k+1}(s) = sum_a pi(a|s) [ R(s, a) + gamma * sum_s' P(s'|s, a) V_k(s') ]

    Args:
        env: GridWorldEnv instance
        policy: (WORLD_SIZE, WORLD_SIZE, NUM_ACTIONS) array. If None, uniform policy is used.
        tol: convergence threshold for max absolute difference
        max_iter: maximum number of iterations

    Returns:
        values: (WORLD_SIZE, WORLD_SIZE) evaluated state-value matrix
    """
    if policy is None:
        policy = get_uniform_policy(env)

    values = np.zeros((env.WORLD_SIZE, env.WORLD_SIZE))
    for _ in range(max_iter):
        new_values = np.zeros_like(values)
        for i in range(env.WORLD_SIZE):
            for j in range(env.WORLD_SIZE):
                s = [i, j]
                v = 0.0
                for a_idx in range(env.NUM_ACTIONS):
                    action_prob = policy[i, j, a_idx]
                    if action_prob <= 0:
                        continue
                    P_mat, reward = env.get_transitions(s, a_idx)
                    expected_future = np.sum(P_mat * values)
                    v += action_prob * (reward + env.DISCOUNT * expected_future)
                new_values[i, j] = v

        if np.max(np.abs(values - new_values)) < tol:
            return new_values
        values = new_values

    return values


def value_iteration(
    env: GridWorldEnv, tol: float = 1e-4, max_iter: int = 10000
) -> tuple:
    """
    Value Iteration using Bellman optimality backup:
    V_{k+1}(s) = max_a [ R(s, a) + gamma * sum_s' P(s'|s, a) V_k(s') ]

    Args:
        env: GridWorldEnv instance
        tol: convergence threshold
        max_iter: maximum iterations

    Returns:
        values: (WORLD_SIZE, WORLD_SIZE) optimal state-value matrix
        policy: (WORLD_SIZE, WORLD_SIZE, NUM_ACTIONS) optimal greedy policy
    """
    values = np.zeros((env.WORLD_SIZE, env.WORLD_SIZE))
    for _ in range(max_iter):
        new_values = np.zeros_like(values)
        for i in range(env.WORLD_SIZE):
            for j in range(env.WORLD_SIZE):
                s = [i, j]
                q_candidates = []
                for a_idx in range(env.NUM_ACTIONS):
                    P_mat, reward = env.get_transitions(s, a_idx)
                    expected_future = np.sum(P_mat * values)
                    q_candidates.append(
                        reward + env.DISCOUNT * expected_future
                    )
                new_values[i, j] = np.max(q_candidates)

        if np.max(np.abs(values - new_values)) < tol:
            values = new_values
            break
        values = new_values

    optimal_policy = get_epsilon_greedy_policy(env, values, epsilon=0.0)
    return values, optimal_policy


def policy_iteration(
    env: GridWorldEnv,
    epsilon: float = 0.0,
    tol: float = 1e-4,
    max_iter: int = 10000,
) -> tuple:
    """
    Policy Iteration with epsilon-greedy exploration:
    Alternates policy evaluation and epsilon-greedy policy improvement until convergence.

    Args:
        env: GridWorldEnv instance
        epsilon: exploration factor (0.0 for pure greedy, 0.2 for exploratory)
        tol: convergence threshold
        max_iter: maximum iterations

    Returns:
        values: (WORLD_SIZE, WORLD_SIZE) converged state-value matrix
        policy: (WORLD_SIZE, WORLD_SIZE, NUM_ACTIONS) converged policy
    """
    values = np.zeros((env.WORLD_SIZE, env.WORLD_SIZE))

    for _ in range(max_iter):
        policy = get_epsilon_greedy_policy(env, values, epsilon)
        new_values = np.zeros_like(values)

        for i in range(env.WORLD_SIZE):
            for j in range(env.WORLD_SIZE):
                s = [i, j]
                v = 0.0
                for a_idx in range(env.NUM_ACTIONS):
                    prob = policy[i, j, a_idx]
                    if prob <= 0:
                        continue
                    P_mat, reward = env.get_transitions(s, a_idx)
                    expected_future = np.sum(P_mat * values)
                    v += prob * (reward + env.DISCOUNT * expected_future)
                new_values[i, j] = v

        if np.max(np.abs(values - new_values)) < tol:
            values = new_values
            break
        values = new_values

    final_policy = get_epsilon_greedy_policy(env, values, epsilon)
    return values, final_policy
