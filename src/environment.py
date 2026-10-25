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


class GridWorldEnv:
    """
    GridWorld MDP Environment based on Sutton & Barto (2018), Example 3.5.

    Supports both:
    1. Deterministic transitions:
       - Actions always succeed in the intended direction.
    2. Stochastic transitions (Action slipping):
       - Intended action executes with probability 0.85.
       - Each of the other 3 orthogonal/opposite actions executes with probability 0.05.
    """

    WORLD_SIZE = 5
    A_POS = [0, 1]
    A_PRIME_POS = [4, 1]
    B_POS = [0, 3]
    B_PRIME_POS = [2, 3]
    DISCOUNT = 0.9

    # Action order: 0: left, 1: up, 2: right, 3: down
    ACTIONS = [
        np.array([0, -1]),
        np.array([-1, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
    ]
    ACTION_NAMES = ["left", "up", "right", "down"]
    ACTION_SYMBOLS = ["←", "↑", "→", "↓"]
    NUM_ACTIONS = 4

    def __init__(self, is_stochastic: bool = False, slip_prob: float = 0.05):
        """
        Initialize GridWorld environment.

        Args:
            is_stochastic: If True, transition dynamics include action slipping.
            slip_prob: Probability of slipping into each alternative direction (default 0.05).
                       Intended action probability is 1.0 - 3 * slip_prob = 0.85.
        """
        self.is_stochastic = is_stochastic
        self.slip_prob = slip_prob
        self.intended_prob = 1.0 - (self.NUM_ACTIONS - 1) * slip_prob

    def deterministic_step(self, state, action):
        """
        Single deterministic transition.

        Args:
            state: [row, col]
            action: np.array([delta_row, delta_col]) or action index (int)

        Returns:
            next_state: [next_row, next_col]
            reward: float
        """
        if isinstance(action, (int, np.integer)):
            action = self.ACTIONS[action]

        state_list = list(state)
        if state_list == self.A_POS:
            return list(self.A_PRIME_POS), 10.0
        if state_list == self.B_POS:
            return list(self.B_PRIME_POS), 5.0

        next_state = (np.array(state) + action).tolist()
        x, y = next_state
        if x < 0 or x >= self.WORLD_SIZE or y < 0 or y >= self.WORLD_SIZE:
            reward = -1.0
            next_state = state_list
        else:
            reward = 0.0
        return next_state, reward

    def get_transitions(self, state, action):
        """
        Get transition probability distribution P(s' | s, a) and immediate reward.

        Args:
            state: [row, col]
            action: np.array or action index (int)

        Returns:
            P_mat: (WORLD_SIZE, WORLD_SIZE) matrix of next-state probabilities
            reward: immediate reward
        """
        if isinstance(action, (int, np.integer)):
            action_idx = int(action)
            action_vec = self.ACTIONS[action_idx]
        else:
            action_vec = action
            action_idx = None
            for idx, a in enumerate(self.ACTIONS):
                if np.array_equal(a, action_vec):
                    action_idx = idx
                    break

        P_mat = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))

        if not self.is_stochastic:
            next_state, reward = self.deterministic_step(state, action_vec)
            P_mat[next_state[0], next_state[1]] = 1.0
            return P_mat, reward

        # Stochastic dynamics
        mps, reward = self.deterministic_step(state, action_vec)
        for idx, a in enumerate(self.ACTIONS):
            if idx == action_idx:
                P_mat[mps[0], mps[1]] += self.intended_prob
            else:
                s, _ = self.deterministic_step(state, a)
                P_mat[s[0], s[1]] += self.slip_prob

        return P_mat, reward

    def state_to_index(self, state):
        """Convert 2D state [row, col] to 1D index (0..24)."""
        return np.ravel_multi_index(state, (self.WORLD_SIZE, self.WORLD_SIZE))

    def index_to_state(self, index):
        """Convert 1D index (0..24) to 2D state [row, col]."""
        return list(np.unravel_index(index, (self.WORLD_SIZE, self.WORLD_SIZE)))
