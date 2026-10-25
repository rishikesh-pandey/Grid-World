#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang (zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada (hyperkentakun@gmail.com)                         #
# 2024 Rishikesh                                                      #
#                                                                     #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use("Agg")


def _add_grid_cells(ax, grid_data, special_markers=True):
    """Helper to draw grid table with row/col labels."""
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = grid_data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    a_pos = [0, 1]
    a_prime_pos = [4, 1]
    b_pos = [0, 3]
    b_prime_pos = [2, 3]

    for (i, j), val in np.ndenumerate(grid_data):
        text_val = str(val)
        cell_facecolor = "#ffffff"

        if special_markers:
            if [i, j] == a_pos:
                text_val += "\n(A)"
                cell_facecolor = "#e8f4fd"
            elif [i, j] == a_prime_pos:
                text_val += "\n(A')"
                cell_facecolor = "#e8f4fd"
            elif [i, j] == b_pos:
                text_val += "\n(B)"
                cell_facecolor = "#fef9e7"
            elif [i, j] == b_prime_pos:
                text_val += "\n(B')"
                cell_facecolor = "#fef9e7"

        tb.add_cell(
            i,
            j,
            width,
            height,
            text=text_val,
            loc="center",
            facecolor=cell_facecolor,
            edgecolor="#333333",
        )

    # Row and Column coordinate labels (1..WORLD_SIZE)
    for i in range(nrows):
        tb.add_cell(
            i,
            -1,
            width * 0.4,
            height,
            text=str(i + 1),
            loc="center",
            edgecolor="none",
            facecolor="none",
        )
        tb.add_cell(
            -1,
            i,
            width,
            height * 0.4,
            text=str(i + 1),
            loc="center",
            edgecolor="none",
            facecolor="none",
        )

    ax.add_table(tb)


def plot_grid_values(
    values: np.ndarray,
    title: str = "State-Value Function V(s)",
    save_path: str = None,
    decimals: int = 1,
):
    """
    Render state-value matrix as a clean formatted table.

    Args:
        values: (WORLD_SIZE, WORLD_SIZE) numpy array
        title: Figure title
        save_path: Filepath to save image (e.g. assets/values.png)
        decimals: Decimal precision for displayed values
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    rounded = np.round(values, decimals=decimals)
    formatted = np.vectorize(lambda x: f"{x:.{decimals}f}")(rounded)

    _add_grid_cells(ax, formatted, special_markers=True)
    if title:
        plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_policy(
    policy_or_values,
    env,
    title: str = "Optimal Policy π*(s)",
    save_path: str = None,
):
    """
    Render policy arrow grid showing optimal actions for each cell.

    Args:
        policy_or_values: Either (WORLD_SIZE, WORLD_SIZE, NUM_ACTIONS) policy array
                          or (WORLD_SIZE, WORLD_SIZE) value array.
        env: GridWorldEnv instance
        title: Figure title
        save_path: Filepath to save image
    """
    from .dynamic_programming import get_greedy_actions

    fig, ax = plt.subplots(figsize=(6, 6))

    if policy_or_values.ndim == 2:
        # State values passed: compute greedy actions
        greedy_dict = get_greedy_actions(env, policy_or_values)
    else:
        # Policy probabilities passed: find actions with max probability
        greedy_dict = {}
        for i in range(env.WORLD_SIZE):
            for j in range(env.WORLD_SIZE):
                probs = policy_or_values[i, j]
                max_p = np.max(probs)
                best = np.where(np.abs(probs - max_p) < 1e-8)[0].tolist()
                greedy_dict[(i, j)] = best

    arrow_grid = np.empty((env.WORLD_SIZE, env.WORLD_SIZE), dtype=object)
    for i in range(env.WORLD_SIZE):
        for j in range(env.WORLD_SIZE):
            best = greedy_dict[(i, j)]
            symbols = "".join([env.ACTION_SYMBOLS[a] for a in best])
            arrow_grid[i, j] = symbols

    _add_grid_cells(ax, arrow_grid, special_markers=True)
    if title:
        plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison_summary(
    det_values: np.ndarray,
    det_policy,
    stoch_values: np.ndarray,
    stoch_policy,
    env_det,
    env_stoch,
    save_path: str = "assets/comparison_summary.png",
):
    """
    Create a 2x2 multi-panel comparison figure:
    - Top: Deterministic Values & Policy
    - Bottom: Stochastic Values & Policy (Action slip = 15%)
    """
    from .dynamic_programming import get_greedy_actions

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Deterministic Values
    rounded_det = np.vectorize(lambda x: f"{x:.1f}")(
        np.round(det_values, decimals=1)
    )
    _add_grid_cells(axes[0, 0], rounded_det, special_markers=True)
    axes[0, 0].set_title(
        "Deterministic: Optimal Values V*(s)", fontsize=12, fontweight="bold"
    )

    # 2. Deterministic Policy
    greedy_det = get_greedy_actions(env_det, det_values)
    arrows_det = np.empty(
        (env_det.WORLD_SIZE, env_det.WORLD_SIZE), dtype=object
    )
    for i in range(env_det.WORLD_SIZE):
        for j in range(env_det.WORLD_SIZE):
            arrows_det[i, j] = "".join(
                [env_det.ACTION_SYMBOLS[a] for a in greedy_det[(i, j)]]
            )
    _add_grid_cells(axes[0, 1], arrows_det, special_markers=True)
    axes[0, 1].set_title(
        "Deterministic: Optimal Policy π*(s)", fontsize=12, fontweight="bold"
    )

    # 3. Stochastic Values
    rounded_stoch = np.vectorize(lambda x: f"{x:.1f}")(
        np.round(stoch_values, decimals=1)
    )
    _add_grid_cells(axes[1, 0], rounded_stoch, special_markers=True)
    axes[1, 0].set_title(
        "Stochastic (15% slip): Optimal Values V*(s)",
        fontsize=12,
        fontweight="bold",
    )

    # 4. Stochastic Policy
    greedy_stoch = get_greedy_actions(env_stoch, stoch_values)
    arrows_stoch = np.empty(
        (env_stoch.WORLD_SIZE, env_stoch.WORLD_SIZE), dtype=object
    )
    for i in range(env_stoch.WORLD_SIZE):
        for j in range(env_stoch.WORLD_SIZE):
            arrows_stoch[i, j] = "".join(
                [env_stoch.ACTION_SYMBOLS[a] for a in greedy_stoch[(i, j)]]
            )
    _add_grid_cells(axes[1, 1], arrows_stoch, special_markers=True)
    axes[1, 1].set_title(
        "Stochastic (15% slip): Optimal Policy π*(s)",
        fontsize=12,
        fontweight="bold",
    )

    plt.suptitle(
        "GridWorld Dynamic Programming: Deterministic vs. Stochastic Transitions",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
