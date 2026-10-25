#!/usr/bin/env python3
"""
Main CLI entry point for the GridWorld MDP Dynamic Programming Suite.

Usage:
    python main.py --all
    python main.py --env deterministic --algo value_iteration
    python main.py --env stochastic --algo policy_iteration --epsilon 0.2
"""

import argparse
import os
import sys

from src.environment import GridWorldEnv
from src.dynamic_programming import (
    evaluate_policy_linear,
    evaluate_policy_iterative,
    value_iteration,
    policy_iteration,
)
from src.visualization import (
    plot_grid_values,
    plot_policy,
    plot_comparison_summary,
)


def run_suite(output_dir: str = "assets"):
    """Run all experiments and generate all figures for the documentation."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(" Running Complete GridWorld Dynamic Programming Suite")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Deterministic Environment Experiments
    # -------------------------------------------------------------
    print("\n[1/2] Running Deterministic Environment Experiments...")
    env_det = GridWorldEnv(is_stochastic=False)

    # 1.1 Policy Evaluation (Linear System) - Uniform Random Policy
    print("  -> Deterministic: Exact Linear System Policy Evaluation")
    v_det_linear = evaluate_policy_linear(env_det)
    plot_grid_values(
        v_det_linear,
        title="Deterministic: Linear System Policy Evaluation (π_equiprobable)",
        save_path=os.path.join(output_dir, "deterministic_linear_eval.png"),
    )

    # 1.2 Policy Evaluation (Iterative) - Uniform Random Policy
    print("  -> Deterministic: Iterative Policy Evaluation")
    v_det_iter = evaluate_policy_iterative(env_det)
    plot_grid_values(
        v_det_iter,
        title="Deterministic: Iterative Policy Evaluation (π_equiprobable)",
        save_path=os.path.join(output_dir, "deterministic_iterative_eval.png"),
    )

    # 1.3 Value Iteration (Bellman Optimality)
    print("  -> Deterministic: Value Iteration")
    v_det_opt, pi_det_opt = value_iteration(env_det)
    plot_grid_values(
        v_det_opt,
        title="Deterministic: Optimal Value Function V*(s)",
        save_path=os.path.join(
            output_dir, "deterministic_value_iteration_values.png"
        ),
    )
    plot_policy(
        pi_det_opt,
        env_det,
        title="Deterministic: Optimal Policy π*(s)",
        save_path=os.path.join(
            output_dir, "deterministic_value_iteration_policy.png"
        ),
    )

    # 1.4 Policy Iteration (epsilon = 0.0)
    print("  -> Deterministic: Policy Iteration (epsilon = 0.0)")
    v_det_pi0, pi_det_pi0 = policy_iteration(env_det, epsilon=0.0)
    plot_grid_values(
        v_det_pi0,
        title="Deterministic: Policy Iteration Values (ε = 0.0)",
        save_path=os.path.join(
            output_dir, "deterministic_policy_iteration_eps0_values.png"
        ),
    )
    plot_policy(
        pi_det_pi0,
        env_det,
        title="Deterministic: Policy Iteration Policy (ε = 0.0)",
        save_path=os.path.join(
            output_dir, "deterministic_policy_iteration_eps0_policy.png"
        ),
    )

    # 1.5 Policy Iteration (epsilon = 0.2)
    print("  -> Deterministic: Policy Iteration (epsilon = 0.2)")
    v_det_pi2, pi_det_pi2 = policy_iteration(env_det, epsilon=0.2)
    plot_grid_values(
        v_det_pi2,
        title="Deterministic: Policy Iteration Values (ε = 0.2)",
        save_path=os.path.join(
            output_dir, "deterministic_policy_iteration_eps02_values.png"
        ),
    )
    plot_policy(
        pi_det_pi2,
        env_det,
        title="Deterministic: Policy Iteration Policy (ε = 0.2)",
        save_path=os.path.join(
            output_dir, "deterministic_policy_iteration_eps02_policy.png"
        ),
    )

    # -------------------------------------------------------------
    # 2. Stochastic Environment Experiments (15% slip)
    # -------------------------------------------------------------
    print("\n[2/2] Running Stochastic Environment Experiments (15% slip)...")
    env_stoch = GridWorldEnv(is_stochastic=True, slip_prob=0.05)

    # 2.1 Stochastic Policy Evaluation (Linear System)
    print("  -> Stochastic: Exact Linear System Policy Evaluation")
    v_stoch_linear = evaluate_policy_linear(env_stoch)
    plot_grid_values(
        v_stoch_linear,
        title="Stochastic: Linear System Policy Evaluation (π_equiprobable)",
        save_path=os.path.join(output_dir, "stochastic_linear_eval.png"),
    )

    # 2.2 Stochastic Policy Evaluation (Iterative)
    print("  -> Stochastic: Iterative Policy Evaluation")
    v_stoch_iter = evaluate_policy_iterative(env_stoch)
    plot_grid_values(
        v_stoch_iter,
        title="Stochastic: Iterative Policy Evaluation (π_equiprobable)",
        save_path=os.path.join(output_dir, "stochastic_iterative_eval.png"),
    )

    # 2.3 Stochastic Value Iteration
    print("  -> Stochastic: Value Iteration")
    v_stoch_opt, pi_stoch_opt = value_iteration(env_stoch)
    plot_grid_values(
        v_stoch_opt,
        title="Stochastic: Optimal Value Function V*(s) (15% slip)",
        save_path=os.path.join(
            output_dir, "stochastic_value_iteration_values.png"
        ),
    )
    plot_policy(
        pi_stoch_opt,
        env_stoch,
        title="Stochastic: Optimal Policy π*(s) (15% slip)",
        save_path=os.path.join(
            output_dir, "stochastic_value_iteration_policy.png"
        ),
    )

    # 2.4 Stochastic Policy Iteration (epsilon = 0.0)
    print("  -> Stochastic: Policy Iteration (epsilon = 0.0)")
    v_stoch_pi0, pi_stoch_pi0 = policy_iteration(env_stoch, epsilon=0.0)
    plot_grid_values(
        v_stoch_pi0,
        title="Stochastic: Policy Iteration Values (ε = 0.0)",
        save_path=os.path.join(
            output_dir, "stochastic_policy_iteration_eps0_values.png"
        ),
    )
    plot_policy(
        pi_stoch_pi0,
        env_stoch,
        title="Stochastic: Policy Iteration Policy (ε = 0.0)",
        save_path=os.path.join(
            output_dir, "stochastic_policy_iteration_eps0_policy.png"
        ),
    )

    # 2.5 Stochastic Policy Iteration (epsilon = 0.2)
    print("  -> Stochastic: Policy Iteration (epsilon = 0.2)")
    v_stoch_pi2, pi_stoch_pi2 = policy_iteration(env_stoch, epsilon=0.2)
    plot_grid_values(
        v_stoch_pi2,
        title="Stochastic: Policy Iteration Values (ε = 0.2)",
        save_path=os.path.join(
            output_dir, "stochastic_policy_iteration_eps02_values.png"
        ),
    )
    plot_policy(
        pi_stoch_pi2,
        env_stoch,
        title="Stochastic: Policy Iteration Policy (ε = 0.2)",
        save_path=os.path.join(
            output_dir, "stochastic_policy_iteration_eps02_policy.png"
        ),
    )

    # -------------------------------------------------------------
    # 3. Multi-Panel Comparison Summary Plot
    # -------------------------------------------------------------
    print("\n[3/3] Generating Master Comparison Summary Figure...")
    summary_path = os.path.join(output_dir, "comparison_summary.png")
    plot_comparison_summary(
        v_det_opt,
        pi_det_opt,
        v_stoch_opt,
        pi_stoch_opt,
        env_det,
        env_stoch,
        save_path=summary_path,
    )

    print("\n" + "=" * 70)
    print(f" All 13 high-resolution figures successfully saved to '{output_dir}/'!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="GridWorld MDP Dynamic Programming Suite"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Run full experimental suite and generate all assets",
    )
    parser.add_argument(
        "--env",
        choices=["deterministic", "stochastic"],
        default="deterministic",
        help="Environment transition model",
    )
    parser.add_argument(
        "--algo",
        choices=[
            "linear_eval",
            "iterative_eval",
            "value_iteration",
            "policy_iteration",
        ],
        default="value_iteration",
        help="Dynamic programming algorithm to execute",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration rate for policy iteration (default: 0.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets",
        help="Directory to save output figures (default: assets)",
    )

    args = parser.parse_args()

    # If specific flags are supplied via command line other than default --all
    if len(sys.argv) > 1 and not args.all:
        is_stoch = args.env == "stochastic"
        env = GridWorldEnv(is_stochastic=is_stoch)
        os.makedirs(args.output_dir, exist_ok=True)

        if args.algo == "linear_eval":
            vals = evaluate_policy_linear(env)
            plot_grid_values(
                vals,
                title=f"{args.env.capitalize()}: Linear System Eval",
                save_path=os.path.join(args.output_dir, f"{args.env}_linear.png"),
            )
        elif args.algo == "iterative_eval":
            vals = evaluate_policy_iterative(env)
            plot_grid_values(
                vals,
                title=f"{args.env.capitalize()}: Iterative Eval",
                save_path=os.path.join(
                    args.output_dir, f"{args.env}_iterative.png"
                ),
            )
        elif args.algo == "value_iteration":
            vals, pol = value_iteration(env)
            plot_grid_values(
                vals,
                title=f"{args.env.capitalize()}: Value Iteration Values",
                save_path=os.path.join(args.output_dir, f"{args.env}_vi_vals.png"),
            )
            plot_policy(
                pol,
                env,
                title=f"{args.env.capitalize()}: Value Iteration Policy",
                save_path=os.path.join(
                    args.output_dir, f"{args.env}_vi_policy.png"
                ),
            )
        elif args.algo == "policy_iteration":
            vals, pol = policy_iteration(env, epsilon=args.epsilon)
            plot_grid_values(
                vals,
                title=f"{args.env.capitalize()}: Policy Iteration (ε={args.epsilon})",
                save_path=os.path.join(
                    args.output_dir, f"{args.env}_pi_eps{args.epsilon}_vals.png"
                ),
            )
            plot_policy(
                pol,
                env,
                title=f"{args.env.capitalize()}: Policy Iteration Policy (ε={args.epsilon})",
                save_path=os.path.join(
                    args.output_dir, f"{args.env}_pi_eps{args.epsilon}_policy.png"
                ),
            )
    else:
        run_suite(args.output_dir)


if __name__ == "__main__":
    main()
