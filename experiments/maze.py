import random

from matplotlib import pyplot as plt

from utils.algorithms import ValueIteration, PolicyIteration, QLearning
from utils.mdp import CatMouseMaze


def run():
    seed = 50
    random.seed(seed)

    rows = 4
    cols = 4

    cat_mouse_maze = CatMouseMaze(rows=rows, cols=cols, seed=seed)
    print(f"{len(cat_mouse_maze.mouse_traps)} traps set")
    cat_mouse_maze.print_grid()

    print("Running Value Iteration")
    value_iteration = ValueIteration(cat_mouse_maze, seed=seed)
    print("Running Policy Iteration")
    policy_iteration = PolicyIteration(cat_mouse_maze,seed=seed)
    print("Running Q-Learning")
    q_learning = QLearning(cat_mouse_maze, seed=seed)

    pi_vi, avg_state_vals_vi, deltas_vi = value_iteration(n_iters=25)
    pi_poli, avg_policy_deltas_poli, avg_state_values_poli = policy_iteration(n_iters=25)
    pi_ql, avg_state_values_ql, rewards, delta_q = q_learning(num_episodes=5000)

    cat_mouse_maze.plot_grid_with_actions(pi_vi, f"Optimal Policy - Value Iteration")
    cat_mouse_maze.plot_grid_with_actions(pi_poli, f"Optimal Policy - Policy Iteration")
    cat_mouse_maze.plot_grid_with_actions(pi_ql, f"Optimal Policy - Q-Learning")

    plt.figure(figsize=(8,6))
    plt.plot(avg_state_vals_vi, label="Value Iteration")
    plt.plot(avg_state_values_poli, label="Policy Iteration")
    plt.xlabel('Iteration')
    plt.ylabel('V')
    plt.title('Average V Value: Value/Policy Iteration')
    plt.legend(loc='best')
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='gray')
    plt.savefig("./results/maze/avg_v_vi_pi.png")
    
    plt.figure(figsize=(8,6))
    plt.plot(avg_policy_deltas_poli, label="Policy Iteration")
    plt.plot(deltas_vi, label="Value Iteration")
    plt.xlabel('Iteration')
    plt.ylabel('Policy Delta')
    plt.title('Convergence of Average Policy Delta: Value/Policy Iteration')
    plt.legend(loc='best')
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='gray')
    plt.savefig("./results/maze/pi_delta_vi_poli.png")
    
    plt.figure(figsize=(8,6))
    plt.plot(avg_state_values_ql, label="Q-Learning")
    plt.xlabel('Episode')
    plt.ylabel('Q')
    plt.title('Convergence of Q Value: Q-Learning')
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='gray')
    plt.savefig("./results/maze/ql_q_conv.png")
    
    plt.figure(figsize=(8,6))
    plt.plot(delta_q)
    plt.xlabel('Iteration')
    plt.ylabel('delta_q')
    plt.title('Convergence of Delta Q Value')
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='gray')
    plt.savefig("./results/maze/delta_q_ql.png")
    
    plt.figure(figsize=(8,6))
    plt.plot(rewards)
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Convergence of Reward Value')
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='gray')
    plt.savefig("./results/maze/reward_conv_ql.png")
