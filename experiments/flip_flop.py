import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from utils.algorithms import QLearning, ValueIteration, PolicyIteration
from utils.mdp import FlipFlop

def run():
    seed=42
    random.seed(seed)

    flip_flop = FlipFlop(n=9)

    value_iteration = ValueIteration(flip_flop, seed=seed)
    policy_iteration = PolicyIteration(flip_flop, seed=seed)
    q_learning = QLearning(flip_flop, seed=seed)

    print("Running Value Iteration")
    pi_vi, avg_state_vals_vi, deltas_vi = value_iteration(n_iters=25)
    print("Running Policy Iteration")
    pi_poli, avg_policy_deltas_poli, avg_state_values_poli = policy_iteration(n_iters=25)
    print("Running Q-Learning")
    pi_ql, avg_state_values_ql, rewards, delta_q = q_learning()

    print(f"Optimal Policy (VI): {pi_vi}")
    print(f"Optimal Policy (PI): {pi_poli}")
    print(f"Optimal Policy (QL): {pi_ql}")

    plt.figure(figsize=(8,6))
    plt.plot(avg_state_vals_vi, label="Value Iteration")
    plt.plot(avg_state_values_poli, label="Policy Iteration")
    plt.xlabel('Iteration')
    plt.ylabel('V')
    plt.title('Average V Value: Value/Policy Iteration')
    plt.legend(loc='best')
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25,color='gray')
    plt.savefig("./results/flip_flop/avg_v_vi_pi.png")
    
    plt.figure(figsize=(8,6))
    plt.plot(avg_policy_deltas_poli, label="Policy Iteration")
    plt.plot(deltas_vi, label="Value Iteration")
    plt.xlabel('Iteration')
    plt.ylabel('Policy Delta')
    plt.title('Convergence of Average Policy Delta: Value/Policy Iteration')
    plt.legend(loc='best')
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25,color='gray')
    plt.savefig("./results/flip_flop/avg_pi_vi_poli.png")
    
    plt.figure(figsize=(8,6))
    plt.plot(avg_state_values_ql, label="Q-Learning")
    plt.xlabel('Episode')
    plt.ylabel('Q')
    plt.title('Convergence of Q Value: Q-Learning')
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='gray')
    plt.savefig("./results/flip_flop/q_conv_ql.png")
    
    plt.figure(figsize=(8,6))
    plt.plot(delta_q)
    plt.xlabel('Iteration')
    plt.ylabel('delta_q')
    plt.title('Convergence of Delta Q Value')
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='gray')
    plt.savefig("./results/flip_flop/delta_q_conv_ql.png")
    
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Convergence of Reward Value')
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25,color='gray')
    plt.savefig("./results/flip_flop/reward_conv_ql.png")

    # Optional plot for small graphs
    # G = nx.DiGraph()
    #
    # # Add nodes and edges to the graph
    # for state, action in pi_vi.items():
    #     next_state = list(state)
    #     next_state[action] = '1' if state[action] == '0' else '0'  # Flip the bit
    #     next_state = ''.join(next_state)
    #     G.add_edge(state, next_state, label=str(action))
    #
    # pos = nx.spring_layout(G)  # Position the nodes using the spring layout
    #
    # plt.figure(figsize=(15, 15))
    # # Draw the nodes and edges
    # nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue', font_size=8)
    # edge_labels = nx.get_edge_attributes(G, 'label')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    #
    # plt.title('Policy Visualization for Flip Flop Problem')
    # plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='gray')
    # plt.savefig("./results/flip_flop/pi_graph.png")

    action_counts = np.zeros((len(pi_vi), max(pi_vi.values()) + 1))
    for i, (state, action) in enumerate(pi_vi.items()):
        action_counts[i, action] = 1  # Mark the action for this state


    action_distribution = np.sum(action_counts, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(action_distribution)), action_distribution)
    plt.xlabel('Bit Position')
    plt.ylabel('Frequency of Flip')
    plt.title('Bit-Flipping Patterns in Optimal Policy - Value Iteration')
    plt.xticks(range(len(action_distribution)))
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='gray')
    plt.savefig("./results/flip_flop/histogram_vi.png")
    plt.show()

    action_counts = np.zeros((len(pi_poli), max(pi_poli.values()) + 1))
    for i, (state, action) in enumerate(pi_poli.items()):
        action_counts[i, action] = 1  # Mark the action for this state


    action_distribution = np.sum(action_counts, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(action_distribution)), action_distribution)
    plt.xlabel('Bit Position')
    plt.ylabel('Frequency of Flip')
    plt.title('Bit-Flipping Patterns in Optimal Policy - Policy Iteration')
    plt.xticks(range(len(action_distribution)))
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='gray')
    plt.savefig("./results/flip_flop/histogram_pi.png")
    plt.show()

    action_counts = np.zeros((len(pi_ql), max(pi_ql.values()) + 1))
    for i, (state, action) in enumerate(pi_ql.items()):
        action_counts[i, action] = 1  # Mark the action for this state


    action_distribution = np.sum(action_counts, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(action_distribution)), action_distribution)
    plt.xlabel('Bit Position')
    plt.ylabel('Frequency of Flip')
    plt.title('Bit-Flipping Patterns in Optimal Policy - Q-Learning')
    plt.xticks(range(len(action_distribution)))
    plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='gray')
    plt.savefig("./results/flip_flop/histogram_ql.png")
    plt.show()