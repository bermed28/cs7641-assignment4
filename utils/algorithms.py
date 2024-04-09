import random
import warnings

from tqdm import tqdm
from utils.mdp import MDP


class ValueIteration:
    def __init__(self, mdp: MDP, gamma=0.9, epsilon=0.0001, seed=42):
        random.seed(seed)
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon

        self.V = {}
        self.pi = {}
        self.states, self.actions = self.mdp.get_s_a()

        for s in self.states:
            self.V[s] = 0

    def __call__(self, n_iters=1000):
        """
        Performs Value Iteration using the Bellman Equation

        :return: A tuple containing the optimal policy obtained in VI
                 and a list of the average state values as the algorithm ran
        """
        mean_state_vals, deltas = [self.get_mean_V()], []
        converged = False
        for _ in range(n_iters):
            delta = 0
            for s in self.states:
                cur_v = self.V[s]
                self.V[s] = max([self.Q(s,a) for a in self.actions])
                delta = max(delta, abs(cur_v - self.V[s]))
            deltas.append(delta)
            mean_state_vals.append(self.get_mean_V())
            if delta < self.epsilon:
                converged = True
        if not converged:
            warnings.warn(f"Warning: Value Iteration did not converge within the specified number of iterations ({n_iters})")
        for s in self.states:
            Q_s = [self.Q(s, a) for a in self.actions]
            self.pi[s] = Q_s.index(max(Q_s))
        return self.pi, mean_state_vals, deltas

    def get_mean_V(self):
        return sum(self.V.values()) / len(self.V)

    def Q(self, s, a):
        """
        Evaluates the best Q value for a given state action pair

        :param s: Current state the agent is in before taking action a
        :param a: Current action take by the agent in state s
        :return: The optimum Q value
        """
        Q_s_a = 0
        for s_p in self.states:
            Q_s_a += (
                    self.mdp.get_transition_probability(s, a, s_p)
                    * (self.mdp.get_reward(s, a, s_p) + self.gamma * self.V[s_p])
            )
        return Q_s_a

class PolicyIteration:
    def __init__(self, mdp: MDP, gamma: float=0.9, epsilon:float=0.0001, seed=42):
        random.seed(seed)
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon
        self.states, self.actions = self.mdp.get_s_a()
        self.V = {}
        self.pi = {}

        for s in self.states:
            self.pi[s] = random.choice(self.actions)

    def __call__(self, n_iters=1000):
        avg_policy_deltas, avg_state_values = [], []
        converged = False
        for _ in range(n_iters):
            self.evaluate_policy()
            avg_state_values.append(self.get_avg_V())

            avg_delta = self.policy_update()
            avg_policy_deltas.append(avg_delta)

            if avg_delta == 0:
                converged = True

        if not converged:
            warnings.warn(f"Warning: Policy iteration did not converge within the specified number of iterations ({n_iters})")

        return self.pi, avg_policy_deltas, avg_state_values

    def evaluate_policy(self):
        for s in self.states:
            self.V[s] = 0
        done = False
        while not done:
            delta = 0
            for s in self.states:
                cur_v = self.V[s]
                self.V[s] = self.Q(s, self.pi[s])
                delta += max(delta, abs(cur_v - self.V[s]))
            if delta < self.epsilon:
                done = True

    def policy_update(self):
        total_pi_delta = 0
        for s in self.states:
            prev_a = self.pi[s]
            Q_s = [self.Q(s, a) for a in self.actions]
            self.pi[s] = Q_s.index(max(Q_s))
            total_pi_delta += abs(prev_a - self.pi[s])
        return total_pi_delta / len(self.states)

    def Q(self, s, a):
        Q_s_a = 0
        for s_p in self.states:
            Q_s_a += (
                    self.mdp.get_transition_probability(s, a, s_p)
                    * (self.mdp.get_reward(s, a, s_p) + self.gamma * self.V[s_p])
            )
        return Q_s_a

    def get_avg_V(self):
        return sum(self.V.values()) / len(self.V)

class QLearning:
    def __init__(self, mdp: MDP, alpha: float =0.01, gamma: float=0.9, epsilon: float=1.0, epsilon_decay_rate: float=0.995, seed=42):
        random.seed(seed)
        self.mdp = mdp
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.gamma = gamma
        self.Q = {}
        self.states, self.actions = self.mdp.get_s_a()
        for s in self.states:
            self.Q[s] = [0] * len(self.actions)

    def __call__(self, num_episodes=10000):
        avg_state_values, rewards, delta_qs = [self.get_avg_V()], [], []

        for episode in tqdm(range(num_episodes)):
            total_rewards, max_delta_q = 0, 0
            self.epsilon *= self.epsilon_decay_rate
            s, _, done = self.mdp.reset()

            while not done:
                a = self.select_action(s)
                s_p, reward, done = self.mdp.step(a)

                # Store the old Q value
                old_q_value = self.Q[s][a]

                # Update the Q value
                self.Q[s][a] = old_q_value + self.alpha * (reward + self.gamma * max(self.Q[s_p]) - old_q_value)

                # Calculate and track the delta Q
                delta_q = abs(self.Q[s][a] - old_q_value)
                max_delta_q = max(max_delta_q, delta_q)

                # Update state and total rewards
                s = s_p
                total_rewards += reward

            # After each episode, store the stats
            avg_state_values.append(self.get_avg_V())
            rewards.append(total_rewards)
            delta_qs.append(max_delta_q)  # You could also append the sum of delta Qs here

        # Generate the policy from Q-values
        pi = {s: self.Q[s].index(max(self.Q[s])) for s in self.states}

        return pi, avg_state_values, rewards, delta_qs

    def get_avg_V(self):
        return sum(self.get_V().values())/len(self.states)

    def get_V(self):
        V = {}
        for s in self.states:
            V[s] = max(self.Q[s])
        return V

    def select_action(self, s):
        p = random.random()
        if p < self.epsilon:
            return random.randint(0, len(self.actions) - 1)

        return self.Q[s].index(max(self.Q[s]))
