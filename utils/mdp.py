from abc import ABC, abstractmethod
import random

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


class MDP(ABC):

    @abstractmethod
    def get_terminal_states(self):
        pass

    @abstractmethod
    def get_s_a(self):
        pass

    @abstractmethod
    def get_transition_probability(self, s, a, s_p):
        pass

    @abstractmethod
    def get_reward(self, s, a, s_p):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, a):
        pass


class CatMouseMaze(MDP):

    def __init__(self, cat_position=None, mouse_traps=None, rows=4, cols=4, seed=42):
        """
        MDP for a Cat & Mouse Maze Game
        Goal: Mouse has to reach cheese starting from position (0,0) without getting caught
        Rewards:
            +1 if mouse reaches cheese without getting caught
            -1 if mouse gets caught by cat before getting to cheese
        Actions:
            0: Up
            1: Right
            2: Down
            3: Left
        Transitions:
            Mouse moves in correct direction with probability of 0.33 (i.e. 1/3)
            Mouse moves in wrong direction (i.e. perpendicular to desired action) with probability of 0.33 (i.e. 1/3)
            Mouse does not move if there is a trap or a grid wall in any direction

        :param cat_position: Tuple containing the initial (x,y) coordinate in grid of the cat (i.e. -1 terminal state)
        :param mouse_traps: Set containing (x,y) coordinates of mousetraps in the grid
        :param rows: Number of rows the grid world will have
        :param cols: Number of columns the grid world will have
        :param seed: Randomization seed for reproducibility
        """
        random.seed(seed)
        self.rows = rows
        self.cols = cols
        self.mouse_position = (0, 0)
        self.cheese_position = (rows - 1, cols - 1) # Always place cheese in bottom right corner of grid
        self.cat_position = cat_position or self.place_cat()
        self.mouse_traps = mouse_traps or self.generate_mouse_traps()

        if not self.is_path(self.mouse_position, self.cheese_position, self.mouse_traps, self.cat_position):
            self.cat_position = self.reposition_cat()

        self.actions = {0, 1, 2, 3}  # Allowed actions to take by mouse

    def is_path(self, start, goal, traps, cat_pos):
        """
        Check if there's a path from start to cheese without hitting any traps or getting caught by cat using BFS.

        :param start: The starting position (tuple).
        :param goal: The goal position (tuple).
        :param traps: A set of positions representing traps.
        :param cat_pos: The position of the cat
        :return: True if a path exists, False otherwise.
        """
        if start == goal:
            return True

        visited = set()
        queue = [start]

        while queue:
            current = queue.pop(0)

            if current in visited or current in traps or current == cat_pos:
                continue

            visited.add(current)

            if current == goal:
                return True

            # Generate all possible next moves
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Directions: right, down, left, up
                next_x, next_y = current[0] + dx, current[1] + dy
                if 0 <= next_x < self.rows and 0 <= next_y < self.cols and (next_x, next_y) not in visited:
                    queue.append((next_x, next_y))

        return False
    def generate_mouse_traps(self):
        traps = set()
        possible_trap_locations = [
            (row, col) for row in range(self.rows) for col in range(self.cols)
            if (row, col) != self.mouse_position and (row, col) != self.cheese_position and (row, col) != self.cat_position
        ]

        max_number_of_traps = len(possible_trap_locations) // 4
        while len(traps) < max_number_of_traps:
            trap_candidate = random.choice(possible_trap_locations)
            traps.add(trap_candidate)
            if not self.is_path(self.mouse_position, self.cheese_position, traps, self.cat_position):
                traps.remove(trap_candidate)  # Remove the trap if it blocks the path
            possible_trap_locations.remove(trap_candidate)  # Avoid retrying the same location

        return traps

    def place_cat(self):
        possible_locations = [
            (row, col) for row in range(self.rows) for col in range(self.cols)
            if (row, col) != self.mouse_position and (row, col) != self.cheese_position
        ]
        return random.choice(possible_locations)

    def reposition_cat(self):
        # Reposition the cat if the initial placement blocks the path
        for potential_cat_pos in [(row, col) for row in range(self.rows) for col in range(self.cols)
                                  if (row, col) != self.mouse_position and
                                     (row, col) != self.cheese_position and
                                     (row, col) not in self.mouse_traps]:
            if self.is_path(self.mouse_position, self.cheese_position, self.mouse_traps, potential_cat_pos):
                return potential_cat_pos
        return self.cat_position

    def get_terminal_states(self):
        """
        :return: The positions of the terminal states (Cat for -1 reward, Cheese for +1 reward)
        """
        return [self.cat_position, self.cheese_position]

    def get_s_a(self):
        """
        Returns the allowed states and actions based on the traps set in the grid world
        :return: A tuple containing a list of allowed states and a list of allowed actions in teh grid world
        """
        states = []
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) not in self.mouse_traps:
                    states.append((row, col))
        return states, list(self.actions)

    def get_transition_probability(self, s, a, s_p):
        out_of_bounds = (s[0] < 0 or s[0] > self.rows) or (s[1] < 0 or s[1] > self.cols)
        invalid_action = a not in self.actions
        in_terminal_state = s in self.get_terminal_states()

        if out_of_bounds or invalid_action or in_terminal_state:
            return 0  # No probability of moving to next state s_p

        next_states = set()
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]

        next_actions = [1, 3] if a == 0 or a == 2 else [0, 2]
        next_actions += [a]
        for a in next_actions:
            x = s[0] + dx[a]
            y = s[1] + dy[a]

            new_state = (x, y)
            in_bounds = 0 <= x <= self.rows and 0 <= y < + self.cols
            if new_state not in self.mouse_traps and in_bounds:
                next_states.add(new_state)

        if s_p in next_states:
            return 1 / 3

        return 0

    def get_reward(self, s, a, s_p):
        return 1 if s_p == self.cheese_position else -1 if s_p == self.cat_position else 0

    def reset(self):
        self.mouse_position = (0, 0)
        return self.mouse_position, 0, False

    def step(self, a):
        if a not in self.actions:
            raise ValueError(f"Invalid action {a}")

        if self.mouse_position not in self.get_terminal_states():
            # Choose between moving in the desired action or perpendicular to it
            a = random.choice([a, 1, 3] if a == 0 or a == 2 else [a, 0, 2])
            dx = [-1, 0, 1, 0]
            dy = [0, 1, 0, -1]
            x, y = self.mouse_position[0] + dx[a], self.mouse_position[1] + dy[a]
            reward = 0
            done = False
            if (x, y) not in self.mouse_traps and 0 <= x <= self.rows - 1 and 0 <= y <= self.cols - 1:
                reward += self.get_reward(self.mouse_position, a, (x, y))
                self.mouse_position = (x, y)
                if reward != 0:
                    done = True
            return self.mouse_position, reward, done

        return self.mouse_position, 0, True

    def print_grid(self):
        """
        Print the grid with the following symbols:
        - 'C' for Cheese
        - 'T' for Traps
        - 'M' for Mouse
        - 'X' for Cat
        - '.' for empty spaces
        """
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) == self.mouse_position:
                    print('M', end=' ')
                elif (i, j) == self.cheese_position:
                    print('C', end=' ')
                elif (i, j) in self.mouse_traps:
                    print('T', end=' ')
                elif (i, j) == self.cat_position:
                    print('X', end=' ')
                else:
                    print('.', end=' ')
            print()  # Newline after each row

    def print_grid_with_actions(self, policy):
        """
        Print the grid with actions for each state.
        - 'C' for Cheese
        - 'T' for Traps
        - 'M' for Mouse
        - 'X' for Cat
        - Arrows (↑, →, ↓, ←) for actions in other cells

        :param policy: A dictionary where keys are state coordinates and values are actions.
        """
        action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) == self.cheese_position:
                    print('C', end=' ')
                elif (i, j) in self.mouse_traps:
                    print('T', end=' ')
                elif (i, j) == self.cat_position:
                    print('X', end=' ')
                elif (i, j) in policy:
                    print(action_symbols[policy[(i, j)]], end=' ')
                else:
                    print('.', end=' ')
            print()  # Newline after each row

    def plot_grid_with_actions(self, policy, title):
        """
        Plot the grid as a board game with traps, the cat, and the cheese.
        """
        fig, ax = plt.subplots(figsize=(8, 8))  # Adjust the figure size as needed

        ax.set_title(title)
        # Set up the axes to span the whole grid with the origin at the top-left corner
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)

        # Draw the grid background
        ax.add_patch(Rectangle((-0.5, -0.5), self.cols, self.rows, facecolor='green', edgecolor="black"))

        # Draw the grid elements
        for i in range(self.rows):
            for j in range(self.cols):
                # Inverting the y-coordinate for plotting to match grid coordinate system
                inv_i = self.rows - 1 - i

                # Draw the traps
                if (i, j) in self.mouse_traps:
                    ax.add_patch(Rectangle((j - 0.5, inv_i - 0.5), 1, 1, facecolor='gray', edgecolor='black'))
                    ax.text(j, inv_i, 'Trap', ha='center', va='center', color='yellow', fontsize=12, weight='bold')

                # Draw the cat's position
                elif (i, j) == self.cat_position:
                    ax.add_patch(Rectangle((j - 0.5, inv_i - 0.5), 1, 1, facecolor='red', edgecolor='black'))
                    ax.text(j, inv_i, 'Cat', ha='center', va='center', color='white', fontsize=12, weight='bold')

                # Draw the cheese's position
                elif (i, j) == self.cheese_position:
                    ax.add_patch(Rectangle((j - 0.5, inv_i - 0.5), 1, 1, facecolor='blue', edgecolor='black'))
                    ax.text(j, inv_i, 'Goal', ha='center', va='center', color='yellow', fontsize=12, weight='bold')
                else:
                    ax.add_patch(Rectangle((j - 0.5, inv_i - 0.5), 1, 1, facecolor='green', edgecolor='black'))

        # Draw the policy arrows
        action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        for (i, j), action in policy.items():
            if (i, j) != self.cheese_position and (i, j) != self.cat_position:
                inv_i = self.rows - 1 - i
                ax.text(j, inv_i, action_symbols[action], ha='center', va='center', color='white', fontsize=16, weight='bold')

        ax.axis("off")
        plt.annotate('fjb7@gatech.edu', xy=(0.3, 0.3), xycoords='axes fraction', rotation=45, alpha=0.5, fontsize=25, color='darkgray')
        plt.savefig(f"./results/maze/policy_{'ql' if 'Q' in title else 'vi' if 'Value' in title else 'pi'}.png")

class FlipFlop(MDP):
    def __init__(self, n=9):
        """
        MDP for Flip Flop problem with size n
        Goal: Flip certain bit strings such that the string has an alternating binary pattern
        Rewards:
            +1 if string has an alternating binary pattern (e.g., 010101... or 101010... for any n)
            -2 if string has all bits flipped to 1 (e.g., 111111... for any n)
        Actions:
            Any bit in the string can be flipped
        Transitions:
            * If action a was taken to flip bit i, then bit i + 1 can be flipped with a probability of 0.5
            * If action a was taken to flip bit i, then bit i + 1 cannot be flipped with a probability of 0.5
            * The last bit in the string is guaranteed to flip since there is no bit next to it
        :param n: Length of binary string
        """
        self.size = n
        self.start = "0" * self.size
        self.positive_end_states = [
            "01" * (self.size // 2) + "0" * (self.size % 2),  # Alternating bit string starting with 1
            "10" * (self.size // 2) + "1" * (self.size % 2)  # Alternating bit string starting with 0
        ]
        self.negative_end_states = ["1" * self.size]
        self.cur_state = self.start

    def get_terminal_states(self):
        return self.positive_end_states + self.negative_end_states

    def get_s_a(self):
        states = []
        self.get_states("", states)
        actions = list(range(self.size))
        return states, actions

    def get_states(self, s, states):
        if len(s) == self.size:
            states.append(s)
            return
        self.get_states(s + '0', states)
        self.get_states(s + '1', states)

    def get_transition_probability(self, s, a, s_p):
        out_of_bounds = a < 0 or a > self.size - 1
        in_end_state = s in self.get_terminal_states()
        if out_of_bounds or in_end_state:
            return 0
        next_states = set()
        new_value = 1 - int(s[a])
        if a < self.size - 1:
            a += 1
            new_value = 1 - int(s[a])  # Flip Bit
            next_states.add(s[0:a] + str(new_value) + s[a + 1:])  # Replace current bit with flipped bit

        if s_p in next_states:
            return 0.5
        return 0

    def get_reward(self, s, a, s_p):
        return -2 if s_p in self.negative_end_states else 1 if s_p in self.positive_end_states else 0

    def reset(self):
        self.cur_state = self.start
        return self.cur_state, 0, False

    def step(self, a):
        if a < 0 or a > self.size - 1:
            raise IndexError(f"Invalid index in action {a}")

        if self.cur_state not in self.get_terminal_states():
            if a < self.size - 1:
                a = random.choice([a, a + 1])
            new_value = 1 - int(self.cur_state[a])
            s_p = self.cur_state[0:a] + str(new_value) + self.cur_state[a+1:]
            reward = self.get_reward(self.cur_state, a, s_p)
            self.cur_state = s_p
            done = False
            if reward != 0:
                done = True
            return self.cur_state, reward, done

        return self.cur_state, 0, True
