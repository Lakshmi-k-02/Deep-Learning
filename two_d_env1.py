import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class TwoDEnv(gym.Env):
    def __init__(self, grid_size=10, render_mode=True):
        super(TwoDEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)

        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([grid_size - 1, grid_size - 1])

        # Obstacles
        self.obstacles = {(1, 6), (5, 4), (4, 3), (7, 6), (5, 2)}

        # Rendering
        self.render_mode = render_mode
        if render_mode:
            plt.ion()
            self.fig, self.ax = plt.subplots()

    def reset(self):
        self.agent_pos = np.array([0, 0])
        if self.render_mode:
            self.render()
        return self.agent_pos

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())

        # Movement mapping
        move = {
            0: np.array([0, 1]),    # up
            1: np.array([0, -1]),   # down
            2: np.array([-1, 0]),   # left
            3: np.array([1, 0])     # right
        }

        new_pos = self.agent_pos + move[action]
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)

        reward = -0.1
        done = False

        # Obstacle collision
        if tuple(new_pos) in self.obstacles:
            reward -= 5
            print("❌ Hit obstacle")
        else:
            self.agent_pos = new_pos

        # Goal reached
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 20
            done = True
            print("🏁 GOAL REACHED! Reward +20")

        # Distance-based shaping reward
        distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward -= 0.01 * distance

        if self.render_mode:
            self.render()

        return self.agent_pos, reward, done, {}

    def render(self):
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True)

        # Obstacles
        for obs in self.obstacles:
            self.ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color='gray'))

        # Agent
        self.ax.add_patch(plt.Circle((self.agent_pos[0], self.agent_pos[1]), 0.3, color='blue'))

        # Goal
        self.ax.add_patch(plt.Circle((self.goal_pos[0], self.goal_pos[1]), 0.3, color='red'))

        self.ax.set_title(f"Agent: {self.agent_pos}, Goal: {self.goal_pos}")
        plt.pause(0.2)

    def close(self):
        if self.render_mode:
            plt.ioff()
            plt.close()
