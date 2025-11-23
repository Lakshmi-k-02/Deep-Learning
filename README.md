2D RL Obstacle Avoidance (PPO + Stable-Baselines3)

A lightweight 2D grid environment for training reinforcement learning agents to perform obstacle avoidance and reach a target goal using PPO.
Includes real-time visualization, custom rewards, and easy training/testing scripts.

🚀 Features

10×10 grid environment

4-direction movement (Up/Down/Left/Right)

Obstacle penalties & goal rewards

Distance-based shaping reward

PPO training with Stable-Baselines3

Matplotlib visualization

Optional Gymnasium support

📦 Installation
pip install gym
pip install stable-baselines3
pip install matplotlib numpy


If Gym fails:

pip install gymnasium
pip install gymnasium[classic-control]
