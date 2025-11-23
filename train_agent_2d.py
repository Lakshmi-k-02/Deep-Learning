from stable_baselines3 import PPO
from two_d_env1 import TwoDEnv

env = TwoDEnv(render_mode=False)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

model.save("ppo_2d_agent")
env.close()
