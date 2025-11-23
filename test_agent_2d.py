from stable_baselines3 import PPO
from two_d_env1 import TwoDEnv
import time

env = TwoDEnv(render_mode=True)
model = PPO.load("ppo_2d_agent")

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    time.sleep(0.1)

env.close()
