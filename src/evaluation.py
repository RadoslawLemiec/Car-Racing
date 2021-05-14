from statistics import mean, stdev

from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm


def evaluate(model: BaseAlgorithm, env: Env, n_episodes: int = 100):
    episodical_rewards = []

    for _ in range(n_episodes):
        obs = env.reset()
        episodical_reward = 0
        done = False

        while not done:
            action = model.predict(obs)
            obs, rewards, dones, _ = env.step(action[0])
            episodical_reward += rewards[0]
            done = dones[0]   
        episodical_rewards.append(episodical_reward)

    return mean(episodical_rewards), stdev(episodical_rewards)
