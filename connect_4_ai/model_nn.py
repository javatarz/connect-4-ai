from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from kaggle_environments import make
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn as nn


class ConnectFourGym(gym.Env):
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=2, shape=(1, self.rows, self.columns), dtype=int)
        self.reward_range = (-10, 1)
        self.spec = None
        self.metadata = {}
        self.obs = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(1, self.rows, self.columns), {}

    def change_reward(self, old_reward: int, done: int) -> int:
        if old_reward == 1:  # The agent won the game
            return 1
        elif done:  # The opponent won the game
            return -1
        else:
            return 1 / (self.rows * self.columns)

    def step(self, action):
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:  # Play the move
            self.obs, old_reward, done, info = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:  # End the game and penalize agent
            reward, done, info = -10, True, {}
        return np.array(self.obs['board']).reshape(1, self.rows, self.columns), reward, done, False, info


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def create_model(agent2: str = "random") -> PPO:
    env = ConnectFourGym(agent2)
    policy_kwargs = dict(features_extractor_class=CustomCNN)
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

    return model
