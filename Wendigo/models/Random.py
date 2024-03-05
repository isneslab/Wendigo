import random

import numpy as np
import torch

from Wendigo.utils.env_utils import start_env, step_env


class RandomSearch:

    def __init__(self, agent_settings, envs, device):

        # Action Space
        self.action_space = envs.single_action_space
        self.action_space_size = int(self.action_space.n)

        # Environment
        self.envs = envs
        self.num_envs = agent_settings['num-envs']
        assert self.num_envs == 1, "multiple envs are not supported at the moment"
        self.total_timesteps = agent_settings['total-timesteps']
        self.global_step = 0
        self.crash_memory = set()
        self.tracking_data = None

        # Device
        self.device = device

    def start(self):
        """
        Starts the environment and saves initial values to object
        :return: start_time, num_updates
        """
        start_data = start_env(envs=self.envs, num_envs=self.num_envs,
                               total_timesteps=self.total_timesteps, batch_size=1, device=self.device)

        self.global_step, start_time, _, _, num_updates = start_data

        return start_time

    def step(self, step):
        actions = torch.tensor([random.randint(0, self.action_space_size-1)])
        step_env(envs=self.envs, action=actions, device=self.device)


