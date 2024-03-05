import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim

from Wendigo.utils.env_utils import start_env, step_env, restart_envs


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize torch layer
    :param layer: layer object to be initialized
    :param std: standard deviation
    :param bias_const: bias constant
    :return: initialized layer
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO_Network(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        """
        Get the value based on observation
        :param x: observation
        :return: value
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Get action and value based on observation
        :param x: observations
        :param action: optional (will sample probability if not provided
        :return: action, probability of action, entropy and value
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def storage_setup(envs, num_envs, num_steps, device):
    """
    Initializes the storage required
    :param envs: The setup gymnasium environments
    :param num_envs: the number of environments
    :param num_steps: the number of steps
    :param device: the device used
    :return: observation, actions, logprobs, reward, dones and value storage objects
    """
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    return obs, actions, logprobs, rewards, dones, values


class PPO:

    def __init__(self, agent_settings, envs, device, agent=None, global_step=None):

        # Agent - General
        self.agent = PPO_Network(envs).to(device) if agent is None else agent

        # Environment
        self.envs = envs
        self.num_envs = agent_settings['num-envs']
        self.total_timesteps = agent_settings['total-timesteps']
        self.num_steps = agent_settings['num-steps']
        self.crash_memory = set()
        self.resume_data = None

        # Agent - Returns Calculation Settings
        self.gamma = agent_settings['gamma']
        self.gae_lambda = agent_settings['gae-lambda']

        # Agent - Optimization Settings
        self.update_epochs = agent_settings['update-epochs']
        self.batch_size = int(self.num_envs * self.num_steps)
        self.num_minibatches = agent_settings['num-minibatches']
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.norm_adv = agent_settings['norm-adv']
        self.clip_coef = agent_settings['clip-coef']
        self.clip_vloss = agent_settings['clip-vloss']
        self.vf_coef = agent_settings['vf-coef']
        self.ent_coef = agent_settings['ent-coef']
        self.target_kl = agent_settings['target-kl']

        # Optimizer Settings
        self.learning_rate = agent_settings['learning-rate']
        self.optimizer_eps = agent_settings['optimizer-eps']
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=self.optimizer_eps)
        self.max_grad_norm = agent_settings['max-grad-norm']

        # Device
        self.device = device

        # Storage
        storage = storage_setup(envs=envs, num_envs=self.num_envs, num_steps=self.num_steps, device=device)
        self.obs = storage[0]
        self.actions = storage[1]
        self.logprobs = storage[2]
        self.rewards = storage[3]
        self.dones = storage[4]
        self.values = storage[5]

        # Run Data
        self.global_step = None if global_step is None else global_step
        self.next_obs = None
        self.next_done = None
        self.advantages = None
        self.returns = None

        # Batch Data
        self.b_obs = None
        self.b_logprobs = None
        self.b_actions = None
        self.b_advantages = None
        self.b_returns = None
        self.b_values = None

        # Optimization Data
        self.v_loss = None
        self.pg_loss = None
        self.entropy_loss = None
        self.clipfracs = None
        self.approx_kl = None
        self.old_approx_kl = None

    def start(self):
        """
        Starts the environment and saves initial values to object
        :return: start_time, num_updates
        """
        start_data = start_env(envs=self.envs, num_envs=self.num_envs,
                               total_timesteps=self.total_timesteps, batch_size=self.batch_size, device=self.device)

        global_step, start_time, self.next_obs, self.next_done, num_updates = start_data
        self.global_step = global_step if self.global_step is None else self.global_step

        return start_time, num_updates

    def anneal_learning_rate(self, update, num_updates):
        """
        Performs annealing of the learning rate in the optimizer
        :param update: the update number
        :param num_updates: the total number of updated
        :return: nothing
        """
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * self.learning_rate
        self.optimizer.param_groups[0]["lr"] = lrnow

    def __action(self, step):
        """
        Performs the action logic for the current step and saves to the storage
        :param step: The current step
        :return: Nothing, values, actions, logprobs are updates internally at step
        """
        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
            self.values[step] = value.flatten()

        self.actions[step] = action
        self.logprobs[step] = logprob

    def _log_step(self, info, writer):
        """
        Performs the logging for the step
        :param info: the info returned from step
        :param writer: the writer object for logging
        :return: None
        """
        for item in info:
            if "episode" in item.keys():
                print("global_step=" + str(self.global_step) + ", episodic_return=" + str(item['episode']['r']))
                writer.add_scalar("charts/episodic_return", item["episode"]["r"], self.global_step)
                writer.add_scalar("charts/episodic_length", item["episode"]["l"], self.global_step)
                break

    def step(self, step, writer):
        """
        Performs the step handling interaction between agent and env
        :param step: the current step
        :param writer: the writer object for logging
        :return: Nothing
        """
        if self.global_step is None or self.next_obs is None or self.next_done is None:
            print("Need to start before performing step")
            return

        self.global_step += 1 * self.num_envs
        self.obs[step] = self.next_obs
        self.dones[step] = self.next_done

        self.__action(step)

        self.next_obs, self.rewards[step], self.next_done, _, info = step_env(envs=self.envs, action=self.actions[step],
                                                                              device=self.device)

        self.resume_data = [x for x in info['resume_data']]

        # Logging
        # self._log_step(info=info, writer=writer)
        # TODO re-enable and fix it

    def calculate_returns(self):
        """
        Calculates the values plus te advantages
        :return: Nothing, returns is updated internally
        """
        with torch.no_grad():
            # bootstrap value if not done
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            self.advantages = torch.zeros_like(self.rewards).to(self.device)
            last_gae_lam = 0

            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_non_terminal = 1.0 - self.next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_values = self.values[t + 1]

                delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
                self.advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

            self.returns = self.advantages + self.values

    def flatten_batch(self):
        """
        Flatten the batches to prepare for optimization
        :return: nothing, updates internally
        """
        if self.returns is None or self.advantages is None:
            print("Need to calculate returns before able to flatten batch")
            return

        self.b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        self.b_logprobs = self.logprobs.reshape(-1)
        self.b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        self.b_advantages = self.advantages.reshape(-1)
        self.b_returns = self.returns.reshape(-1)
        self.b_values = self.values.reshape(-1)

    def __policy_loss(self, minibatch_indexes, ratio):
        """
        Calculate the policy loss for optimization
        :param minibatch_indexes: the minibatch indexes for the minibatch optimization
        :param ratio: the log prob ratio
        :return: nothing, updated internally
        """
        # Advantages
        mb_advantages = self.b_advantages[minibatch_indexes]
        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        self.pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    def __value_loss(self, minibatch_indexes, new_value):
        """
        Calculates the value loss for minibatch optimization
        :param minibatch_indexes: the minibatch indexes for the minibatch optimization
        :param new_value: the action value for the minibatch
        :return: nothing, updated internally
        """
        new_value = new_value.view(-1)

        if self.clip_vloss:
            v_loss_unclipped = (new_value - self.b_returns[minibatch_indexes]) ** 2
            v_clipped = self.b_values[minibatch_indexes] + torch.clamp(new_value - self.b_values[minibatch_indexes],
                                                                       -self.clip_coef, self.clip_coef,)
            v_loss_clipped = (v_clipped - self.b_returns[minibatch_indexes]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            self.v_loss = 0.5 * v_loss_max.mean()

        else:
            self.v_loss = 0.5 * ((new_value - self.b_returns[minibatch_indexes]) ** 2).mean()

    def __optimize_agent(self, loss):
        """
        Runs a step in the optimizer
        :param loss: the result of the loss function
        :return: nothing, updated internally
        """
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def __optimize_minibatch(self, start, batch_indexes):
        """
        Runs the model update/optimization for a minibatch
        :param start: start index of the minibatch
        :param batch_indexes: the list of indexes for the batch
        :return: nothing, updated internally
        """
        # Determine minibatch indexes
        end = start + self.minibatch_size
        minibatch_indexes = batch_indexes[start:end]

        # Get action values for minibatch
        action_value_results = self.agent.get_action_and_value(self.b_obs[minibatch_indexes],
                                                               self.b_actions.long()[minibatch_indexes])
        _, new_log_prob, entropy, new_value = action_value_results

        # Determine log probability ratio
        log_ratio = new_log_prob - self.b_logprobs[minibatch_indexes]
        ratio = log_ratio.exp()

        # Approximate KL and clipfracs
        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            self.approx_kl = ((ratio - 1) - log_ratio).mean()

            # save old kl for logging
            self.old_approx_kl =(-log_ratio).mean()

            # save clipfracs for logging
            self.clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

        # Policy Loss
        self.__policy_loss(minibatch_indexes=minibatch_indexes, ratio=ratio)

        # Value loss
        self.__value_loss(minibatch_indexes=minibatch_indexes, new_value=new_value)

        # Entropy Loss
        self.entropy_loss = entropy.mean()

        # Loss Function
        loss = self.pg_loss - self.ent_coef * self.entropy_loss + self.v_loss * self.vf_coef

        # Optimize
        self.__optimize_agent(loss=loss)

    def optimize_network(self):
        """
        Optimizes the policy and value network
        :return: nothing, updated internally
        """
        batch_indexes = np.arange(self.batch_size)
        self.clipfracs = []
        self.approx_kl = None

        for epoch in range(self.update_epochs):
            # Shuffle batch indexes
            np.random.shuffle(batch_indexes)

            for start in range(0, self.batch_size, self.minibatch_size):
                self.__optimize_minibatch(start=start, batch_indexes=batch_indexes)

            # If target is defined, and it has been reached then stop optimization
            if self.target_kl is not None and self.approx_kl is not None:
                if self.approx_kl > self.target_kl:
                    break

    def log_update(self, writer, start_time):
        """
        Logs data about the update
        :param writer: The writer object for logging
        :param start_time: start time of application running
        :return: nothing
        """
        y_pred, y_true = self.b_values.cpu().numpy(), self.b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        writer.add_scalar("losses/value_loss", self.v_loss.item(), self.global_step)
        writer.add_scalar("losses/policy_loss", self.pg_loss.item(), self.global_step)
        writer.add_scalar("losses/entropy", self.entropy_loss.item(), self.global_step)
        writer.add_scalar("losses/old_approx_kl", self.old_approx_kl.item(), self.global_step)
        writer.add_scalar("losses/approx_kl", self.approx_kl.item(), self.global_step)
        writer.add_scalar("losses/clipfrac", np.mean(self.clipfracs), self.global_step)
        writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
        print("SPS:", int(self.global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - start_time)), self.global_step)

    def save_model(self, run_name, exp_name):
        model_path = "runs/" + run_name + "/" + exp_name + "_"
        extension = ".pickle"

        with open((model_path + "PPO_MODEL" + extension), 'wb') as fp:
            pickle.dump(self.agent, fp)
            print("PPO model saved to " + (model_path + "PPO_MODEL" + extension))

        with open((model_path + "PPO_RESUME_DATA" + extension), 'wb') as fp:
            pickle.dump((self.resume_data, self.global_step), fp)
            print("PPO resume data saved to " + (model_path + "PPO_RESUME_DATA" + extension))

        print('SAVED AT GLOBAL STEP: ' + str(self.global_step))

