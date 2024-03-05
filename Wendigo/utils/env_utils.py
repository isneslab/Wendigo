import time
import gymnasium as gym
import torch
import graphql


def make_env(env_id, seed):
    """
    Make a single environment
    :param env_id: the Gymnasium ID for the environment
    :param seed: the seed for the random number generator
    :return: return the environment created
    """

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_envs(seed, env_id, num_envs):
    """
    Make multiple envs for PPO
    :param seed: the seed of random number generator
    :param env_id: the Gymnasium ID for the environment
    :param num_envs:  the number of environments
    :return: Synchronized vector environment
    """
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed + i) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    return envs


def start_env(envs, num_envs, total_timesteps, batch_size, device):
    """
    Starts the environment
    :param envs: The environments object
    :param num_envs: the number of environments
    :param total_timesteps: the total time steps
    :param batch_size: the batch size
    :param device: torch device
    :return: global step, start_time, next_obs, next_done, num_updates
    """
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // batch_size

    return global_step, start_time, next_obs, next_done, num_updates


def restart_envs(envs, num_envs, device):
    """
    Restarts the environment after termination
    :param envs: The environments object
    :param num_envs: the number of environments
    :param device: torch device
    :return: next_obs, next_done
    """
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    return next_obs, next_done


def step_env(envs, action, device):
    """
    Executes the next step in the environment
    :param envs: the environments object
    :param action: the action to take in the env
    :param device: the torch device
    :return: reward, next_obs, next_done and info
    """
    next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
    reward = torch.tensor(reward).to(device).view(-1)
    next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

    return next_obs, reward, next_done, truncated, info


def get_schema_from_file(schema_path):
    """
    Returns the schema from file
    :param schema_path: the file path to the schema
    :return: graphql schema object
    """
    with open(file=schema_path, mode='r') as schema_file:
        schema = schema_file.read()
        schema = graphql.build_ast_schema(graphql.parse(schema))

    return schema

