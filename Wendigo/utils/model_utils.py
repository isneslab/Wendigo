import stable_baselines3 as sb3
from tqdm import tqdm

from Wendigo.models.PPO import PPO
from Wendigo.models.Random import RandomSearch


def run_model(MODEL, agent_settings, envs, writer, device, save_settings, agent=None, global_step=None):
    if MODEL == "PPO":
        model = run_ppo(agent_settings=agent_settings, agent=agent, global_step=global_step,
                        envs=envs, writer=writer, device=device, save_settings=save_settings)

    elif MODEL == "Random":
        model = run_random(agent_settings=agent_settings, envs=envs, device=device)

    else:
        print("NOT IMPLEMENTED")
        model = None

    return model


def run_ppo(agent_settings, envs, writer, device, save_settings, agent=None, global_step=None):
    # Create ppo manager object
    ppo = PPO(agent_settings=agent_settings, agent=agent, global_step=global_step, envs=envs, device=device)

    # Start the environment and agent manager
    start_time, num_updates = ppo.start()

    # Model update loop
    for update in tqdm(range(1, num_updates + 1)):

        if agent_settings['anneal-lr']:
            # Annealing the rate if requested
            ppo.anneal_learning_rate(update=update, num_updates=num_updates)

        # Perform steps in environment
        for step in tqdm(range(0, agent_settings['num-steps'])):
            ppo.step(step=step, writer=writer)

        # Calculate returns (value + advantage)
        ppo.calculate_returns()

        # Flatten the batch
        ppo.flatten_batch()

        # Optimize Network
        ppo.optimize_network()

        # Log Update
        # ppo.log_update(writer=writer, start_time=start_time)
        # ppo.save_model(run_name=save_settings[0], exp_name=save_settings[1])

    return ppo


def run_random(agent_settings, envs, device):
    # Create random search manager object
    rs = RandomSearch(agent_settings=agent_settings, envs=envs, device=device)

    # Start the environment and agent manager
    rs.start()

    # step env loop
    for step in tqdm(range(0, agent_settings['total-timesteps'])):
        rs.step(step=step)

    return rs
