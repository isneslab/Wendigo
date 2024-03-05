import json
import pickle
import time
from gymnasium.envs.registration import register
from Wendigo.utils.env_utils import make_envs, get_schema_from_file
from Wendigo.utils.model_utils import run_model
from Wendigo.utils.setup_utils import torch_setup, seeding, writer_setup, wandb_setup

MODEL = "Random"  # PPO or Random
TEST = "DoS"  # DoS

DESC = 'Greedy-Large'  # Regular, Large, Greedy-Regular, Greedy-Large

RESUME = False  # Resume a previous run


def main():
    # Load Settings
    with open('settings/general_settings.json', 'r') as settings_file:
        settings = json.load(settings_file)

    with open('settings/agent-settings/' + MODEL + '-' + DESC + '_settings.json', 'r') as settings_file:
        agent_settings = json.load(settings_file)

    with open('settings/attack-settings/' + TEST + '-' + DESC + '_settings.json', 'r') as settings_file:
        attack_settings = json.load(settings_file)

    # Agent Settings
    num_envs = agent_settings['num-envs']

    # Identifiers of Experiment
    settings['env-id'] = settings['env-id'] + TEST + 'Env'
    settings['exp-name'] = settings['exp-name'] + TEST + 'Env' + '_' + DESC

    run_name = (str(settings['env-id']) + '_' + DESC + '_' + MODEL + '_' + str(settings['seed']) + '_'
                + str(num_envs))

    step_save = 'Wendigo-DVGA-' + MODEL + '-' + DESC + '-Step'

    if RESUME:
        model_path = "runs/" + run_name + "/" + settings['exp-name'] + "_"
        extension = ".pickle"

        try:
            agent = pickle.load(open((model_path + MODEL + '_MODEL' + extension), 'rb'))
            print(MODEL + ' model loaded from ' + (model_path + MODEL + '_MODEL' + extension))

            resume_data = pickle.load(open((model_path + MODEL + '_RESUME_DATA' + extension), 'rb'))
            print(MODEL + ' resume data loaded from ' + (model_path + MODEL + '_RESUME_DATA' + extension))

            global_step = resume_data[2]
            print('RESUMED AT GLOBAL STEP: ' + str(resume_data[2]))

        except FileNotFoundError:
            print('Settings to resume ' + run_name + ': NOT FOUND')
            exit(0)

    else:
        agent = None
        global_step = None
        resume_data = None

    # Tracking setup if using Weights and Biases
    if settings['track']:
        wandb_setup(project_name=settings['wandb-project-name'],
                    wandb_entity=settings['wandb-entity'],
                    run_name=run_name,
                    args=settings)

    # Writer setup for logging
    writer = writer_setup(run_name=run_name, settings=settings)

    # Random number seeding for reproducibility
    seeding(seed=settings['seed'])

    # Determine and setup device to run on (CPU or GPU)
    device = torch_setup(torch_deterministic=settings['torch-deterministic'], cuda=settings['cuda'])

    # Create the environments for the agent to interact with
    register(
        id=settings['env-id'],
        entry_point='Wendigo.environments.GraphQL' + TEST + 'Env' + ':GraphQL' + TEST + 'Env',
        kwargs={'schema': get_schema_from_file(schema_path=settings['schema-path']),
                'attack_settings': attack_settings,
                'connection_settings': settings['connection-settings'],
                'resume_data': (RESUME, resume_data),
                'step_save': step_save},
    )
    envs = make_envs(seed=settings['seed'], env_id=settings['env-id'], num_envs=num_envs)

    # Run Agent Main
    model = run_model(MODEL=MODEL, agent_settings=agent_settings, agent=agent, global_step=global_step,
                      envs=envs, writer=writer, device=device,
                      save_settings=(run_name, settings['exp-name']))

    if settings['save-model']:
        model.save_model(run_name=run_name, exp_name=settings['exp-name'])

    envs.close()
    writer.close()


if __name__ == '__main__':
    main()
