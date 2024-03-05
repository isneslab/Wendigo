import random
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


def wandb_setup(project_name, wandb_entity, run_name, args):
    """
    Experiment will be tracked with Weights and Biases
    :param project_name: the weights and biases' project name
    :param wandb_entity: the entity (team) of weights and biases' project
    :param run_name: name to identify the particular run
    :param args: TODO Refactor to Remove
    :return:
    """
    wandb.init(
        project=project_name,
        entity=wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )


def writer_setup(run_name, settings):
    """
    Sets up writer which updates the console
    :param run_name: name to identify the particular run
    :param settings:  the dict of settings loaded from setting json file
    :return: writer
    """
    writer = SummaryWriter("runs/" + str(run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join(["|" + str(key) + "|" + str(value) + "|" for key, value in settings.items()])),
    )
    return writer


def seeding(seed):
    """
    Sets the seeds for random ness the same to allow reproducibility
    :param seed: The seed for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_setup(torch_deterministic=True, cuda=False):
    """
    Sets up torch
    :param torch_deterministic: if torch backend should be deterministic
    :param cuda: if torch should use cuda
    :return device: the torch device
    """
    torch.backends.cudnn.deterministic = torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    return device
