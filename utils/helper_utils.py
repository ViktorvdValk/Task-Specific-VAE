import random
import torch
import numpy as np
import os
from torch.nn import init
import git
from datetime import datetime

model_batchsize_dict = {
    "VAE_XAI": 6,
    "VAE_2D": 16,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_autocommit(version):
    """ Commit and push to git """
    print(f'-- Commit and Push to git {version} --')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"{version} run at {timestamp}"
    repo = git.Repo(os.getcwd())
    repo.git.add(update=True)
    repo.index.commit(commit_message)
    origin = repo.remote(name='origin')
    origin.push()


def init_weights(net, init_type='normal', gain=0.02, a=0):
    """Initialize network weights."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=a, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def set_deterministic():
    """ Set model deterministic """

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)


def set_all_seeds(seed):
    """ Set all seeds to make experiments reproducible """

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
