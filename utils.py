import sys
import yaml
import pickle
import argparse
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet50
from models.unets.EDM import get_edm_cifar_uncond
from models.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from models.SmallResolutionModel import WideResNet_70_16_dropout

import warnings
warnings.filterwarnings("ignore")
_DEFENSES = {}
_ATTACKS = {}


def register(cls=None, *, name=None, funcs='defenses'):
    """A decorator for registering model classes."""

    dic = eval('_' + funcs.upper())

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in dic:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        dic[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_log(log, dic, dim=0):
    for k, v in dic.items():
        v = v.detach()
        if k not in log:
            log[k] = v
        else:
            log[k] = torch.cat((log[k], v), dim)
    return log


def save_dict(dic, path):
    with open(path, 'wb') as f:
        pickle.dump(dic, f)


def load_dict(path):
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    return dic


def clamp(x, ori_x, eps, p=2):
    if p == 'inf':
        return torch.clamp(x, ori_x - eps, ori_x + eps)
    else:
        norm = torch.norm(x - ori_x, dim=(1, 2, 3), p=p, keepdim=True)
        rescale = torch.min(eps / norm, torch.ones_like(norm)).detach()
        return rescale * (x - ori_x) + ori_x


def dlr_loss(logits, y):
    logits_sorted, ind_sorted = logits.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(logits.shape[0])

    return -(logits[u, y] - logits_sorted[:, -2] * ind - logits_sorted[:, -1] * (
        1. - ind)) / (logits_sorted[:, -1] - logits_sorted[:, -3] + 1e-3)


def cw_loss(logits, y):
    idx_onehot = torch.nn.functional.one_hot(y, num_classes=logits.size()[1])
    loss = (logits - 1e5 * idx_onehot).max(1)[0] - (logits * idx_onehot).sum(1)
    return loss


def judge_success(logits, y, y_target=None):
    if y_target is None:
        return logits.max(1)[1] != y
    else:
        return logits.max(1)[1] == y_target


def get_model(model_name):
    if model_name == 'edm_unet':
        edm_unet = get_edm_cifar_uncond()
        edm_unet.load_state_dict(torch.load(
            "./resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
        return edm_unet.cuda().eval()
    elif model_name == 'score_sde':
        '''
        Follow https://github.com/NVlabs/DiffPure/blob/master/runners/diffpure_sde.py to load model,
        then save `ema` to score_sde_ema.pth to accerate the loading.
        '''
        diffusion = torch.load(
            './resources/checkpoints/score_sde/score_sde_ema.pth')
        return diffusion
    elif model_name == 'guided_diffusion':
        with open('./resources/configs/diffusion_configs/imagenet.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            config = dict2namespace(config)
            model_config = model_and_diffusion_defaults()
            model_config.update(vars(config.model))
            diffusion, _ = create_model_and_diffusion(**model_config)
            diffusion.load_state_dict(torch.load(
                './resources/checkpoints/guided_diffusion/256x256_diffusion_uncond.pt'))
            return diffusion.cuda().eval()
    elif model_name == 'imt_resnet50':
        model = resnet50()
        model.fc = torch.nn.Linear(2048, 10)
        model.load_state_dict(torch.load(
            './resources/checkpoints/models/resnet50.pt', map_location='cpu'))
        return model.cuda().eval() 
    elif model_name == 'wrn':
        return WideResNet_70_16_dropout().cuda().eval()


def get_defense(defense_method):
    return _DEFENSES[defense_method]


def get_attacker(attack_method):
    return _ATTACKS[attack_method]


def get_loss_func(loss_name):
    if loss_name == 'CE':
        return torch.nn.CrossEntropyLoss(reduction='none')
    if loss_name == 'DLR':
        return dlr_loss
    if loss_name == 'CW':
        return cw_loss


def get_dataloader(cfg):
    set_seed(0)
    if cfg.DATA.NAME == 'CIFAR10':
        test_loader = get_CIFAR10_test(batch_size=cfg.DATA.BATCH_SIZE)
        test_loader = [item for i, item in 
        enumerate(test_loader) if i < cfg.DATA.NUM // cfg.DATA.BATCH_SIZE]
    if cfg.DATA.NAME == 'imagenette':
        transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor(),])
        dataset = datasets.ImageFolder("./resources/datasets/imagenette2-160/val/", 
            transform=transform)
        test_loader = DataLoader(
            dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
        test_loader = [item for i, item in 
            enumerate(test_loader) if i < cfg.DATA.NUM // cfg.DATA.BATCH_SIZE]
    return test_loader


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_CIFAR10_test(batch_size=256, num_workers=8,
    pin_memory=True, transform=transforms.Compose([transforms.ToTensor(),])):
    dataset = datasets.CIFAR10(root='./resources/datasets/CIFAR10/', train=False, 
                               download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader
