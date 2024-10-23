import re
from fvcore.common.config import CfgNode as _CfgNode

class CfgNode(_CfgNode):

    @classmethod
    def _open_cfg(cls, filename):
        return PathManager.open(filename, "r")

    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)


_C = CfgNode()
_C.DBG = False
_C.OUTPUT_DIR = './experiment/'
_C.GPU = 0
_C.NAME = 'test'
_C.EXP = ''

_C.DATA = CfgNode()
_C.DATA.NAME = 'CIFAR10'
_C.DATA.BATCH_SIZE = 16
_C.DATA.NUM = 512

_C.DEFENSE = CfgNode()
_C.DEFENSE.METHOD = 'diffpure'
_C.DEFENSE.CLASSIFIER_NAME = 'wrn'
_C.DEFENSE.DIFFUSION_NAME = 'edm_unet'

_C.DEFENSE.LLHD_MAXIMIZE = CfgNode()
_C.DEFENSE.LLHD_MAXIMIZE.N_LM = 5
_C.DEFENSE.LLHD_MAXIMIZE.EPS = 100
_C.DEFENSE.LLHD_MAXIMIZE.T_MIN = 0.4
_C.DEFENSE.LLHD_MAXIMIZE.T_MAX = 0.6
_C.DEFENSE.LLHD_MAXIMIZE.LR = 0.1
_C.DEFENSE.LLHD_MAXIMIZE.BETA1 = 0.9
_C.DEFENSE.LLHD_MAXIMIZE.BETA2 = 0.999

_C.DEFENSE.DIFFPURE = CfgNode()
_C.DEFENSE.DIFFPURE.SAMPLING_METHOD = 'ddpm'
_C.DEFENSE.DIFFPURE.DEF_MAX_TIMESTEPS = '100'
_C.DEFENSE.DIFFPURE.DEF_DENOISING_STEPS = '10'
_C.DEFENSE.DIFFPURE.ATT_MAX_TIMESTEPS = '100'
_C.DEFENSE.DIFFPURE.ATT_DENOISING_STEPS = '20'
_C.DEFENSE.DIFFPURE.DIFF_ATTACK = False
_C.DEFENSE.DIFFPURE.GUIDED = False
_C.DEFENSE.DIFFPURE.EPS = 8 / 255

_C.ATTACK = CfgNode()
_C.ATTACK.RESUME = False
_C.ATTACK.METHOD = 'apgd'
_C.ATTACK.N_ITERS = [50,50,50]
_C.ATTACK.NORM = 'inf'
_C.ATTACK.N_RESTART = 3
_C.ATTACK.RESTART_THR = 0.5
_C.ATTACK.N_EVAL = 10
_C.ATTACK.N_EOT = 1
_C.ATTACK.EPS = 8 / 255
_C.ATTACK.LOSS_NAMES = ['CW', 'CE', 'DLR']
_C.ATTACK.GRAD_MODE = 'full'
_C.ATTACK.PGD_CMD = ''
_C.ATTACK.PGD_STEP_SIZE = 0.007
_C.ATTACK.EM = True
_C.ATTACK.EM_ALPHA = 0.5
_C.ATTACK.EM_LAM = 5.0
_C.ATTACK.EM_STEPS = 5

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()


def setup(cfg_file=None, cfg_list=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    if cfg_file is not None: cfg.merge_from_file(cfg_file)
    if cfg_list is not None: cfg.merge_from_list(cfg_list)
    cfg = auto_setter(cfg)
    if cfg_file is not None: cfg.merge_from_file(cfg_file)
    if cfg_list is not None: cfg.merge_from_list(cfg_list)
    return cfg

def auto_setter(cfg):
    if cfg.DEFENSE.METHOD in ('llhd_maximize'):
        cfg.DEFENSE.DIFFUSION_NAME = 'edm_unet'
    if cfg.DEFENSE.METHOD in ('diffpure', 'GDMP'):
        cfg.DEFENSE.DIFFUSION_NAME = 'score_sde'
    if cfg.DEFENSE.DIFFPURE.GUIDED: 
        cfg.DEFENSE.DIFFPURE.EPS = cfg.ATTACK.EPS
        cfg.DEFENSE.DIFFPURE.DEF_MAX_TIMESTEPS = "36,36,36,36"
        cfg.DEFENSE.DIFFPURE.DEF_DENOISING_STEPS = "6,6,6,6"
        cfg.DEFENSE.DIFFPURE.ATT_MAX_TIMESTEPS = "36,36,36,36" 
        cfg.DEFENSE.DIFFPURE.ATT_DENOISING_STEPS = "12,12,12,12"
    if cfg.DATA.NAME == 'imagenette':
        cfg.DEFENSE.DIFFUSION_NAME = 'guided_diffusion'
        cfg.DEFENSE.CLASSIFIER_NAME = 'imt_resnet50'
        if cfg.DEFENSE.METHOD == 'diffpure':
            cfg.DEFENSE.DIFFPURE.DEF_MAX_TIMESTEPS = "50"
            cfg.DEFENSE.DIFFPURE.DEF_DENOISING_STEPS = "10"
            cfg.DEFENSE.DIFFPURE.ATT_MAX_TIMESTEPS = "50" 
            cfg.DEFENSE.DIFFPURE.ATT_DENOISING_STEPS = "50"
        if cfg.DEFENSE.DIFFPURE.GUIDED: 
            cfg.DEFENSE.DIFFPURE.DEF_MAX_TIMESTEPS = "30,30"
            cfg.DEFENSE.DIFFPURE.DEF_DENOISING_STEPS = "10,10"
            cfg.DEFENSE.DIFFPURE.ATT_MAX_TIMESTEPS = "30,30" 
            cfg.DEFENSE.DIFFPURE.ATT_DENOISING_STEPS = "30,30"
    if cfg.DBG: 
        cfg.DATA.NUM = 16
        cfg.ATTACK.N_ITERS = [50]
        cfg.ATTACK.N_EVAL = 5
    if cfg.ATTACK.METHOD in ('vmi', 'svre'):
        cfg.ATTACK.EM = True
    if (rs := cfg.ATTACK.N_RESTART) > 1:
        cfg.ATTACK.LOSS_NAMES = (cfg.ATTACK.LOSS_NAMES * rs)[:rs]
        cfg.ATTACK.N_ITERS = (cfg.ATTACK.N_ITERS * rs)[:rs]
    return cfg

def cmd_parse(cmd_str):
    cmd_list = cmd_str.split(' ')
    assert len(cmd_list) % 2 == 0
    cmd_dict = {}
    for k, v in zip(cmd_list[0::2], cmd_list[1::2]):
        if re.match(r'^[-+]?[0-9]+$', v): v = int(v)
        elif re.match(r'^[-+]?[0-9]+\.[0-9]*$', v): v = float(v)
        elif v.lower() == 'true': v = True
        elif v.lower() == 'false': v = False
        cmd_dict[k] = v
    return cmd_dict        
