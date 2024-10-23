import os
import sys
import logging
import argparse



if __name__ == '__main__':

    os.system('')
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file",
                        default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    from config import setup
    cfg = setup(cfg_list=args.opts)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU)

    import torch
    from defense import *
    from attacker import *
    from utils import *

    base_path = f'{cfg.OUTPUT_DIR}/{cfg.EXP or cfg.DEFENSE.METHOD}/{cfg.NAME}/'

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(base_path + 'exp.log', 'w'),
                  logging.FileHandler(base_path + 'info.log', 'w'),
                  logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger()
    logger.handlers[1].setLevel(logging.INFO)
    logger.handlers[2].setLevel(logging.INFO)
    seeder = iter(range(int(1e9)))

    logger.info('Configs:\n{:}\n{:}\n'.format(cfg, '-' * 30))
    df_config = cfg.DEFENSE[cfg.DEFENSE.METHOD.upper()]
    diffusion = get_model(cfg.DEFENSE.DIFFUSION_NAME).cuda().eval()
    classifier = get_model(cfg.DEFENSE.CLASSIFIER_NAME).cuda().eval()
    model = get_defense(cfg.DEFENSE.METHOD)(diffusion, classifier, df_config)
    test_loader = get_dataloader(cfg)
    attacker = get_attacker(cfg.ATTACK.METHOD)(model, cfg, logger, seeder)
    data, robustness = attacker.evaluate_robustness(test_loader)
    accuracy = attacker.evaluate_accuracy(test_loader)
    torch.save(data, base_path + 'data.pt')
    for k, v in {**accuracy, **robustness}.items():
        if 'loss' not in k:
            logger.info('{:13}: {:8.3%}'.format(k, v))
