import yaml
import argparse
from src.trainer import Trainer
from src.utils import init_logging, set_global_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    init_logging()
    set_global_seed(config['run']['seed'])
    
    trainer = Trainer(config)
    trainer.train()