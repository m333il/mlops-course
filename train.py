import yaml
import argparse
from src.trainer import create_trainer_from_config
from src.utils import init_logging, set_global_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    init_logging()
    set_global_seed(config['run']['seed'])
    
    trainer = create_trainer_from_config(config)
    trainer.train()