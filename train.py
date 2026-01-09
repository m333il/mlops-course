import yaml
import argparse
import json
from src.trainer import create_trainer_from_config
from src.utils import init_logging, set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--metrics_output', type=str, default='metrics.json')
    args = parser.parse_args()
    
    init_logging()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_global_seed(config['run']['seed'])
    
    trainer = create_trainer_from_config(config)
    trainer.train()
    
    accuracy, report = trainer.evaluate()
    
    metrics = {
        "accuracy": accuracy,
        "num_epochs": config['training']['num_train_epochs'],
        "learning_rate": config['training']['learning_rate'],
        "batch_size": config['training']['batch_size']
    }
    
    with open(args.metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training complete. Accuracy: {accuracy:.4f}")
    print(f"Metrics saved to {args.metrics_output}")


if __name__ == "__main__":
    main()