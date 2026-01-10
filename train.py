import yaml
import argparse
import json
import os
import hashlib
from tqdm import trange

import mlflow
import mlflow.transformers

from src.trainer import create_trainer_from_config
from src.utils import init_logging, set_global_seed


def get_dvc_data_hash():
    try:
        with open('dvc.lock', 'r') as f:
            content = f.read()
        return hashlib.md5(content.encode()).hexdigest()[:12]
    except FileNotFoundError:
        return 'no_dvc_lock'


def get_git_commit():
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()[:8]
    except Exception:
        return 'unknown'


def setup_mlflow(config):
    tracking_uri = os.environ.get(
        'MLFLOW_TRACKING_URI',
        'https://dagshub.com/m333il/mlops-course.mlflow'
    )
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = config.get('mlflow', {}).get(
        'experiment_name', 
        'fashion-classification'
    )
    mlflow.set_experiment(experiment_name)
    
    return tracking_uri


def log_params_from_config(config):
    mlflow.log_params({
        'model_name': config['model']['name'],
        'dataset_name': config['data']['dataset_name'],
        'image_column': config['data']['image_column'],
        'label_column': config['data']['label_column'],
        'validation_size': config['data']['validation_size'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'batch_size': config['training']['batch_size'],
        'num_epochs': config['training']['num_train_epochs'],
        'device': config['training']['device'],
        'seed': config['run']['seed'],
    })


def log_tags():
    #log tags to connect DVC and Git.
    dvc_hash = get_dvc_data_hash()
    git_hash = get_git_commit()
    mlflow.set_tag('dvc_data_hash', dvc_hash)
    mlflow.set_tag('git_commit', git_hash)
    mlflow.set_tag('mlflow.runName', f'run_{git_hash}_{dvc_hash[:6]}')


def log_model_with_transformers(trainer):
    mlflow.transformers.log_model(
        transformers_model={
            'model': trainer.model,
            'image_processor': trainer.processor,
        },
        artifact_path='model',
        task='image-classification',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--metrics_output', type=str, default='metrics.json')
    args = parser.parse_args()
    
    init_logging()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_global_seed(config['run']['seed'])

    tracking_uri = setup_mlflow(config)
    print(f'MLflow tracking URI: {tracking_uri}')

    with mlflow.start_run():
        log_params_from_config(config)
        log_tags()

        mlflow.log_artifact(args.config, artifact_path='config')

        trainer = create_trainer_from_config(config)

        for epoch in trange(config['training']['num_train_epochs'], desc='Training', leave=False):
            avg_train_loss = trainer._one_epoch(epoch)
            mlflow.log_metric('train_loss', avg_train_loss, step=epoch)

            accuracy, report = trainer.evaluate()
            mlflow.log_metric('val_accuracy', accuracy, step=epoch)

            print(f'Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_accuracy={accuracy:.4f}')

        trainer.save_model()

        final_accuracy, final_report = trainer.evaluate()

        mlflow.log_metric('final_accuracy', final_accuracy)

        metrics = {
            'accuracy': final_accuracy,
            'num_epochs': config['training']['num_train_epochs'],
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['training']['batch_size'],
        }

        with open(args.metrics_output, 'w') as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(args.metrics_output)

        if os.path.exists('dvc.lock'):
            mlflow.log_artifact('dvc.lock')

        log_model_with_transformers(trainer)

        run_id = mlflow.active_run().info.run_id
        print(f'MLflow run ID: {run_id}')
        print(f'Training complete. Final accuracy: {final_accuracy:.4f}')
        print(f'Metrics saved to {args.metrics_output}')


if __name__ == '__main__':
    main()