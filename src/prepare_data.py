import yaml
import argparse
import logging
import os
import json
from datasets import load_dataset

from src.utils import init_logging, set_global_seed


def prepare_data(config):
    """
    download and prepare dataset
    """
    logging.info("prepare_data: start")
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Loading dataset: {config['data']['dataset_name']}")
    dataset = load_dataset(
        config['data']['dataset_name'],
        split=config['data']['split_name']
    )
    
    logging.info(f"Dataset loaded: {len(dataset)} samples")
    logging.info(f"Columns: {dataset.column_names}")
    
    label_column = config['data']['label_column']
    unique_labels = sorted(set(dataset[label_column]))
    logging.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    
    dataset_path = os.path.join(output_dir, "dataset")
    dataset.save_to_disk(dataset_path)
    logging.info(f"Dataset saved to {dataset_path}")
    
    metadata = {
        "num_samples": len(dataset),
        "num_labels": len(unique_labels),
        "labels": unique_labels,
        "columns": dataset.column_names,
        "dataset_name": config['data']['dataset_name'],
        "split_name": config['data']['split_name']
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Metadata saved to {metadata_path}")
    
    logging.info("prepare_data: finish")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    args = parser.parse_args()
    
    init_logging()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_global_seed(config['run']['seed'])
    
    metadata = prepare_data(config)
    logging.info(f"Data preparation complete. Stats: {metadata}")


if __name__ == "__main__":
    main()