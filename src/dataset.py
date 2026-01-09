import logging
from datasets import load_dataset, ClassLabel
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from typing import Tuple, Dict
import os

from src.validation import (
    validate_dataset_split,
    validate_image_tensor,
    validate_config
)

def load_processed_dataset(processed_path="data/processed/dataset"):
    """load dataset from disk"""
    from datasets import load_from_disk, ClassLabel
    logging.info(f"Loading processed dataset from {processed_path}")
    dataset = load_from_disk(processed_path)
    return dataset


def load_and_validate_dataset(config: Dict) -> Tuple:
    logging.info("load_and_validate_dataset: start")
    
    dataset = load_dataset(
        config['data']['dataset_name'],
        split=config['data']['split_name']
    )
    
    required_columns = [
        config['data']['image_column'],
        config['data']['label_column']
    ]
    validate_dataset_split(dataset, required_columns=required_columns, min_samples=10)
    logging.info(f"Dataset loaded: {len(dataset)} samples with columns {dataset.column_names}")
    
    dataset, unique_labels = convert_labels(dataset, config['data']['label_column'])
    logging.info(f"load_and_validate_dataset: finish with {len(unique_labels)} classes")
    return dataset, unique_labels


def prepare_label_mappings(unique_labels: list) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    logging.debug(f"Created label mappings for {len(unique_labels)} classes")
    return label2id, id2label


def convert_labels(dataset, label_column):
    unique_labels = sorted(set(dataset[label_column]))
    class_label_feature = ClassLabel(names=unique_labels)
    dataset = dataset.cast_column(label_column, class_label_feature)
    return dataset, unique_labels


def split_dataset(dataset, test_size, seed, label_column):
    res = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
        stratify_by_column=label_column
    )
    return res['train'], res['test']


def create_dataset(split, processor, image_column, label_column, batch_size, shuffle, validate_samples=True):
    def _transform(examples):
        images = [img.convert("RGB") for img in examples[image_column]]
        processed = processor(images, return_tensors="pt")
        examples["pixel_values"] = processed["pixel_values"]
        
        if validate_samples and len(images) > 0:
            try:
                pixel_values = examples["pixel_values"]
                if pixel_values.dim() == 4:
                    sample = pixel_values[0:1]
                    validate_image_tensor(
                        sample,
                        expected_dims=4,
                        expected_channels=3,
                        min_height=1,
                        min_width=1
                    )
            except Exception as e:
                logging.warning(f"Image validation warning during transformation: {e}")
        
        return examples

    def _collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x[label_column] for x in batch]),
        }
    
    dataset = split.with_transform(_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        shuffle=shuffle,
        num_workers=0,
    )
    
    logging.debug(f"Created DataLoader: batch_size={batch_size}, shuffle={shuffle}")
    return loader


def get_dataloaders(config, use_local=True):
    logging.info("get_dataloaders: start")

    validate_config(config)
    
    label_column = config['data']['label_column']
    
    if use_local and os.path.exists("data/processed/dataset"):
        dataset = load_processed_dataset("data/processed/dataset")

        unique_labels = sorted(set(dataset[label_column]))
        
        if not isinstance(dataset.features[label_column], ClassLabel):
            logging.info("Converting label column to ClassLabel")
            class_label_feature = ClassLabel(names=unique_labels)
            dataset = dataset.cast_column(label_column, class_label_feature)
        
        logging.info(f"Loaded local dataset: {len(dataset)} samples")
    else:
        dataset, unique_labels = load_and_validate_dataset(config)
    
    label2id, id2label = prepare_label_mappings(unique_labels)
    logging.info(f"get_dataloaders: labels: {unique_labels}")

    train_split, val_split = split_dataset(
        dataset,
        config['data']['validation_size'],
        config['run']['seed'],
        label_column
    )
    logging.info(f"get_dataloaders: train size: {len(train_split)}")
    logging.info(f"get_dataloaders: validation size: {len(val_split)}")

    processor = AutoImageProcessor.from_pretrained(config['model']['name'])
    logging.info(f"get_dataloaders: loaded processor from {config['model']['name']}")

    train_loader = create_dataset(
        train_split, processor, 
        config['data']['image_column'], 
        label_column, 
        config['training']['batch_size'], 
        shuffle=True,
        validate_samples=True
    )

    val_loader = create_dataset(
        val_split, processor, 
        config['data']['image_column'], 
        label_column, 
        config['training']['batch_size'], 
        shuffle=False,
        validate_samples=True
    )
    logging.info("get_dataloaders: finish")
    return train_loader, val_loader, processor, id2label, label2id