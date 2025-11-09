import logging
from datasets import load_dataset, ClassLabel
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor


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


def create_dataset(split, processor, image_column, label_column, batch_size, shuffle):
    def _transform(examples):
        images = [img.convert("RGB") for img in examples[image_column]]
        examples["pixel_values"] = processor(images, return_tensors="pt")["pixel_values"]
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
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    return loader


def get_dataloaders(config):
    logging.info("get_dataloaders: start")

    dataset = load_dataset(
        config['data']['dataset_name'],
        split=config['data']['split_name']
    )
    dataset, unique_labels = convert_labels(dataset, config['data']['label_column'])

    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    logging.info(f"get_dataloaders: labels: {unique_labels}")

    train_split, val_split = split_dataset(
        dataset,
        config['data']['validation_size'],
        config['run']['seed'],
        config['data']['label_column']

    )
    logging.info(f"get_dataloaders: train size: {len(train_split)}")
    logging.info(f"get_dataloaders: validation size: {len(val_split)}")

    processor = AutoImageProcessor.from_pretrained(config['model']['name'])

    train_loader = create_dataset(
        train_split, processor, 
        config['data']['image_column'], 
        config['data']['label_column'], 
        config['training']['batch_size'], 
        True
    )

    val_loader = create_dataset(
        val_split, processor, 
        config['data']['image_column'], 
        config['data']['label_column'], 
        config['training']['batch_size'], 
        False
    )
    logging.info("get_dataloaders: finish")
    return train_loader, val_loader, processor, id2label, label2id