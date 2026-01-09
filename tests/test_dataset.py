import torch
from torch.utils.data import DataLoader
from unittest.mock import MagicMock, patch
from datasets import Dataset, ClassLabel
from PIL import Image
import numpy as np

from src.dataset import (
    prepare_label_mappings,
    convert_labels,
    split_dataset,
    create_dataset,
)


def test_prepare_label_mappings():
    unique_labels = ['Dresses', 'Shorts', 'Tees_Tanks']
    
    label2id, id2label = prepare_label_mappings(unique_labels)
    
    assert label2id == {'Dresses': 0, 'Shorts': 1, 'Tees_Tanks': 2}
    assert id2label == {0: 'Dresses', 1: 'Shorts', 2: 'Tees_Tanks'}


def test_convert_labels():
    data = {
        "images": [
            Image.fromarray(np.uint8(np.random.rand(5, 5, 3) * 255)) 
            for _ in range(5)
        ],
        "cloth_type": ['Dresses', 'Shorts', 'Dresses', 'Tees_Tanks', 'Shorts'],
    }
    dataset = Dataset.from_dict(data)
    converted_dataset, unique_labels = convert_labels(dataset, 'cloth_type')
    
    assert set(unique_labels) == {'Dresses', 'Shorts', 'Tees_Tanks'}
    
    assert isinstance(converted_dataset.features['cloth_type'], ClassLabel)
    
    assert all(isinstance(label, int) for label in converted_dataset['cloth_type'])


def test_split_dataset(mock_fashion_dataset):
    test_size = 0.2
    
    train_split, val_split = split_dataset(
        mock_fashion_dataset, 
        test_size=test_size, 
        seed=42,
        label_column='cloth_type'
    )
    
    total_size = len(mock_fashion_dataset)
    expected_val_size = int(total_size * test_size)
    expected_train_size = total_size - expected_val_size
    
    assert len(train_split) == expected_train_size
    assert len(val_split) == expected_val_size


def test_create_dataset(mock_fashion_dataset, mock_image_processor):
    batch_size = 8
    
    loader = create_dataset(
        mock_fashion_dataset,
        mock_image_processor,
        image_column='images',
        label_column='cloth_type',
        batch_size=batch_size,
        shuffle=False,
        validate_samples=False
    )

    assert isinstance(loader, DataLoader)

    batch = next(iter(loader))
    assert batch['pixel_values'].shape[0] == batch_size
    assert batch['labels'].shape[0] == batch_size

    assert 'pixel_values' in batch
    assert 'labels' in batch
    
    assert isinstance(batch['pixel_values'], torch.Tensor)
    assert isinstance(batch['labels'], torch.Tensor)
    
    assert batch['labels'].dtype == torch.int64
