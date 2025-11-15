import pytest
import torch
from datasets import Dataset, Features, ClassLabel, Image as HFImage
from PIL import Image
import numpy as np

from src.validation import (
    ValidationError,
    validate_image_tensor,
    validate_config,
    validate_dataset_split
)

def test_validate_image_tensor_valid_4d():
    tensor = torch.randn(2, 3, 224, 224)
    validate_image_tensor(tensor, expected_dims=4, expected_channels=3)


def test_validate_image_tensor_invalid_type():
    not_a_tensor = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(ValidationError, match="Expected torch.Tensor"):
        validate_image_tensor(not_a_tensor)


def test_validate_image_tensor_custom_min_dimensions():
    tensor = torch.randn(2, 3, 100, 100)
    with pytest.raises(ValidationError, match="Image height 100 is below minimum 200"):
        validate_image_tensor(tensor, min_height=200, min_width=100)


def test_validate_config_valid(mock_config):
    validate_config(mock_config)


def test_validate_config_missing_data_key():
    config = {
        'data': {
            'dataset_name': 'SaffalPoosh/deepFashion-with-masks',
            'image_column': 'images',
            # missing 'label_column'
            'split_name': 'train',
            'validation_size': 0.1
        },
        'model': {'name': 'microsoft/resnet-18'},
        'training': {
            'output_dir': './models/resnet-18',
            'num_train_epochs': 5,
            'learning_rate': 5.0e-5,
            'weight_decay': 0.01,
            'batch_size': 32,
            'device': 'cuda:4'
        },
        'run': {'seed': 42}
    }
    
    with pytest.raises(ValidationError, match="Missing required key in 'data' section: label_column"):
        validate_config(config)


def test_validate_dataset_split_valid(mock_fashion_dataset):
    required_columns = ['images', 'cloth_type']
    validate_dataset_split(mock_fashion_dataset, required_columns, min_samples=1)


def test_validate_dataset_split_missing_columns(mock_fashion_dataset):
    required_columns = ['images', 'cloth_type', 'nonexistent_column']
    
    with pytest.raises(ValidationError, match="Dataset missing required columns"):
        validate_dataset_split(mock_fashion_dataset, required_columns)


def test_validate_dataset_split_empty_dataset():
    features = Features({
        'images': HFImage(),
        'cloth_type': ClassLabel(num_classes=3, names=['Dresses', 'Shorts', 'Tees_Tanks'])
    })
    
    data = {
        "images": [],
        "cloth_type": [],
    }
    
    empty_dataset = Dataset.from_dict(data, features=features)
    
    with pytest.raises(ValidationError, match="Dataset has 0 samples"):
        validate_dataset_split(empty_dataset, ['images', 'cloth_type'], min_samples=1)