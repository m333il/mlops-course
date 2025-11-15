import pytest
from PIL import Image
import numpy as np
from datasets import Dataset, Features, ClassLabel, Image as HFImage
from unittest.mock import MagicMock
import torch


@pytest.fixture(scope="session")
def mock_config():
    return {
        'data': {
            'dataset_name': 'SaffalPoosh/deepFashion-with-masks',
            'image_column': 'images',
            'label_column': 'cloth_type',
            'split_name': 'train',
            'validation_size': 0.1
        },
        'model': {
            'name': 'microsoft/resnet-18'
        },
        'training': {
            'output_dir': './models/resnet-18',
            'num_train_epochs': 5,
            'learning_rate': 5.0e-5,
            'weight_decay': 0.01,
            'batch_size': 32,
            'device': 'cuda:4'
        },
        'run': {
            'seed': 42
        }
    }


@pytest.fixture(scope="session")
def mock_fashion_dataset():
    cloth_types = ['0', '1', '2']
    
    num_samples = 50
    features = Features({
        'images': HFImage(),
        'cloth_type': ClassLabel(num_classes=len(cloth_types), names=cloth_types)
    })
    
    data = {
        "images": [
            Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
            for _ in range(num_samples)
        ],
        "cloth_type": np.random.randint(0, len(cloth_types), num_samples).tolist(),
    }
    
    ds = Dataset.from_dict(data, features=features)
    return ds


@pytest.fixture
def mock_image_processor():
    mock = MagicMock()
    def mock_processor_call(images, return_tensors=None):
        num_images = len(images) if isinstance(images, list) else 1
        return {"pixel_values": torch.stack([torch.randn(3, 224, 224) for _ in range(num_images)])}
    
    mock.side_effect = mock_processor_call
    mock.save_pretrained = MagicMock()
    return mock
