import pytest
import torch
from unittest.mock import MagicMock, patch

from src.models import (
    get_model,
    get_image_processor,
    get_optimizer
)

@patch('src.models.AutoModelForImageClassification.from_pretrained')
def test_get_model(mock_from_pretrained, mock_config):
    mock_model = MagicMock()
    mock_from_pretrained.return_value = mock_model
    
    id2label = {0: 'Dresses', 1: 'Shorts', 2: 'Tees_Tanks'}
    label2id = {'Dresses': 0, 'Shorts': 1, 'Tees_Tanks': 2}
    
    model = get_model(
        mock_config['model']['name'],
        id2label=id2label,
        label2id=label2id
    )
    
    mock_from_pretrained.assert_called_once_with(
        mock_config['model']['name'],
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    assert model == mock_model


@patch('src.models.AutoImageProcessor.from_pretrained')
def test_get_image_processor(mock_from_pretrained, mock_config):
    mock_processor = MagicMock()
    mock_from_pretrained.return_value = mock_processor
    
    processor = get_image_processor(mock_config['model']['name'])
    
    mock_from_pretrained.assert_called_once_with(mock_config['model']['name'])
    assert processor == mock_processor


def test_get_optimizer(mock_config):
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.randn(10, 10)]

    optimizer = get_optimizer(mock_model, mock_config['training'])

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]['lr'] == mock_config['training']['learning_rate']
    assert optimizer.param_groups[0]['weight_decay'] == mock_config['training']['weight_decay']


@patch('src.models.AutoModelForImageClassification.from_pretrained')
def test_full_model_setup_flow(mock_from_pretrained, mock_config):
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.randn(10, 10)]
    mock_from_pretrained.return_value = mock_model
    
    id2label = {0: 'Dresses', 1: 'Shorts'}
    label2id = {'Dresses': 0, 'Shorts': 1}
    
    model = get_model(mock_config['model']['name'], id2label, label2id)
    optimizer = get_optimizer(model, mock_config['training'])
    
    assert model is not None
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    mock_model.parameters.assert_called()