import pytest
import torch
from unittest.mock import MagicMock, patch, call
from torch.utils.data import DataLoader, Dataset
from src.trainer import Trainer, create_trainer_from_config


class DummyDataset(Dataset):
    def __init__(self, num_samples=20):
        self.num_samples = num_samples
        self.pixel_values = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.randint(0, 3, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "pixel_values": self.pixel_values[idx],
            "labels": self.labels[idx]
        }


@patch('src.trainer.get_dataloaders')
@patch('src.trainer.get_model')
@patch('src.trainer.get_optimizer')
def test_create_trainer_from_config(
    mock_get_optimizer,
    mock_get_model,
    mock_get_dataloaders,
    mock_config
):
    dummy_dataset = DummyDataset(num_samples=20)
    train_loader = DataLoader(dummy_dataset, batch_size=4)
    val_loader = DataLoader(dummy_dataset, batch_size=4)
    mock_processor = MagicMock()
    mock_processor.save_pretrained = MagicMock()
    
    id2label = {0: 'Dresses', 1: 'Shorts', 2: 'Tees_Tanks'}
    label2id = {'Dresses': 0, 'Shorts': 1, 'Tees_Tanks': 2}
    
    mock_get_dataloaders.return_value = (
        train_loader, val_loader, mock_processor, id2label, label2id
    )
    
    mock_model = MagicMock()
    mock_model.config.id2label = id2label
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.train = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.save_pretrained = MagicMock()
    
    def mock_forward(**kwargs):
        batch_size = kwargs['pixel_values'].shape[0]
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        mock_output.logits = torch.randn(batch_size, 3, requires_grad=True)
        return mock_output
    
    mock_model.side_effect = mock_forward
    mock_get_model.return_value = mock_model
    
    mock_optimizer = MagicMock()
    mock_get_optimizer.return_value = mock_optimizer
    
    trainer = create_trainer_from_config(mock_config)
    
    assert mock_get_dataloaders.call_count == 1
    assert mock_get_model.call_count == 1
    assert mock_get_optimizer.call_count == 1

    assert isinstance(trainer, Trainer)
    assert trainer.model == mock_model
    assert trainer.optimizer == mock_optimizer
    assert trainer.id2label == id2label
    assert trainer.label2id == label2id


@patch('src.trainer.get_dataloaders')
@patch('src.trainer.get_model')
@patch('src.trainer.get_optimizer')
def test_full_training_pipeline(
    mock_get_optimizer,
    mock_get_model,
    mock_get_dataloaders,
    mock_config
):
    dummy_dataset = DummyDataset(num_samples=20)
    train_loader = DataLoader(dummy_dataset, batch_size=4)
    val_loader = DataLoader(dummy_dataset, batch_size=4)
    mock_processor = MagicMock()
    mock_processor.save_pretrained = MagicMock()
    
    id2label = {0: 'Dresses', 1: 'Shorts', 2: 'Tees_Tanks'}
    label2id = {'Dresses': 0, 'Shorts': 1, 'Tees_Tanks': 2}
    
    mock_get_dataloaders.return_value = (
        train_loader, val_loader, mock_processor, id2label, label2id
    )
    
    mock_model = MagicMock()
    mock_model.config.id2label = id2label
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.train = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.save_pretrained = MagicMock()
    
    def mock_forward(**kwargs):
        batch_size = kwargs['pixel_values'].shape[0]
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        mock_output.logits = torch.randn(batch_size, 3, requires_grad=True)
        return mock_output
    
    mock_model.side_effect = mock_forward
    mock_get_model.return_value = mock_model
    
    mock_optimizer = MagicMock()
    mock_get_optimizer.return_value = mock_optimizer
    
    test_config = mock_config.copy()
    test_config['training'] = mock_config['training'].copy()
    test_config['training']['num_train_epochs'] = 2
    
    trainer = create_trainer_from_config(test_config)
    trainer.train(save_model=False)
    
    assert mock_model.train.call_count >= 2
    assert mock_model.eval.call_count >= 2
    assert mock_optimizer.zero_grad.call_count > 0
    assert mock_optimizer.step.call_count > 0
