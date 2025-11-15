import torch
import logging
from typing import Dict, Any, List, Optional


class ValidationError(Exception):
    """
        custom exception for validation errors.
    """
    pass


def validate_image_tensor(
    tensor: torch.Tensor, 
    expected_dims: int = 4,
    expected_channels: int = 3,
    min_height: int = 1,
    min_width: int = 1
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(f"Expected torch.Tensor, got {type(tensor)}")
    
    if tensor.dim() != expected_dims:
        raise ValidationError(
            f"Expected {expected_dims}D tensor (batch, channels, height, width), "
            f"got {tensor.dim()}D tensor with shape {tensor.shape}"
        )
    
    if expected_dims == 4:
        batch_size, channels, height, width = tensor.shape
        
        if channels != expected_channels:
            raise ValidationError(
                f"Expected {expected_channels} channels, got {channels}. "
                f"Tensor shape: {tensor.shape}"
            )
        
        if height < min_height:
            raise ValidationError(
                f"Image height {height} is below minimum {min_height}. "
                f"Tensor shape: {tensor.shape}"
            )
        
        if width < min_width:
            raise ValidationError(
                f"Image width {width} is below minimum {min_width}. "
                f"Tensor shape: {tensor.shape}"
            )
    logging.debug(
        f"Image tensor validated: shape={tensor.shape}"
    )


def validate_config(config: Dict[str, Any]) -> None:
    required_sections = ['data', 'model', 'training', 'run']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValidationError(
            f"Configuration missing required sections: {missing_sections}"
        )
    
    data_required = ['dataset_name', 'image_column', 'label_column', 'split_name', 'validation_size']
    for key in data_required:
        if key not in config['data']:
            raise ValidationError(f"Missing required key in 'data' section: {key}")
    
    validation_size = config['data']['validation_size']
    if not (0 < validation_size < 1):
        raise ValidationError(
            f"validation_size must be in range (0, 1), got {validation_size}"
        )
    
    if 'name' not in config['model']:
        raise ValidationError("Missing required key in 'model' section: name")
    
    training_required = ['output_dir', 'num_train_epochs', 'learning_rate', 'weight_decay', 'batch_size', 'device']
    for key in training_required:
        if key not in config['training']:
            raise ValidationError(f"Missing required key in 'training' section: {key}")
    
    lr = config['training']['learning_rate']
    if lr <= 0:
        raise ValidationError(f"learning_rate must be positive, got {lr}")
    
    weight_decay = config['training']['weight_decay']
    if weight_decay < 0:
        raise ValidationError(f"weight_decay must be non-negative, got {weight_decay}")
    
    batch_size = config['training']['batch_size']
    if batch_size <= 0:
        raise ValidationError(f"batch_size must be positive, got {batch_size}")
    
    num_epochs = config['training']['num_train_epochs']
    if num_epochs <= 0:
        raise ValidationError(f"num_train_epochs must be positive, got {num_epochs}")
    
    if 'seed' not in config['run']:
        raise ValidationError("Missing required key in 'run' section: seed")
    
    seed = config['run']['seed']
    if not isinstance(seed, int):
        raise ValidationError(f"seed must be an integer, got {type(seed)}")
    
    logging.info("Configuration validated successfully")


def validate_dataset_split(
    dataset,
    required_columns: List[str],
    min_samples: int = 1
) -> None:
    if not hasattr(dataset, 'column_names'):
        raise ValidationError(
            "Dataset must have 'column_names' attribute. "
            "Is this a valid HuggingFace dataset?"
        )
    
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    if missing_columns:
        raise ValidationError(
            f"Dataset missing required columns: {missing_columns}. "
            f"Available columns: {dataset.column_names}"
        )
    
    dataset_size = len(dataset)
    if dataset_size < min_samples:
        raise ValidationError(
            f"Dataset has {dataset_size} samples, "
            f"but minimum required is {min_samples}"
        )
    
    logging.debug(
        f"Dataset validated: size={dataset_size}, "
        f"columns={dataset.column_names}"
    )

