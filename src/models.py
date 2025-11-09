import logging
from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
import torch


def get_model(model_name, id2label, label2id):
    logging.info(f"\nget_model: start")
    model = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=len(id2label),
        id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    logging.info(f"get_model: finished")
    return model


def get_image_processor(model_name):
    logging.info(f"\nget_image_processor: start")
    processor = AutoImageProcessor.from_pretrained(model_name)
    logging.info(f"get_image_processor: finished")
    return processor


def get_optimizer(model, config):
    logging.info(f"\nget_optimizer: start")
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, weight_decay=weight_decay
    )
    logging.info(f"get_optimizer: finish")
    return optimizer