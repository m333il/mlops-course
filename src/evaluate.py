import yaml
import argparse
import logging
import json
import os
import torch
from datasets import load_from_disk, ClassLabel
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForImageClassification, AutoImageProcessor
from tqdm import tqdm

from src.utils import init_logging, set_global_seed
from src.dataset import create_dataset, split_dataset


def evaluate_model(config):
    """
    evaluate trained model on validation set.
    """
    logging.info("evaluate_model: start")
    
    model_path = config['training']['output_dir']
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"Loading model from {model_path}")
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    logging.info("Loading processed dataset")
    dataset = load_from_disk("data/processed/dataset")
    
    label_column = config['data']['label_column']
    unique_labels = sorted(set(dataset[label_column]))
    
    if not isinstance(dataset.features[label_column], ClassLabel):
        logging.info("Converting label column to ClassLabel for stratified split")
        dataset = dataset.cast_column(label_column, ClassLabel(names=unique_labels))
    
    _, val_split = split_dataset(
        dataset,
        test_size=config['data']['validation_size'],
        seed=config['run']['seed'],
        label_column=label_column
    )
    
    val_loader = create_dataset(
        val_split,
        processor,
        image_column=config['data']['image_column'],
        label_column=label_column,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        validate_samples=False
    )
    
    logging.info("Running predictions on validation set")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']
            
            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=-1).cpu()
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    id2label = model.config.id2label
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    
    report = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        "accuracy": accuracy,
        "num_samples": len(all_labels),
        "num_classes": len(label_names),
        "per_class_metrics": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1-score": report[label]["f1-score"],
                "support": report[label]["support"]
            }
            for label in label_names if label in report
        }
    }
    
    logging.info(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    logging.info("evaluate_model: finish")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--output', type=str, default='evaluation_metrics.json')
    args = parser.parse_args()
    
    init_logging()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_global_seed(config['run']['seed'])
    
    metrics = evaluate_model(config)
    
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()