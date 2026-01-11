# python -m src.predict --input_path /path/to/images --output_path /path/to/preds.csv

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

from src.utils import init_logging


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def load_model(model_path, device):
    logging.info(f"Loading model from {model_path}")
    
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    logging.info(f"Model loaded successfully. Classes: {list(model.config.id2label.values())}")
    return model, processor


def get_image_files(input_path):
    input_dir = Path(input_path)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_path}")
    
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    
    if not image_files:
        raise ValueError(f"No supported images found in {input_path}. "
                        f"Supported formats: {SUPPORTED_EXTENSIONS}")
    
    logging.info(f"Found {len(image_files)} images in {input_path}")
    return sorted(image_files)


def predict_single_image(
    image_path,
    model,
    processor,
    device
):
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_idx = outputs.logits.argmax(dim=-1).item()
    
    predicted_label = model.config.id2label[predicted_idx]
    return predicted_label


def predict_batch(
    image_files,
    model,
    processor,
    device
):
    results = []
    
    for image_path in image_files:
        try:
            predicted_label = predict_single_image(
                image_path, model, processor, device
            )
            results.append({
                "image_name": image_path.name,
                "predicted_label": predicted_label
            })
            logging.debug(f"{image_path.name} -> {predicted_label}")
            
        except Exception as e:
            logging.warning(f"Failed to process {image_path.name}: {e}")
            results.append({
                "image_name": image_path.name,
                "predicted_label": "ERROR"
            })
    
    return results


def save_predictions(results, output_path):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    logging.info(f"Predictions saved to {output_path}")
    logging.info(f"Total predictions: {len(results)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to directory containing input images"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output CSV file for predictions"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/resnet-18",
        help="Path to trained model directory (default: models/resnet-18)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (default: cpu)"
    )
    
    args = parser.parse_args()
    
    init_logging()
    logging.info("Starting offline inference")
    logging.info(f"Input: {args.input_path}")
    logging.info(f"Output: {args.output_path}")
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Device: {args.device}")
    
    device = torch.device(args.device)
    
    model, processor = load_model(args.model_path, device)
    
    image_files = get_image_files(args.input_path)
    
    results = predict_batch(image_files, model, processor, device)
    
    save_predictions(results, args.output_path)
    
    logging.info("Inference complete!")


if __name__ == "__main__":
    main()
