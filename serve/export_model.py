import argparse
import json
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        return outputs.logits


def export_to_torchscript(model_path, output_path):
    logging.info(f"Loading model from {model_path}")
    
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.eval()
    
    id2label = model.config.id2label
    logging.info(f"Model has {len(id2label)} classes")
    logging.info(f"Classes: {list(id2label.values())}")
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    logging.info("Tracing model with TorchScript...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, dummy_input)
    
    logging.info("Verifying traced model...")
    with torch.no_grad():
        original_output = wrapped_model(dummy_input)
        traced_output = traced_model(dummy_input)
        
        diff = torch.abs(original_output - traced_output).max().item()
        logging.info(f"Max difference between original and traced: {diff:.6f}")
        
    traced_model.save(output_path)
    logging.info(f"Model saved to {output_path}")
    
    return id2label


def save_label_mapping(id2label, output_path):
    id2label_str = {str(k): v for k, v in id2label.items()}
    
    with open(output_path, 'w') as f:
        json.dump(id2label_str, f, indent=2)
    
    logging.info(f"Label mapping saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/resnet-18"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="serve/model.pt"
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="serve/index_to_name.json"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    id2label = export_to_torchscript(args.model_path, args.output_path)
    save_label_mapping(id2label, args.labels_path)
    
    logging.info("Export complete!")


if __name__ == "__main__":
    main()