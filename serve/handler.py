import io
import logging
import os
import json
import time
import uuid
from datetime import datetime, timezone
import torch
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown") # set during Docker build


class FashionClassifierHandler(BaseHandler):
    ID2LABEL = {
        0: "Blouses_Shirts",
        1: "Cardigans",
        2: "Denim",
        3: "Dresses",
        4: "Graphic_Tees",
        5: "Jackets_Coats",
        6: "Jackets_Vests",
        7: "Leggings",
        8: "Pants",
        9: "Rompers_Jumpsuits",
        10: "Shirts_Polos",
        11: "Shorts",
        12: "Skirts",
        13: "Suiting",
        14: "Sweaters",
        15: "Sweatshirts_Hoodies",
        16: "Tees_Tanks",
    }
    
    def __init__(self):
        super().__init__()
        self.transform = None
        self.initialized = False
        self.model_version = MODEL_VERSION
    
    def initialize(self, context):
        super().initialize(context)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.initialized = True
        logger.info(f"FashionClassifierHandler initialized successfully. Model version: {self.model_version}")
        logger.info(f"Number of classes: {len(self.ID2LABEL)}")
    
    def _log_request(
        self, request_id, status,
        latency_ms, batch_size=1, 
        error_message=None
    ):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model_version": self.model_version,
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "input amount": batch_size,
        }
        
        if error_message:
            log_entry["error"] = error_message
        
        logger.info(json.dumps(log_entry))
    
    def handle(self, data, context):
        request_id = str(uuid.uuid4())
        start_time = time.time()
        batch_size = len(data) if data else 0
        
        try:
            result = super().handle(data, context)
            
            latency_ms = (time.time() - start_time) * 1000
            self._log_request(
                request_id=request_id,
                status="success",
                latency_ms=latency_ms,
                batch_size=batch_size
            )
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._log_request(
                request_id=request_id,
                status="error",
                latency_ms=latency_ms,
                batch_size=batch_size,
                error_message=str(e)
            )
            raise
    
    def preprocess(self, data):
        images = []
        
        for row in data:
            image_data = row.get("data") or row.get("body")
            
            if isinstance(image_data, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = Image.open(image_data)
            
            image = image.convert("RGB")
            
            image_tensor = self.transform(image)
            images.append(image_tensor)
        
        batch = torch.stack(images)
        logger.debug(f"Preprocessed batch shape: {batch.shape}")
        
        return batch
    
    def inference(self, data, *args, **kwargs):
        data = data.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(data)
        
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        return logits
    
    def postprocess(self, inference_output):
        probabilities = torch.softmax(inference_output, dim=-1)
        predicted_indices = inference_output.argmax(dim=-1).tolist()
        max_probs = probabilities.max(dim=-1).values.tolist()
        
        results = []
        for idx, (pred_idx, prob) in enumerate(zip(predicted_indices, max_probs)):
            label = self.ID2LABEL.get(pred_idx, f"Unknown_{pred_idx}")
            results.append({
                "label": label,
                "class_id": pred_idx,
                "confidence": round(prob, 4)
            })
            logger.debug(f"Prediction {idx}: {label} (confidence: {prob:.4f})")
        
        return results
