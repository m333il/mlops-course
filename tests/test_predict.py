import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch
from PIL import Image
import numpy as np
import pandas as pd

from src.predict import (
    get_image_files,
    predict_single_image,
    predict_batch,
    save_predictions,
    SUPPORTED_EXTENSIONS,
)


class TestGetImageFiles:
    def test_finds_supported_images(self, tmp_path):
        for ext in ['.jpg', '.png', '.jpeg']:
            img = Image.fromarray(np.uint8(np.random.rand(10, 10, 3) * 255))
            img.save(tmp_path / f"test{ext}")
        
        (tmp_path / "test.txt").write_text("not an image")
        
        result = get_image_files(str(tmp_path))
        
        assert len(result) == 3
        assert all(f.suffix.lower() in SUPPORTED_EXTENSIONS for f in result)
    
    def test_raises_on_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            get_image_files("/nonexistent/path")
    
    def test_raises_on_file_path(self, tmp_path):
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")
        
        with pytest.raises(ValueError, match="not a directory"):
            get_image_files(str(file_path))
    
    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(ValueError, match="No supported images"):
            get_image_files(str(tmp_path))


class TestPredictSingleImage:
    def test_returns_label_string(self, tmp_path):
        img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        mock_model = MagicMock()
        mock_model.config.id2label = {0: 'Dresses', 1: 'Shorts', 2: 'Tees_Tanks'}
        
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 0.8, 0.1]])
        mock_model.return_value = mock_outputs
        
        mock_processor = MagicMock()
        mock_processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}
        
        device = torch.device('cpu')
        
        result = predict_single_image(img_path, mock_model, mock_processor, device)
        
        assert result == 'Shorts'
        assert isinstance(result, str)


class TestPredictBatch:
    def test_processes_multiple_images(self, tmp_path):
        image_files = []
        for i in range(3):
            img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
            img_path = tmp_path / f"test_{i}.jpg"
            img.save(img_path)
            image_files.append(img_path)
        
        mock_model = MagicMock()
        mock_model.config.id2label = {0: 'Dresses', 1: 'Shorts'}
        
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.8, 0.2]])
        mock_model.return_value = mock_outputs
        
        mock_processor = MagicMock()
        mock_processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}
        
        device = torch.device('cpu')
        
        results = predict_batch(image_files, mock_model, mock_processor, device)
        
        assert len(results) == 3
        assert all('image_name' in r for r in results)
        assert all('predicted_label' in r for r in results)
        assert all(r['predicted_label'] == 'Dresses' for r in results)
    
    def test_handles_corrupted_image(self, tmp_path):
        valid_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        valid_path = tmp_path / "valid.jpg"
        valid_img.save(valid_path)
        
        corrupted_path = tmp_path / "corrupted.jpg"
        corrupted_path.write_bytes(b"not a valid image")
        
        image_files = [valid_path, corrupted_path]
        
        mock_model = MagicMock()
        mock_model.config.id2label = {0: 'Dresses'}
        
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.8]])
        mock_model.return_value = mock_outputs
        
        mock_processor = MagicMock()
        mock_processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}
        
        device = torch.device('cpu')
        
        results = predict_batch(image_files, mock_model, mock_processor, device)
        
        assert len(results) == 2
        assert results[0]['predicted_label'] == 'Dresses'
        assert results[1]['predicted_label'] == 'ERROR'


class TestSavePredictions:
    def test_saves_csv_correctly(self, tmp_path):
        results = [
            {'image_name': 'img1.jpg', 'predicted_label': 'Dresses'},
            {'image_name': 'img2.jpg', 'predicted_label': 'Shorts'},
        ]
        
        output_path = tmp_path / "preds.csv"
        save_predictions(results, str(output_path))
        
        assert output_path.exists()
        
        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert list(df.columns) == ['image_name', 'predicted_label']
        assert df.iloc[0]['image_name'] == 'img1.jpg'
        assert df.iloc[0]['predicted_label'] == 'Dresses'
    
    def test_creates_parent_directories(self, tmp_path):
        results = [{'image_name': 'img.jpg', 'predicted_label': 'Dresses'}]
        
        output_path = tmp_path / "nested" / "dir" / "preds.csv"
        save_predictions(results, str(output_path))
        
        assert output_path.exists()
