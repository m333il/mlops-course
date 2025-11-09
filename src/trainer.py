import os
import logging
import torch
from tqdm import tqdm, trange

from sklearn.metrics import accuracy_score, classification_report

from src.dataset import get_dataloaders
from src.models import get_model, get_image_processor
from src.models import get_optimizer


class Trainer:
    def __init__(self, config):
        logging.info(f'Trainer.__init__: start')
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.num_epochs = self.config['training']['num_train_epochs']
        self.output_dir = self.config['training']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        logging.info(f'Cuda is available: {torch.cuda.is_available()}')
        logging.info(f'Device: {self.device}')

        self.train_dl, self.val_dl, self.processor, self.id2label, self.label2id = get_dataloaders(self.config)
        self.model = get_model(self.config['model']['name'], self.id2label, self.label2id).to(self.device)
        self.optimizer = get_optimizer(self.model, self.config['training'])
        logging.info(f'Trainer.__init__: finish')


    def _one_epoch(self, epoch_num):
        logging.info(f'Trainer._one_epoch: start')
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_dl, desc=f"Training: epoch {epoch_num}", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            loss = self.model(**batch).loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        total_loss /= len(self.train_dl)
        logging.info(f'Trainer._one_epoch: finish')
        return total_loss


    @torch.no_grad()
    def _predict(self):
        logging.info(f'Trainer._predict: start')
        self.model.eval()
        y_pred = []
        y_true = []
        
        for batch in tqdm(self.val_dl, desc="Predicting"):
            labels = batch.pop("labels")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            y_pred.append(outputs.logits.cpu().argmax(dim=-1))
            y_true.append(labels.cpu())
        
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        logging.info(f'Trainer._predict: finish')
        return y_true.numpy(), y_pred.numpy(),


    def evaluate(self):
        logging.info(f'Trainer.evaluate: start')
        y_true, y_pred = self._predict()
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, 
            target_names=self.model.config.id2label.values(), 
            zero_division=0
        )
        logging.info(f'Trainer.evaluate: finish')
        return accuracy, report


    def train(self, save_model=True):
        logging.info(f'Trainer.train: start')

        for epoch in trange(self.num_epochs, desc='Training', leave=False):
            logging.info(f"Epoch {epoch}/{self.num_epochs}")
            avg_train_loss = self._one_epoch(epoch)
            logging.info(f"Train loss: {avg_train_loss:.4f}")
            accuracy, report = self.evaluate()
            logging.info(f"Validation:")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Classification Report:\n{report}")

        if save_model:
            self.save_model()
        logging.info(f'Trainer.train: finish')


    def save_model(self):
        logging.info(f'Trainer.save_model: start')
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        logging.info(f'Trainer.save_model: finish')