import os
import logging
import torch
from tqdm import tqdm, trange

from sklearn.metrics import accuracy_score, classification_report

from src.dataset import get_dataloaders
from src.models import get_model
from src.models import get_optimizer


class Trainer:
    def __init__(
        self, model, optimizer, 
        train_loader, val_loader, 
        device, num_epochs,
        output_dir, processor,
        id2label, label2id
    ):
        logging.info(f'Trainer.__init__: start')
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.train_dl = train_loader
        self.val_dl = val_loader
        self.processor = processor
        self.id2label = id2label
        self.label2id = label2id
        self.model = model
        self.optimizer = optimizer
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        logging.info(f'Cuda is available: {torch.cuda.is_available()}')
        logging.info(f'Device: {self.device}')
        logging.info(f'Trainer.__init__: finish')


    def _prepare_batch(self, batch, device):
        return {k: v.to(device) for k, v in batch.items()}
    
    
    def _training_step(self, batch):
        self.optimizer.zero_grad()
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    
    def _compute_epoch_loss(self, losses):
        if len(losses) == 0:
            return 0.0
        return sum(losses) / len(losses)
    
    
    def _one_epoch(self, epoch_num):
        logging.info(f'Trainer._one_epoch: start')
        self.model.train()
        losses = []
        
        for batch in tqdm(self.train_dl, desc=f"Training: epoch {epoch_num}", leave=False):
            prepared_batch = self._prepare_batch(batch, self.device)
            loss = self._training_step(prepared_batch)
            losses.append(loss)
        
        avg_loss = self._compute_epoch_loss(losses)
        logging.info(f'Trainer._one_epoch: finish')
        return avg_loss


    def _prepare_batch_for_inference(self, batch, device):
        labels = batch.pop("labels")
        prepared_batch = {k: v.to(device) for k, v in batch.items()}
        return prepared_batch, labels
    
    
    def _inference_step(self, batch):
        outputs = self.model(**batch)
        predictions = outputs.logits.cpu().argmax(dim=-1)
        return predictions
    
    
    def _aggregate_predictions(self, all_predictions, all_labels):
        y_pred = torch.cat(all_predictions)
        y_true = torch.cat(all_labels)
        return y_true.numpy(), y_pred.numpy()
    
    
    @torch.no_grad()
    def _predict(self):
        logging.info(f'Trainer._predict: start')
        self.model.eval()
        y_pred = []
        y_true = []
        
        for batch in tqdm(self.val_dl, desc="Predicting"):
            prepared_batch, labels = self._prepare_batch_for_inference(batch, self.device)
            predictions = self._inference_step(prepared_batch)
            y_pred.append(predictions)
            y_true.append(labels.cpu())
        
        true_labels, pred_labels = self._aggregate_predictions(y_pred, y_true)
        logging.info(f'Trainer._predict: finish')
        return true_labels, pred_labels


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


def create_trainer_from_config(config):
    logging.info('create_trainer_from_config: start')
    
    device = torch.device(config['training']['device'])
    num_epochs = config['training']['num_train_epochs']
    output_dir = config['training']['output_dir']
    
    train_loader, val_loader, processor, id2label, label2id = get_dataloaders(config)
    
    model = get_model(config['model']['name'], id2label, label2id).to(device)
    optimizer = get_optimizer(model, config['training'])
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        output_dir=output_dir,
        processor=processor,
        id2label=id2label,
        label2id=label2id
    )
    
    logging.info('create_trainer_from_config: finish')
    return trainer