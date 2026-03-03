import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from ..utils.metrics import get_all_metrics

class Trainer:
    def __init__(self, model, config, mlflow_logger, device):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.logger = mlflow_logger
        
        self.learning_rate = config['training']['learning_rate']
        self.epochs = config['training']['epochs']
        self.patience = config['training']['patience']
        
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss() # Standard trajectory objective
        
    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0
        
        for epoch in range(self.epochs):
            # --- Training phase ---
            self.model.train()
            train_loss_total = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            batches = 0
            for batch_idx, (x, y) in enumerate(pbar):
                batches += 1
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # BiLSTM returns [batch, seq_len, output_size]. We only want specifically to predict 
                # the target trajectory points. To match logic: let's assume the model is expected
                # to predict the sequential offset OR the entire target sequence.
                # Here we predict utilizing the entire sliding x sequence:
                
                outputs = self.model(x)
                
                # In standard sequence-to-sequence if you want to predict 'pre_len' steps ahead:
                # typically the model architecture either pools or we slice the outputs.
                # Assuming y is of shape [batch, pre_len, target_features], we might need an adapter.
                # For baseline we will compute MSE across the last 'pre_len' outputs or adapt as necessary.
                # We will just mimic standard seq2seq objective: 
                # If outputs has length 'obs_len', but we need 'pre_len' we should slice/adapt.
                
                # Assuming simple architecture where model(x) predicts next step per x step, 
                # or we just compare the last timestep's output against y.
                # Let's align with the shape of `y` (predict_step)
                # Ensure outputs matches y's sequence length:
                
                # Simplest adapter for BiLSTM prediction:
                if outputs.shape[1] != y.shape[1]:
                    # We just take the last timestep outputs and linear map it to predict_step
                    # Or for now, let's just use the last 'y.shape[1]' steps of output if possible
                    # We'll slice outputs to match y length assuming it's a direct seq map
                    outputs_for_loss = outputs[:, -y.shape[1]:, :]
                else:
                    outputs_for_loss = outputs
                    
                loss = self.criterion(outputs_for_loss, y)
                loss.backward()
                
                # Prevent Exploding Gradients (Common in LSTMs causing NaNs)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss_total += loss.item()
                global_step += 1
                
                pbar.set_postfix({"loss": loss.item()})
                
                if global_step % self.config['logging']['log_frequency'] == 0:
                    self.logger.log_metrics({"train_step_loss": loss.item()}, step=global_step)
            
            avg_train_loss = train_loss_total / batches if batches > 0 else 0
            
            # --- Validation phase ---
            val_loss, val_metrics = self.evaluate(val_loader)
            
            # Log everything cleanly
            pbar_epoch = tqdm(total=0, bar_format='{desc}') 
            pbar_epoch.set_description(f"Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ADE: {val_metrics['ade']:.4f}")
            
            epoch_metrics = {
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }
            self.logger.log_metrics(epoch_metrics, step=epoch)
            
            # --- Early Stopping & Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.logger.log_model(self.model, "best_model")
                torch.save(self.model.state_dict(), "best_model.pth")
                print(" >>> Saved new best model")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                    break
                    
    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_trues = []
        batches = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                batches += 1
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)
                if outputs.shape[1] != y.shape[1]:
                    outputs_for_loss = outputs[:, -y.shape[1]:, :]
                else:
                    outputs_for_loss = outputs
                    
                loss = self.criterion(outputs_for_loss, y)
                val_loss += loss.item()
                
                all_preds.append(outputs_for_loss.cpu().numpy())
                all_trues.append(y.cpu().numpy())
                
        avg_val_loss = val_loss / batches if batches > 0 else 0
        
        if len(all_preds) > 0:
            preds_concat = np.concatenate(all_preds, axis=0)
            trues_concat = np.concatenate(all_trues, axis=0)
            metrics = get_all_metrics(trues_concat, preds_concat)
        else:
            metrics = {"ade": 0.0, "fde": 0.0, "mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "smape": 0.0}
        
        return avg_val_loss, metrics
