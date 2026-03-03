import numpy as np
import torch
from sklearn import metrics

def mse(y_true, y_pred):
    if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    return metrics.mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    return metrics.mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    
    # Avoid div-by-zero
    mask = y_true != 0
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred)
    
    # Avoid zeroes in denominator
    mask = denominator != 0
    return 2.0 * np.mean(numerator[mask] / denominator[mask]) * 100

def ade(y_true, y_pred):
    """Average Displacement Error (L2 norm distance over all points)"""
    if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    
    # Assuming shape [batch, seq_len, 2] where last dim is [Lon, Lat]
    diff = y_pred - y_true
    distances = np.linalg.norm(diff, axis=-1)  # shape: [batch, seq_len]
    return np.mean(distances)

def fde(y_true, y_pred):
    """Final Displacement Error (L2 norm distance at the last predicted point)"""
    if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    
    diff = y_pred[:, -1, :] - y_true[:, -1, :]
    distances = np.linalg.norm(diff, axis=-1)  # shape: [batch]
    return np.mean(distances)

def get_all_metrics(y_true, y_pred):
    """Compute and return all regression metrics as a dictionary"""
    # ensure flat structure for straightforward metric collection across large batches
    res = {
        "ade": float(ade(y_true, y_pred)),
        "fde": float(fde(y_true, y_pred))
    }
    
    if y_true.ndim > 2:
        y_true_flat = y_true.reshape(-1, y_true.shape[-1])
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    else:
        y_true_flat = y_true
        y_pred_flat = y_pred
        
    res.update({
        "mse": float(mse(y_true_flat, y_pred_flat)),
        "rmse": float(rmse(y_true_flat, y_pred_flat)),
        "mae": float(mae(y_true_flat, y_pred_flat)),
        "mape": float(mape(y_true_flat, y_pred_flat)),
        "smape": float(smape(y_true_flat, y_pred_flat))
    })
    return res
