import torch
from torch.utils.data import DataLoader
import argparse
import sys
import os
import pyarrow.parquet as pq
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import set_seed, load_config
from src.data.preprocessing import MinMaxScaler
from src.models.bilstm import BiLSTMModel
from src.training.trainer import Trainer
from src.training.mlflow_logger import MLFlowLogger

class DaskAISTrajectoryDataset(torch.utils.data.IterableDataset):
    """
    Very fast iterable dataset streaming chunks sequentially using PyArrow.
    This avoids loading the 17M rows entirely into memory, instead iterating cleanly.
    """
    def __init__(self, parquet_path, feature_cols, label_cols, chunk_range, obs_len=10, pre_len=2, scaler=None):
        self.parquet_path = parquet_path
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.obs_len = obs_len
        self.pre_len = pre_len
        self.scaler = scaler
        
        self.label_indices = [feature_cols.index(col) for col in label_cols]
        self.seq_len = obs_len + pre_len
        
        # Determine the logical slices for this dataset split
        self.start_chunk = chunk_range[0]
        self.end_chunk = chunk_range[1]

    def __iter__(self):
        parquet_file = pq.ParquetFile(self.parquet_path)
        
        # We iterate over Row Groups (chunks) in PyArrow.
        # This keeps RAM perfectly safe.
        for i in range(self.start_chunk, self.end_chunk):
            table = parquet_file.read_row_group(i, columns=self.feature_cols + ['MMSI'])
            df = table.to_pandas()
            df = df.dropna(subset=self.feature_cols + ['MMSI'])
            
            if len(df) == 0:
                continue
            
            # Fast vectorized feature scaling
            if self.scaler is not None:
                df[self.feature_cols] = self.scaler.transform(df[self.feature_cols].values)
                
            # Groupby MMSI seamlessly prevents massive boolean mask memory fragmentation
            for vessel_id, group in df.groupby('MMSI'):
                v_data = group[self.feature_cols].values
                
                if len(v_data) >= self.seq_len:
                    for j in range(len(v_data) - self.seq_len + 1):
                        x_seq = v_data[j : j + self.obs_len]
                        y_seq = v_data[j + self.obs_len : j + self.seq_len, self.label_indices]
                        
                        yield torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Train Trajectory Prediction Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config['random_seed'])
    
    logger = MLFlowLogger(config)
    logger.start_run()
    logger.log_params(config)

    print(f"Preparing to stream the entire 17M row dataset from {config['data']['dataset_path']}...")
    
    parquet_file = pq.ParquetFile(config['data']['dataset_path'])
    num_row_groups = parquet_file.num_row_groups
    print(f"Total Row Groups found in PyArrow: {num_row_groups}")
    
    # We apportion the PyArrow row groups functionally: 70% Train, 15% Val, 15% Test
    # This prevents temporal bleeding across row chunk blocks explicitly.
    train_split = int(num_row_groups * config['data']['train_split'])
    valid_split = train_split + int(num_row_groups * config['data']['valid_split'])
    
    train_range = (0, train_split)
    valid_range = (train_split, valid_split)
    test_range = (valid_split, num_row_groups)
    
    print(f"Allocating Chunks => Train: {train_range}, Val: {valid_range}, Test: {test_range}")
    
    scaler = MinMaxScaler()
    
    # Globally Fit Scaler on the first few chunks to save memory
    print("Fitting Scaler bounds using first 5 row groups representing global distributions...")
    temp_df = parquet_file.read_row_groups(list(range(min(5, train_split))), columns=config['data']['feature_columns']).to_pandas()
    temp_df = temp_df.dropna()
    scaler.fit(temp_df.values)
    
    # OVERRIDE SPATIAL BOUNDS EXPLICITLY USING region_1.geojson
    print("Constraining Longitude & Latitude explicitly via Region 1 GeoJSON mapping...")
    scaler.set_manual_bounds(config['data']['feature_columns'], 'Longitude', -98.73875969190912, -80.20286516849585)
    scaler.set_manual_bounds(config['data']['feature_columns'], 'Latitude', 17.444407942371157, 31.100306355446108)
    
    del temp_df
    
    train_dataset = DaskAISTrajectoryDataset(
        parquet_path=config['data']['dataset_path'],
        feature_cols=config['data']['feature_columns'],
        label_cols=config['data']['label_columns'],
        obs_len=config['data']['time_step'],
        pre_len=config['data']['predict_step'],
        chunk_range=train_range,
        scaler=scaler
    )
    
    val_dataset = DaskAISTrajectoryDataset(
        parquet_path=config['data']['dataset_path'],
        feature_cols=config['data']['feature_columns'],
        label_cols=config['data']['label_columns'],
        obs_len=config['data']['time_step'],
        pre_len=config['data']['predict_step'],
        chunk_range=valid_range,
        scaler=scaler
    )
    
    test_dataset = DaskAISTrajectoryDataset(
        parquet_path=config['data']['dataset_path'],
        feature_cols=config['data']['feature_columns'],
        label_cols=config['data']['label_columns'],
        obs_len=config['data']['time_step'],
        pre_len=config['data']['predict_step'],
        chunk_range=test_range,
        scaler=scaler
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])
    
    # STRICT NVIDIA GPU ENFORCEMENT
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA is not available! Training halted to prevent CPU usage.")
        sys.exit(1)
        
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} NVIDIA Compute Devices.")
    current_device = torch.cuda.get_device_name(0)
    print(f"Targeting Primary GPU: {current_device}")
    
    device = torch.device("cuda:0")
    
    if config['model']['type'] == 'bilstm':
        model = BiLSTMModel(
            input_size=len(config['data']['feature_columns']),
            hidden_size=config['model']['hidden_size'],
            output_size=len(config['data']['label_columns']),
            num_layers=config['model']['lstm_layers'],
            dropout_rate=config['model']['dropout_rate']
        )
    else:
        raise NotImplementedError(f"Model {config['model']['type']} not explicitly implemented yet.")
        
    resume_path = config['training'].get('resume_checkpoint', '')
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming model weights from {resume_path}...")
        model.load_state_dict(torch.load(resume_path, map_location=device))
        
    trainer = Trainer(model, config, logger, device)
    
    # Because iterator length is undefined, we pass expected batches for TQDM approximation
    # 17M rows / 256 batch size = ~65000 total batches roughly. Train ~45k
    print("Beginning Training! Generating Sequences asynchronously...")
    trainer.train(train_loader, val_loader)
    
    print("\n-------------------------------\nEvaluating Best Model on TEST SET\n-------------------------------")
    try:
        model.load_state_dict(torch.load("best_model.pth"))
        test_loss, test_metrics = trainer.evaluate(test_loader)
        
        logger.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        print(f"Test Set Loss: {test_loss:.4f}")
        for k,v in test_metrics.items():
            print(f" -> {k.upper()}: {v:.4f}")
    except FileNotFoundError:
        print("No best model found. Perhaps training ran for 0 epochs or failed.")
    
    logger.end_run()

if __name__ == "__main__":
    main()
