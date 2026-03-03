import mlflow
import os

class MLFlowLogger:
    def __init__(self, config):
        """Initialize MLflow tracking server and set experiment parameters."""
        self.tracking_uri = config['logging']['mlflow_tracking_uri']
        mlflow.set_tracking_uri(self.tracking_uri)
        
        self.experiment_name = config['experiment_name']
        mlflow.set_experiment(self.experiment_name)
        
        self.run_name = config['run_name']
        
    def start_run(self):
        """Starts an MLflow run returning the active run object."""
        return mlflow.start_run(run_name=self.run_name)
        
    def log_params(self, params):
        """Log hyperlink dictionary of config params."""
        # Flatten simple params easily
        flat_params = {}
        for section, dict_val in params.items():
            if isinstance(dict_val, dict):
                for k, v in dict_val.items():
                    flat_params[f"{section}.{k}"] = str(v)
            else:
                flat_params[section] = str(dict_val)
                
        mlflow.log_params(flat_params)
        
    def log_metrics(self, metrics, step=None):
        """Log scalar metrics dictionary."""
        mlflow.log_metrics(metrics, step=step)
        
    def log_model(self, model, artifact_path="model"):
        """Save PyTorch model binary."""
        mlflow.pytorch.log_model(model, artifact_path)
        
    def end_run(self):
        mlflow.end_run()
