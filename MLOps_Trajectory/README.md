# Maritime Vessel Trajectory Forecasting Pipeline 🚢

This repository contains an end-to-end, high-performance Deep Learning MLOps pipeline designed to analyze and predict Maritime Vessel Trajectories utilizing immense AIS tracking datasets.

## 🚀 Key Features

* **Advanced Data Loading** 📦: Custom PyArrow `DaskAISTrajectoryDataset` iterable gracefully handles multi-gigabyte datasets (e.g., streaming over **17.5 Million** coordinate points seamlessly) without encountering memory out-of-bounds errors.
* **GeoJSON Normalization** 🗺️: Bounding-box constraint validation natively overrides `MinMaxScaler` using `.geojson` geometry polygons to firmly project geographical metrics into a reliable 0.0 to 1.0 boundary box (e.g. Region 1 Coordinates).
* **NVIDIA Hardware Optimization** 💻: Strictly binds computation pipelines to dedicated NVIDIA GPUs (e.g., GTX 1650 / RTX) to maximize CUDA processing cores, with automatic batch scaling configured to fit VRAM limitations precisely. Includes built-in Exploding Gradient prevention.
* **Rich Model Tracking** 📈: Includes real-time `MLFlow` integration measuring custom geometric losses: **Average Displacement Error (ADE)** and **Final Displacement Error (FDE)** alongside classic sequences like MSE and RMSE. Local MLflow web GUI connects straight into background training instances.

## 📁 Repository Structure

```
MLOps_Trajectory/
├── configs/
│   └── default.yaml         # Configuration file: epochs, batch_sizes, datasets
├── scripts/
│   ├── train.py             # Main pipeline wrapper streaming rows and models
│   └── ... 
├── src/
│   ├── data/
│   │   ├── dataset_lazy.py  # Custom PyArrow Iterators  
│   │   └── preprocessing.py # Scaling constraints via GeoJSON boundaries  
│   ├── models/
│   │   └── bilstm.py        # Scalable Bidirectional LSTM structure
│   ├── training/
│   │   ├── mlflow_logger.py # Metric server mapping
│   │   └── trainer.py       # Core Recurrent Loop & gradient limits
│   └── utils/
│       ├── metrics.py       # Calculations: ADE, FDE, SMAPE
│       └── helpers.py       # Configuration parsers
```

## 🛠️ Instructions

### 1. Requirements
Ensure you have PyTorch correctly mapped to your local NVIDIA GPU (e.g. CUDA 11/12/12.7). A typical environment requires:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mlflow pyarrow pandas numpy duckdb tqdm
```

### 2. Configuration (`configs/default.yaml`)
You can freely customize training limits across the YAML format. Key pointers:
* To prevent `CUDA Out of Memory`, adjust `batch_size`. (e.g., `64` for 4GB GPUs or `256` for 16GB GPUs).
* Define your dataset target internally via `dataset_path`.
* Ensure `use_cuda` is exactly `true`. Default `train.py` binds perfectly to `cuda:0` upon launch.

### 3. Launching Training
Drop your respective Dataset file onto your storage drive, then execute:
```bash
cd MLOps_Trajectory
python scripts/train.py
```

### 4. Viewing Results in Real-Time
You can view the dynamic charts of your iterations actively in your browser:
```bash
mlflow ui --port 5000
```
Then navigate to: `http://localhost:5000`
