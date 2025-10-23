# Installation and Setup Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- At least 8GB RAM
- 10GB free disk space

## Installation Steps

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv traffic_env

# Activate virtual environment
source traffic_env/bin/activate  # On macOS/Linux
# or
traffic_env\Scripts\activate   # On Windows
```

### 2. Install Dependencies

```bash
# Install PyTorch (choose appropriate version for your system)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test the pipeline
python test_pipeline.py
```

## Quick Start Examples

### Example 1: Basic Regression Training

```bash
python train.py \
    --data_dir ./data \
    --task_type regression \
    --num_epochs 10 \
    --batch_size 4 \
    --sequence_length 8
```

### Example 2: Classification Training

```bash
python train.py \
    --data_dir ./data \
    --task_type classification \
    --num_outputs 3 \
    --num_epochs 10 \
    --batch_size 4
```

### Example 3: Advanced Configuration

```bash
python train.py \
    --data_dir ./data \
    --task_type regression \
    --timesformer_layers 6 \
    --timesformer_heads 12 \
    --head_type conv1d \
    --freeze_backbone \
    --num_epochs 20 \
    --batch_size 8 \
    --learning_rate 5e-5
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 2`
   - Reduce sequence length: `--sequence_length 4`
   - Use CPU: `--device cpu`

2. **Module Import Errors**
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

3. **DINOv2 Download Issues**
   - Check internet connection
   - Verify HuggingFace access
   - Try downloading manually: `python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-small')"`

4. **Slow Training**
   - Increase number of workers: `--num_workers 8`
   - Use SSD storage for data
   - Enable mixed precision training (modify train.py)

### Performance Tips

- **GPU Memory**: Start with small batch sizes and increase gradually
- **Data Loading**: Use multiple workers for faster data loading
- **Model Size**: Use `--freeze_backbone` for faster training
- **Sequence Length**: Longer sequences need more memory

## Data Preparation

### Simulated Data (Automatic)
The system automatically generates simulated data for testing. No manual preparation needed.

### PeMS Data Format
If using real PeMS data, organize it as follows:

```
data/
└── pems/
    ├── train/
    │   └── sequence_0000/
    │       ├── frame_000.png
    │       ├── frame_001.png
    │       └── ...
    ├── val/
    └── test/
```

Each PNG file should be 224x224 pixels with RGB channels.

## Model Checkpoints

### Saving Models
Models are automatically saved in the `--save_dir` directory:
- `best_model.pt`: Best performing model
- `checkpoint_epoch_X.pt`: Regular checkpoints

### Loading Models
```bash
# Resume training from best model
python train.py --resume ./checkpoints/best_model.pt

# Resume from specific epoch
python train.py --resume ./checkpoints/checkpoint_epoch_10.pt
```

## Custom Usage

### Using Individual Components

```python
import torch
from dinov2_feature_extractor import DINOv2SequenceProcessor
from timesformer_wrapper import LightweightTimeSformer
from prediction_head import RegressionHead

# Create components
feature_extractor = DINOv2SequenceProcessor()
timesformer = LightweightTimeSformer(feature_dim=384, sequence_length=8)
prediction_head = RegressionHead(feature_dim=384, num_predictions=2)

# Process data
sequences = torch.randn(4, 8, 3, 224, 224)  # batch_size=4, seq_len=8
features = feature_extractor(sequences)      # (4, 8, 384)
temporal = timesformer(features)             # (4, 8, 384)
predictions = prediction_head(temporal)      # (4, 2)
```

### Custom Dataset

```python
from traffic_dataset import TrafficHeatmapDataset

# Create custom dataset
dataset = TrafficHeatmapDataset(
    data_dir="./my_data",
    sequence_length=10,
    task_type="classification",
    data_format="pems"
)
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 5GB free disk space
- CPU (slower training)

### Recommended Requirements
- Python 3.9+
- 16GB RAM
- 20GB free disk space
- NVIDIA GPU with 8GB+ VRAM
- SSD storage

### Tested Configurations
- macOS 13.6 (Apple Silicon)
- Ubuntu 20.04 (CUDA 11.8)
- Windows 10 (CUDA 11.8)
