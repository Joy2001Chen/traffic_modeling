# Traffic Occupancy Modeling with DINOv2 + TimeSformer

This project provides a complete PyTorch pipeline for traffic occupancy modeling using occupancy heatmap sequences. The architecture combines DINOv2 for visual feature extraction and a lightweight TimeSformer for temporal modeling.

## Architecture Overview

```
Input: (batch_size, T, 3, 224, 224) occupancy heatmap sequence
    ↓
DINOv2 Feature Extractor: Extract visual features from each frame
    ↓ (batch_size, T, 384)
TimeSformer: Temporal modeling with self-attention
    ↓ (batch_size, T, 384)
Prediction Head: Final prediction (regression/classification)
    ↓ (batch_size, num_outputs)
```

## Features

- **DINOv2 Integration**: Uses `facebook/dinov2-small` for robust visual feature extraction
- **Lightweight TimeSformer**: Custom implementation optimized for traffic sequences
- **Multiple Prediction Heads**: Support for MLP, Conv1D, and Attention-based heads
- **Multi-Task Support**: Regression (future occupancy), Classification (traffic state), and Anomaly Detection (temporal)
- **Flexible Data Format**: Support for PeMS data and simulated data
- **Configurable Heatmap Encoding**: Single-channel (occupancy) or Multi-channel (R:occ, G:flow, B:speed)
- **GPU Training**: Full CUDA support with automatic device detection
- **Comprehensive Training**: Includes validation, testing, checkpointing, and logging

## Installation

1. Clone or download the project files
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Training (Regression Task)

```bash
python train.py \
    --data_dir ./data \
    --task_type regression \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### 2. Classification Task

```bash
python train.py \
    --data_dir ./data \
    --task_type classification \
    --num_outputs 3 \
    --num_epochs 50 \
    --batch_size 8
```

### 3. Anomaly Detection Task (Sequence-level, Binary Classification)

```bash
python train.py \
    --data_dir ./data \
    --data_format pems \
    --task_type anomaly \
    --num_outputs 2 \
    --num_epochs 50 \
    --batch_size 8
```

### 3. Advanced Configuration

```bash
python train.py \
    --data_dir ./data \
    --task_type regression \
    --sequence_length 8 \
    --timesformer_layers 6 \
    --timesformer_heads 12 \
    --head_type conv1d \
    --freeze_backbone \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --weight_decay 1e-4
```

## File Structure

```
traffic_modeling_project/
├── dinov2_feature_extractor.py    # DINOv2 feature extraction
├── timesformer_wrapper.py         # Lightweight TimeSformer implementation
├── traffic_dataset.py             # Dataset and data loading
├── prediction_head.py             # Prediction heads (MLP, Conv1D, Attention)
├── train.py                       # Complete training pipeline
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Model Components

### 1. DINOv2 Feature Extractor (`dinov2_feature_extractor.py`)

- **DINOv2FeatureExtractor**: Single image feature extraction
- **DINOv2SequenceProcessor**: Batch processing of image sequences
- Supports freezing backbone for transfer learning
- Automatic device management

### 2. TimeSformer (`timesformer_wrapper.py`)

- **LightweightTimeSformer**: Custom transformer for temporal modeling
- **MultiHeadSelfAttention**: Efficient attention mechanism
- **TemporalBlock**: Transformer block with residual connections
- **TemporalConv1D**: Alternative temporal modeling with convolutions

### 3. Prediction Heads (`prediction_head.py`)

- **RegressionHead**: MLP for occupancy prediction
- **ClassificationHead**: MLP for traffic state classification
- **Conv1DPredictionHead**: Conv1D-based temporal prediction
- **MultiTaskHead**: Joint regression and classification
- **AttentionPoolingHead**: Attention-based sequence pooling

### 4. Dataset (`traffic_dataset.py`)

- **TrafficHeatmapDataset**: PyTorch dataset for heatmap sequences
- **TrafficDataModule**: Complete data management
- Support for PeMS and simulated data formats
- Automatic data augmentation and preprocessing

## Training Configuration

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data` | Data directory |
| `--task_type` | `regression` | Task type (regression/classification/anomaly) |
| `--batch_size` | `8` | Batch size |
| `--sequence_length` | `8` | Number of frames per sequence |
| `--image_size` | `224` | Image size for preprocessing |
| `--num_outputs` | `2` | Number of outputs |
| `--timesformer_layers` | `4` | Number of TimeSformer layers |
| `--timesformer_heads` | `8` | Number of attention heads |
| `--head_type` | `mlp` | Prediction head type |
| `--num_epochs` | `50` | Number of training epochs |
| `--learning_rate` | `1e-4` | Learning rate |
| `--weight_decay` | `1e-5` | Weight decay |
| `--freeze_backbone` | `False` | Freeze DINOv2 backbone |

### Model Architecture Options

1. **Feature Extractor**: DINOv2-small (384-dim features)
2. **Temporal Model**: Lightweight TimeSformer with configurable layers/heads
3. **Prediction Head**: MLP, Conv1D, or Attention-based
4. **Task Support**: Regression (future occupancy), Classification (traffic state), Anomaly Detection (temporal)

## Data Format

### Simulated Data Structure
```
data/
└── simulated/
    ├── train/
    │   └── sequence_0000/
    │       ├── frame_000.png
    │       ├── frame_001.png
    │       └── ...
    ├── val/
    └── test/
```

### PeMS Data Structure
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

## PeMS Data Download

### Automatic Download
The system can automatically download PeMS data when needed:

1. **Set up credentials**:
   ```bash
   export PEMS_USERNAME='your_username'
   export PEMS_PASSWORD='your_password'
   ```

2. **Install additional dependencies**:
   ```bash
   pip install selenium pandas matplotlib webdriver-manager
   ```

3. **Run training with PeMS data** (defaults to single-channel occupancy heatmaps):
   ```bash
   python train.py --data_format pems
   ```

### Manual Download
You can also download PeMS data manually:

```bash
# Download specific VDS data (single-channel by default)
python pems_downloader.py \
    --username your_username \
    --password your_password \
    --vds_ids 400001 400002 400003 \
    --start_date 2024-01-01 \
    --end_date 2024-01-07 \
    --data_type occupancy

# Multi-channel heatmaps (R=occupancy, G=flow, B=speed) example
# (if CLI flags are available in your version):
# python pems_downloader.py --encode_mode multi --no_noise --flow_max 5000 --speed_max 120

# Or use the example script
python download_pems_example.py
```

### PeMS Downloader Features
- **Automatic authentication** with PeMS website
- **Batch download** of multiple VDS stations
- **Data conversion** to occupancy heatmap sequences
- **Configurable parameters** (VDS IDs, date ranges, data types)
- **Error handling** with fallback to simulated data

## Model Saving and Loading

### Saving Pre-trained Models

The DINOv2 model is automatically downloaded from HuggingFace. To save a local copy:

```python
from dinov2_feature_extractor import DINOv2FeatureExtractor

extractor = DINOv2FeatureExtractor()
extractor.save_pretrained("./local_dinov2")
```

### Loading Checkpoints

```bash
# Resume training from checkpoint
python train.py --resume ./checkpoints/best_model.pt

# Load specific checkpoint
python train.py --resume ./checkpoints/checkpoint_epoch_20.pt
```

## Performance Tips

1. **GPU Memory**: Use smaller batch sizes if encountering OOM errors
2. **Data Loading**: Increase `num_workers` for faster data loading
3. **Backbone Freezing**: Use `--freeze_backbone` for faster training
4. **Mixed Precision**: Consider using `torch.cuda.amp` for larger models

## Example Usage

### Custom Model Creation

```python
from dinov2_feature_extractor import DINOv2SequenceProcessor
from timesformer_wrapper import LightweightTimeSformer
from prediction_head import RegressionHead

# Create custom model
feature_extractor = DINOv2SequenceProcessor(freeze_backbone=True)
timesformer = LightweightTimeSformer(feature_dim=384, num_layers=6)
prediction_head = RegressionHead(feature_dim=384, num_predictions=2)

# Forward pass
sequences = torch.randn(4, 8, 3, 224, 224)  # batch_size=4, seq_len=8
features = feature_extractor(sequences)     # (4, 8, 384)
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

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **Slow Data Loading**: Increase `num_workers` or use SSD storage
3. **DINOv2 Download Issues**: Check internet connection and HuggingFace access
4. **Import Errors**: Ensure all dependencies are installed correctly
5. **NumPy ABI Warning / Errors**: If you see messages like "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.x", downgrade NumPy:
   ```bash
   pip install 'numpy<2.0' --upgrade
   ```

### Performance Monitoring

The training script provides comprehensive logging:
- Training/validation loss and metrics
- Learning rate scheduling
- Best model checkpointing
- TensorBoard integration (optional)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{traffic_modeling_dinov2_timesformer,
  title={Traffic Occupancy Modeling with DINOv2 + TimeSformer},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/traffic-modeling}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
