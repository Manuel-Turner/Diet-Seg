# Hardness-based Image Segmentation Model User Guide

## Project Overview

This project implements a medical image segmentation model based on RWKV-UNet, with special focus on hardness-based segmentation methods and edge detection network (EdgeNet). The model is primarily used for brain tumor segmentation tasks, adjusting the training process by calculating the segmentation difficulty (hardness) of samples to improve segmentation performance on difficult samples.

## Key Features

1. **Hardness-based Segmentation**: Adjusts the training process specifically by calculating the segmentation difficulty of each sample, improving segmentation performance on difficult samples
2. **RWKV-UNet Architecture**: Combines the RWKV (Receptance Weighted Key Value) attention mechanism with UNet architecture, enhancing the model's expressive capability
3. **Edge Detection Enhancement**: Uses a specialized EdgeNet to improve the accuracy of segmentation boundaries, particularly suitable for fine structure segmentation in medical images

## File Structure

```
git/
├── models/
│   ├── rwkv_unet.py          # Basic RWKV-UNet model implementation
│   ├── rwkv_unet_edge.py     # RWKV-UNet model with edge detection
│   ├── edge_detection.py     # Edge detection network implementation
│   └── ccm/
│       └── ccm.py            # Channel mixing module
├── datasets/
│   └── dataset.py            # Dataset processing related code
├── utils/
│   └── calc_hardness.py      # Tool for calculating segmentation difficulty
└── README.md                 # Project documentation
```

## Usage Instructions

### 1. Calculate Sample Hardness

First, you need to calculate the segmentation difficulty of each sample in the dataset, which will be used for sample weighting during training:

```python
from utils.calc_hardness import HardnessCalculator

# Initialize hardness calculator
calculator = HardnessCalculator(
    pred_folder='path/to/predictions',  # Prediction results folder
    gt_folder='path/to/ground_truth'   # Ground truth labels folder
)

# Calculate and save hardness dictionary
hardness_dict = calculator.calculate_hardness('./hardness_dict.pkl')

# Visualize hardness distribution
calculator.visualize_hardness_distribution(hardness_dict)
```

### 2. Dataset Preparation

Use the `BratsDataset` class in `datasets/dataset.py` to load data:

```python
from datasets.dataset import BratsConfig, get_brats_dataset
from torch.utils.data import DataLoader

# Configure dataset parameters
config = BratsConfig(
    train_folder='path/to/brats_data',
    target_size=(128, 128, 128),
    is_gz=False  # Set according to data format
)

# Create dataset and data loader
dataset = get_brats_dataset(config)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
```

### 3. Model Training

Train using the RWKV-UNet model with edge detection:

```python
from models.rwkv_unet_edge import RWKV_UNet
import torch
from torch import optim
from torch.nn import functional as F

# Initialize model
model = RWKV_UNet(
    in_chans=4,           # Input channels (4 modalities)
    out_chans=3,          # Output channels (3 segmentation classes)
    img_size=128          # Image size
)
model.cuda()

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(800):
    for batch in loader:
        images = batch['image'].cuda()        # [B, 4, 128, 128, 128]
        labels = batch['label'].cuda()        # [B, 128, 128, 128]
        hardness = batch['hardness'].cuda()   # [B]
        
        # Forward propagation
        outputs = model(images.view(-1, 4*128, 128, 128))
        outputs = outputs.view(-1, 3, 128, 128, 128)
        
        # Calculate loss (weighted by hardness)
        loss_ce = F.cross_entropy(outputs, labels, reduction='none')
        loss_ce = (loss_ce * (1 + hardness.view(-1, 1, 1, 1))).mean()
        
        # Backpropagation
        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()
        
    # Save model
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f'./checkpoints/epoch_{epoch}.pth')
```

### 4. Model Inference

Use the trained model for inference:

```python
import nibabel as nib
import numpy as np
from models.rwkv_unet_edge import RWKV_UNet

# Load model
model = RWKV_UNet(in_chans=4, out_chans=3, img_size=128)
model.load_state_dict(torch.load('path/to/model.pth'))
model.cuda()
model.eval()

# Load test data
test_data = {...}  # Load test data

# Inference
with torch.no_grad():
    outputs = model(test_data.cuda())
    predictions = torch.argmax(outputs, dim=1)

# Save results
predictions = predictions.cpu().numpy()
# Save prediction results as NIfTI file
```

## Notes

1. Model training requires a large GPU memory, it is recommended to use a GPU with at least 16GB of VRAM
2. Data preprocessing steps have a significant impact on model performance, please ensure correct normalization and data augmentation
3. Hardness calculation requires prediction results from a baseline model first, you can use a simple UNet model to generate initial predictions

## References

- Papers related to RWKV attention mechanism
- Edge detection techniques in medical image segmentation
- Research on training strategies based on sample difficulty