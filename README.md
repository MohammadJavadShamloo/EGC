# Efficient Gradient Checkpointing for Memory-Constrained Deep Learning

This repository contains a Jupyter notebook (Efficient Gradient Checkpointing.ipynb) that demonstrates gradient checkpointing in PyTorch to reduce GPU memory usage during training of large models. It includes a from-scratch implementation to understand the mechanics, followed by using PyTorch's built-in checkpointing. The notebook uses a memory-intensive MLP model trained on synthetic data to illustrate memory savings and trade-offs in computation time.

The notebook is designed for educational purposes, helping users grasp how to train larger models on limited hardware by trading compute for memory efficiency.

## Table of Contents

*   Project Overview
*   Dataset
*   Features
*   Requirements
*   Installation
*   Usage
*   Results and Analysis
*   Contributing
*   License

## Project Overview

*   Explain gradient checkpointing: Divide model into segments, save activations at checkpoints, recompute intermediates during backprop.
*   Implement manual checkpointing: Custom forward/backward logic to manage memory.
*   Train without checkpointing: Baseline for comparison.
*   Train with manual checkpointing: Show memory reduction.
*   Use PyTorch's torch.utils.checkpoint: Simplified integration.
*   Compare metrics: Memory usage, training time, and performance.
*   Evaluate model: Visualize predictions vs. real function on noisy data.

The notebook emphasizes practical implementation and interpretation, including second-order optimization concepts like Newton's method.

## Dataset

*   Synthetic linear regression data: Noisy samples from y = 2x + 3.
*   Generated on-the-fly: 100,000 training samples, normalized to \[-1, 1\].
*   Loaded via TensorDataset and DataLoader for batched training.

## Features

*   Memory-intensive MLP model to simulate large networks.
*   Manual checkpointing: Segment-based activation storage and recomputation.
*   PyTorch built-in checkpointing for seamless use.
*   GPU memory tracking and comparison.
*   Training loops with progress (tqdm) and loss monitoring.
*   Visualization: Scatter plots of predictions vs. ground truth.
*   Discussion: Trade-offs (memory vs. time), second-order direction interpretation.

## Requirements

*   Python 3.9+
*   PyTorch
*   Torchvision (optional for transforms)
*   Matplotlib
*   NumPy
*   Tqdm (for progress bars)

See the notebook's import section for the full list:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import numpy as np
```

## Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/MohammadJavadShamloo/EGC.git
    cd EGC
    ```
    
2.  Install dependencies:
    ```sh
    pip install torch torchvision matplotlib numpy tqdm
    ```
    
3.  (Optional) Use a virtual environment:
    ```sh
    python -m venv env
    source env/bin/activate  # On Linux/Mac
    .\env\Scripts\activate   # On Windows
    pip install -r requirements.txt  # Create this file with the above libraries
    ```
    

## Usage

1.  Open the Jupyter notebook:
    ```sh
    jupyter notebook Efficient\ Gradient\ Checkpointing.ipynb
    ```
    
2.  Run the cells sequentially:
    *   Setup device and memory tracking.
    *   Generate synthetic data.
    *   Define and train model without checkpointing.
    *   Implement and train with manual checkpointing.
    *   Use PyTorch checkpointing and compare.
    *   Evaluate and visualize results.

Note: Requires a GPU (CUDA) for meaningful memory comparisons; the notebook checks for availability and uses it automatically. Adjust model size or batch size to test memory limits.

## Results and Analysis

*   **Memory Savings**: Manual and built-in checkpointing reduce peak GPU memory by ~50-70% compared to baseline.
*   **Time Trade-off**: Checkpointing increases training time due to recomputation (e.g., 1.5-2x slower).
*   **Performance**: Model converges similarly across methods; visualizations show accurate fitting to the linear function despite noise.
*   **Insights**: Discusses segment selection, recomputation overhead, and links to optimization concepts like curvature-based step sizes.
*   Refer to the notebook for detailed metrics, plots, and code explanations.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or additional features like advanced models or datasets.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
