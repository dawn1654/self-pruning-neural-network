# Self-Pruning Neural Network

This project implements a neural network that learns to prune its own weights during training using learnable gates and L1 sparsity regularization.

## Overview
Each weight is associated with a gate:
gate = sigmoid(gate_score)

The effective weight becomes:
pruned_weight = weight × gate

During training, an L1 penalty pushes many gates toward zero, effectively removing unnecessary connections.

## Results

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
|------------|-------------|--------------|
| 0.05       | 49.72       | 7.80         |
| 0.20       | 50.11       | 46.60        |
| 1.00       | 48.75       | 92.88        |

This demonstrates a clear trade-off between model performance and sparsity.

## Outputs

- Gate distribution plots (`gate_dist_lambda_*.png`)
- Accuracy vs sparsity trade-off (`tradeoff.png`)

## How to Run

pip install torch torchvision matplotlib  
python main.py

## Key Idea

L1 regularization provides a constant gradient that pushes gate values toward zero, enabling true sparsity and dynamic pruning during training.
