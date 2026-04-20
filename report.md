## 1. Introduction

In modern deep learning systems, large neural networks often contain redundant parameters, making them inefficient for deployment in resource-constrained environments. Pruning techniques help address this by removing less important weights.

In this project, a **self-pruning neural network** is implemented, where the model learns to prune its own weights during training using learnable gate parameters and sparsity regularization.

---

## 2. Methodology

### 2.1 Prunable Linear Layer

A custom linear layer, `PrunableLinear`, is designed where each weight has an associated learnable parameter called `gate_score`.

- Gate values are computed as:
  

gate = sigmoid(gate_score)


- The effective weight becomes:


pruned_weight = weight × gate


If the gate approaches zero, the corresponding weight is effectively removed from the network.

---

### 2.2 Sparsity Regularization

The total loss is defined as:


Total Loss = Classification Loss + λ × Sparsity Loss


Where:

- Classification Loss: Cross-Entropy Loss  
- Sparsity Loss: L1 norm of all gate values  


SparsityLoss = Σ gate_i


---

### 2.3 Why L1 Encourages Sparsity

The L1 penalty provides a constant gradient that pushes values toward zero regardless of their magnitude.

- L1 Gradient → constant → strong push to zero  
- L2 Gradient → proportional → weak near zero  

As a result, L1 regularization drives many gate values to (or near) zero, enabling effective pruning.

---

## 3. Experimental Setup

- Dataset: CIFAR-10  
- Model: 4-layer fully connected network  
- Optimizer: Adam  
- Epochs: 30  
- Batch size: 128  
- Learning rate scheduler: Cosine Annealing  

Three values of λ were tested to observe the sparsity–accuracy trade-off.

---

## 4. Results

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:----------:|:-----------------:|:------------------:|
| 0.05       | 49.72             | 7.80               |
| 0.20       | 50.11             | 46.60              |
| 1.00       | 48.75             | 92.88              |

---

## 5. Analysis

- **Low λ (0.05):**
  - Minimal pruning (~7.8%)
  - High accuracy (~49.7%)
  - Most weights remain active

- **Medium λ (0.20):**
  - Moderate pruning (~46.6%)
  - Highest accuracy (~50.1%)
  - Best balance between sparsity and performance

- **High λ (1.00):**
  - Aggressive pruning (~92.9%)
  - Slight drop in accuracy (~48.7%)
  - Many connections removed

### Key Insight

Increasing λ increases sparsity but may reduce accuracy, demonstrating a clear trade-off between model size and performance.

---

## 6. Gate Distribution

The distribution of gate values shows:

- A **large spike near zero** → pruned weights  
- A **spread of values away from zero** → important weights  

A threshold of 0.1 is used to determine whether a weight is considered pruned.

---

## 7. Outputs Generated

- `gate_dist_lambda_0.05.png`
- `gate_dist_lambda_0.2.png`
- `gate_dist_lambda_1.0.png`
- `tradeoff.png`

These plots visualize:
- Distribution of gate values
- Accuracy vs sparsity trade-off

---

## 8. Conclusion

This project successfully demonstrates a self-pruning neural network where:

- The model learns to identify and remove unnecessary weights
- L1 regularization effectively induces sparsity
- A clear trade-off exists between accuracy and sparsity

The approach enables efficient model compression while maintaining competitive performance, making it suitable for real-world deployment scenarios.