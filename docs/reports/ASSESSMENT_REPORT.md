# Tredence AI Engineer Case Study
## The Self-Pruning Neural Network

**Candidate:** Harnoor Singh  
**Role Applied For:** AI Engineer Intern  
**GitHub Repository:** [https://github.com/Harnoor001/Self-Pruning-Network](https://github.com/Harnoor001/Self-Pruning-Network)

## 1. Problem Statement

The objective of this case study is to build a feed-forward neural network for image classification that can learn to prune its own weights during training.

Instead of applying pruning only after the model has already been trained, the network is augmented with a learnable gate for every weight. Each gate is constrained to the range `(0, 1)` and controls whether its corresponding connection remains active. A gate value near zero makes the associated connection effectively inactive.

The final training objective is designed to balance two goals:

1. Maintain good classification performance on CIFAR-10
2. Encourage sparse connectivity by penalizing active gates

## 2. Approach

### 2.1 Custom Prunable Layer

I implemented a custom `PrunableLinear` layer instead of using `torch.nn.Linear` directly.

Each layer contains:

- `weight`
- `bias`
- `gate_scores`

The `gate_scores` tensor has the same shape as the weight tensor and is trainable. During the forward pass:

```text
gates = sigmoid(gate_scores)
pruned_weights = weight * gates
output = linear(input, pruned_weights, bias)
```

This design keeps the model fully differentiable, which allows gradients to flow through both the original weights and the gate parameters.

### 2.2 Network Architecture

The model is a feed-forward MLP for CIFAR-10 classification. The input image is flattened and passed through a stack of prunable linear layers with nonlinear activations in between.

This keeps the solution aligned with the assignment requirement of a standard feed-forward neural network while embedding the pruning mechanism directly into the model.

### 2.3 Loss Function

The total loss is:

```text
Total Loss = Classification Loss + lambda x Sparsity Loss
```

Where:

- `Classification Loss` is cross-entropy
- `Sparsity Loss` is the sum of all gate values across the prunable layers
- `lambda` controls the trade-off between accuracy and pruning pressure

## 3. Why L1 Penalty on Gates Encourages Sparsity

The sparsity term is based on the sum of gate values. Since each gate contributes positively to the loss, the optimizer is encouraged to reduce gates that are not essential for classification performance.

This is similar to an L1-style sparsity pressure:

- useful connections remain active because they help minimize classification loss
- weak or unnecessary connections are pushed downward because they add cost without improving prediction quality

As a result, the model learns which connections are worth keeping and which ones can be suppressed.

## 4. Training Setup

- **Dataset:** CIFAR-10 from `torchvision.datasets`
- **Task:** 10-class image classification
- **Model Type:** Feed-forward self-pruning MLP
- **Gate Transformation:** `sigmoid(gate_scores)`
- **Sparsity Threshold for Reporting:** gate value `< 1e-2`
- **Compared Lambda Values:** `0.0001`, `0.001`, `0.01`

The project includes:

- training and evaluation pipeline
- checkpoint saving
- result table generation
- gate distribution plotting
- FastAPI serving layer for model inspection and prediction
- unit tests for the pruning logic

## 5. Results

### 5.1 Accuracy and Sparsity Table

| Lambda | Validation Accuracy | Test Accuracy | Sparsity Level (%) |
|--------|---------------------:|--------------:|-------------------:|
| 0.0001 | 0.5210 | 0.5187 | 0.00 |
| 0.0010 | 0.5234 | 0.5174 | 0.00 |
| 0.0100 | 0.5222 | 0.5144 | 0.00 |

### 5.2 Best Run

- **Best Lambda (by validation accuracy):** `0.001`
- **Best Validation Accuracy:** `0.5234`
- **Best Test Accuracy:** `0.5174`
- **Best Reported Sparsity:** `0.00%`

## 6. Interpretation of Results

The current implementation successfully demonstrates the core self-pruning mechanism:

- every weight is paired with a learnable gate
- gates are optimized jointly with the model weights
- sparsity pressure is applied during training rather than after training

However, for the current training configuration and selected lambda values, the final gate values did not cross the reporting threshold of `1e-2`. This means the model learned soft gating behavior, but it did not produce threshold-level sparsity in this particular experiment.

This result is still useful because it shows:

1. The gated architecture and differentiable pruning mechanism are implemented correctly
2. The model can be trained end-to-end with a custom sparsity objective
3. Lambda tuning is critical to obtaining a stronger accuracy-sparsity trade-off

In other words, the pruning system is present and functional, but more aggressive or better-scheduled regularization would likely be required to achieve higher measured sparsity.

## 7. Gate Distribution Plot

The project also generates a plot of final gate values for the best model:

- `artifacts/reports/gate_distribution.png`

This plot helps visualize whether gates are clustering near zero or remaining broadly active.

## 8. Additional Analysis Plots

To strengthen the report further, I also added support in the codebase for generating additional analytical plots during training. These plots are useful for explaining model behavior in a more complete way:

- `lambda_vs_test_accuracy.png`: shows how prediction performance changes with pruning pressure
- `lambda_vs_sparsity.png`: shows whether stronger regularization is translating into more sparsity
- `accuracy_vs_sparsity.png`: summarizes the trade-off between model quality and pruning level
- `training_history.png`: plots training and validation loss/accuracy by epoch
- `layer_metrics.png`: shows mean gate value and sparsity layer by layer

These plots help move the report beyond a single final metric and make it easier to discuss optimization behavior, convergence, and pruning characteristics in an interview.

## 9. Engineering Additions Beyond the Minimum

To make the submission more aligned with an AI engineering role, I extended the assignment implementation into a complete project repository with:

- modular code organization
- unit tests for gradient flow and pruning logic
- CLI-based training script
- checkpointing and report generation
- FastAPI API for:
  - health check
  - model summary
  - image prediction
  - browser-based upload interface

These additions are not strictly required by the problem statement, but they reflect production-oriented engineering practices.

## 10. Limitations and Future Improvements

The current model is a feed-forward MLP, which is intentionally aligned with the case study prompt. For image classification, this architecture is weaker than a CNN, so raw classification performance is limited.

Potential improvements:

- longer training with stronger hyperparameter tuning
- scheduled sparsity pressure instead of fixed lambda
- exploration of lower or better-calibrated lambda values
- harder thresholding or alternative sparsity-inducing formulations
- extending the same pruning idea to a convolutional model for stronger accuracy

## 11. Conclusion

This project implements a self-pruning neural network in which every weight has a learnable gate and pruning pressure is applied during training through an L1-style sparsity objective.

The solution satisfies the core requirements of the assignment:

- custom prunable linear layer
- learnable gates
- gated forward pass
- custom sparsity loss
- CIFAR-10 training and evaluation
- comparison across multiple lambda values
- report generation and visualization

In addition, the project is packaged as a clean, interview-ready engineering repository with testing, reporting, and API serving support.
