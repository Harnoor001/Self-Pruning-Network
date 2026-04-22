# Self-Pruning Neural Network Report

## Summary

This project trains a feed-forward classifier with learnable gates attached to every weight.
Each gate is produced by a sigmoid-transformed parameter and multiplied element-wise with the corresponding weight.
An L1 penalty over the gate values encourages the optimizer to reduce the number of active connections.

## Why L1 Regularization Helps

The L1 penalty adds a direct cost for keeping gates open.
Since the gates are bounded in `(0, 1)`, the optimizer can lower the total loss by pushing many gate values toward zero.
That creates a sparse network where only useful connections remain active enough to justify their cost.

## Results

Best lambda: `0.0001`

Best test accuracy: `0.1406`

Best sparsity level: `0.00%`

|   lambda |   validation_accuracy |   test_accuracy |   sparsity_percent | checkpoint_path                                          |
|---------:|----------------------:|----------------:|-------------------:|:---------------------------------------------------------|
|   0.0001 |                0.0833 |          0.1406 |                  0 | artifacts_graphs_smoke\checkpoints\best_lambda_0.0001.pt |
|   0.001  |                0.0833 |          0.125  |                  0 | artifacts_graphs_smoke\checkpoints\best_lambda_0.0010.pt |
