# Structured Pruning Fine-Tuning Recovery

Every budget starts from the same transferred compact model for its target. The optimizer, data, seed, augmentation, and evaluation protocol are unchanged; only the number of fixed-architecture fine-tuning epochs varies.

```json
{
  "dataset": "CIFAR-10",
  "seed": 42,
  "train_samples": 45000,
  "validation_samples": 5000,
  "test_samples": 10000,
  "source_checkpoint": "artifacts/final_benchmark/checkpoints/soft_lambda_0.0000.pt",
  "source_architecture": [
    3072,
    2048,
    1024,
    512,
    10
  ],
  "targets_percent": [
    20.0,
    40.0,
    60.0
  ],
  "fine_tune_budgets": [
    0,
    1,
    3,
    5
  ],
  "optimizer": "AdamW",
  "fine_tune_learning_rate": 0.0002,
  "weight_decay": 0.0001,
  "label_smoothing": 0.05,
  "importance": "mean sigmoid gate value across outgoing rows"
}
```

## Results

|   target_neuron_sparsity_percent |   fine_tune_epochs | architecture                     |   validation_accuracy_percent |   test_accuracy_percent |   before_finetune_test_accuracy_percent |   fine_tune_recovery_percent_points |   accuracy_drop_vs_soft_percent_points |   accuracy_drop_vs_dense_percent_points |   trainable_parameters |   parameter_reduction_percent |   mac_reduction_percent |
|---------------------------------:|-------------------:|:---------------------------------|------------------------------:|------------------------:|----------------------------------------:|------------------------------------:|---------------------------------------:|----------------------------------------:|-----------------------:|------------------------------:|------------------------:|
|                               20 |                  0 | 3072 -> 1639 -> 820 -> 410 -> 10 |                         35.96 |                   37.89 |                                   37.89 |                                0    |                                  14.39 |                                   14.32 |                6727905 |                       24.6492 |                 24.6549 |
|                               20 |                  1 | 3072 -> 1639 -> 820 -> 410 -> 10 |                         51.52 |                   50.54 |                                   37.89 |                               12.65 |                                   1.74 |                                    1.67 |                6727905 |                       24.6492 |                 24.6549 |
|                               20 |                  3 | 3072 -> 1639 -> 820 -> 410 -> 10 |                         52.74 |                   52.75 |                                   37.89 |                               14.86 |                                  -0.47 |                                   -0.54 |                6727905 |                       24.6492 |                 24.6549 |
|                               20 |                  5 | 3072 -> 1639 -> 820 -> 410 -> 10 |                         53    |                   53.04 |                                   37.89 |                               15.15 |                                  -0.76 |                                   -0.83 |                6727905 |                       24.6492 |                 24.6549 |
|                               40 |                  0 | 3072 -> 1229 -> 615 -> 308 -> 10 |                         26.88 |                   28.02 |                                   28.02 |                                0    |                                  24.26 |                                   24.19 |                4730289 |                       47.022  |                 47.0306 |
|                               40 |                  1 | 3072 -> 1229 -> 615 -> 308 -> 10 |                         48.9  |                   48.42 |                                   28.02 |                               20.4  |                                   3.86 |                                    3.79 |                4730289 |                       47.022  |                 47.0306 |
|                               40 |                  3 | 3072 -> 1229 -> 615 -> 308 -> 10 |                         51.46 |                   51.48 |                                   28.02 |                               23.46 |                                   0.8  |                                    0.73 |                4730289 |                       47.022  |                 47.0306 |
|                               40 |                  5 | 3072 -> 1229 -> 615 -> 308 -> 10 |                         52.58 |                   52.56 |                                   28.02 |                               24.54 |                                  -0.28 |                                   -0.35 |                4730289 |                       47.022  |                 47.0306 |
|                               60 |                  0 | 3072 -> 820 -> 410 -> 205 -> 10  |                         20.58 |                   21.75 |                                   21.75 |                                0    |                                  30.53 |                                   30.46 |                2945655 |                       67.0094 |                 67.018  |
|                               60 |                  1 | 3072 -> 820 -> 410 -> 205 -> 10  |                         47.16 |                   47.29 |                                   21.75 |                               25.54 |                                   4.99 |                                    4.92 |                2945655 |                       67.0094 |                 67.018  |
|                               60 |                  3 | 3072 -> 820 -> 410 -> 205 -> 10  |                         50.52 |                   50.37 |                                   21.75 |                               28.62 |                                   1.91 |                                    1.84 |                2945655 |                       67.0094 |                 67.018  |
|                               60 |                  5 | 3072 -> 820 -> 410 -> 205 -> 10  |                         51.28 |                   51.29 |                                   21.75 |                               29.54 |                                   0.99 |                                    0.92 |                2945655 |                       67.0094 |                 67.018  |
