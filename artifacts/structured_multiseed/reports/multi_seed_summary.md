# Structured Pruning Multi-Seed Validation

```json
{
  "dataset": "CIFAR-10",
  "seeds": [
    42,
    123,
    2024
  ],
  "targets_percent": [
    40.0,
    60.0
  ],
  "fine_tune_epochs": 5,
  "batch_size": 128,
  "fine_tune_learning_rate": 0.0002,
  "weight_decay": 0.0001,
  "label_smoothing": 0.05,
  "source_checkpoints": {
    "42": "C:\\Users\\sharn\\OneDrive\\Desktop\\Self pruning Network\\artifacts\\final_benchmark\\checkpoints\\soft_lambda_0.0000.pt",
    "123": "C:\\Users\\sharn\\OneDrive\\Desktop\\Self pruning Network\\artifacts\\ablation_key_benchmark\\seed123\\checkpoints\\soft_lambda_0.0000.pt",
    "2024": "C:\\Users\\sharn\\OneDrive\\Desktop\\Self pruning Network\\artifacts\\ablation_key_benchmark\\seed2024\\checkpoints\\soft_lambda_0.0000.pt"
  }
}
```

## Per-seed results

|   seed | method             |   target_sparsity_percent |   actual_structural_pruning_percent |   fine_tune_epochs | architecture                     |   validation_accuracy_percent |   test_accuracy_percent |   before_finetune_test_accuracy_percent |   fine_tune_recovery_percent_points |   soft_source_accuracy_percent |   dense_reference_accuracy_percent |   accuracy_drop_vs_soft_percent_points |   accuracy_drop_vs_dense_percent_points |   parameters |   parameter_reduction_percent |   effective_macs |   mac_reduction_percent | checkpoint                                                                     |
|-------:|:-------------------|--------------------------:|------------------------------------:|-------------------:|:---------------------------------|------------------------------:|------------------------:|----------------------------------------:|------------------------------------:|-------------------------------:|-----------------------------------:|---------------------------------------:|----------------------------------------:|-------------:|------------------------------:|-----------------:|------------------------:|:-------------------------------------------------------------------------------|
|     42 | structured_learned |                        40 |                                  40 |                  5 | 3072 -> 1229 -> 615 -> 308 -> 10 |                         52.72 |                   52.8  |                                   28.02 |                               24.78 |                          52.28 |                              52.21 |                                  -0.52 |                                   -0.59 |      4730289 |                       47.022  |          4723823 |                 47.0306 | artifacts/structured_multiseed/checkpoints/structured_seed42_target40_ft5.pt   |
|    123 | structured_learned |                        40 |                                  40 |                  5 | 3072 -> 1229 -> 615 -> 308 -> 10 |                         52.52 |                   52.84 |                                   26.97 |                               25.87 |                          51.81 |                              51.8  |                                  -1.03 |                                   -1.04 |      4730289 |                       47.022  |          4723823 |                 47.0306 | artifacts/structured_multiseed/checkpoints/structured_seed123_target40_ft5.pt  |
|   2024 | structured_learned |                        40 |                                  40 |                  5 | 3072 -> 1229 -> 615 -> 308 -> 10 |                         51.72 |                   52.83 |                                   25.81 |                               27.02 |                          51.69 |                              52.01 |                                  -1.14 |                                   -0.82 |      4730289 |                       47.022  |          4723823 |                 47.0306 | artifacts/structured_multiseed/checkpoints/structured_seed2024_target40_ft5.pt |
|     42 | structured_learned |                        60 |                                  60 |                  5 | 3072 -> 820 -> 410 -> 205 -> 10  |                         51.5  |                   51.84 |                                   21.75 |                               30.09 |                          52.28 |                              52.21 |                                   0.44 |                                    0.37 |      2945655 |                       67.0094 |          2941340 |                 67.018  | artifacts/structured_multiseed/checkpoints/structured_seed42_target60_ft5.pt   |
|    123 | structured_learned |                        60 |                                  60 |                  5 | 3072 -> 820 -> 410 -> 205 -> 10  |                         51.16 |                   51.47 |                                   19.43 |                               32.04 |                          51.81 |                              51.8  |                                   0.34 |                                    0.33 |      2945655 |                       67.0094 |          2941340 |                 67.018  | artifacts/structured_multiseed/checkpoints/structured_seed123_target60_ft5.pt  |
|   2024 | structured_learned |                        60 |                                  60 |                  5 | 3072 -> 820 -> 410 -> 205 -> 10  |                         49.86 |                   50.43 |                                   17.78 |                               32.65 |                          51.69 |                              52.01 |                                   1.26 |                                    1.58 |      2945655 |                       67.0094 |          2941340 |                 67.018  | artifacts/structured_multiseed/checkpoints/structured_seed2024_target60_ft5.pt |

## Mean +/- sample standard deviation

| method             |   target_sparsity_percent |   mean_accuracy_percent |   std_accuracy_percent |   mean_accuracy_drop_vs_dense_percent_points |   std_accuracy_drop_vs_dense_percent_points |
|:-------------------|--------------------------:|------------------------:|-----------------------:|---------------------------------------------:|--------------------------------------------:|
| structured_learned |                        40 |                 52.8233 |              0.0208167 |                                    -0.816667 |                                    0.225019 |
| structured_learned |                        60 |                 51.2467 |              0.731049  |                                     0.76     |                                    0.710422 |
