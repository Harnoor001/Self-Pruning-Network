# Structured Pruning Benchmark

Structured models physically remove hidden neurons and transfer surviving rows, columns, biases, and BatchNorm state. Parameter and MAC reductions below are computed from the instantiated compact models.

## Environment and protocol

```json
{
  "dataset": "CIFAR-10",
  "seed": 42,
  "source_checkpoint": "artifacts/final_benchmark/checkpoints/soft_lambda_0.0000.pt",
  "architecture": [
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
  "fine_tune_epochs": 5,
  "fine_tune_learning_rate": 0.0002,
  "batch_size": 128,
  "batch_sizes": [
    1,
    32
  ],
  "warmup_iterations": 10,
  "measurement_iterations": 30,
  "device": "cpu",
  "torch_version": "2.11.0+cpu",
  "python_version": "3.11.9",
  "neuron_importance": "mean sigmoid gate value across each output row",
  "parameter_definition": "all trainable compact weights, biases, and BatchNorm parameters; gate scores excluded from deployment",
  "benchmark_timestamp_utc": "2026-09-04T12:49:05.723065+00:00",
  "dense_checkpoint": "artifacts/final_benchmark/checkpoints/dense_lambda_0.0000.pt",
  "unstructured_checkpoint": "artifacts/final_benchmark/checkpoints/hard_target_0.6000.pt",
  "source_soft_test_accuracy_percent": 52.28,
  "source_payload_format_version": 3
}
```

## Results

| model            | architecture                      |   target_neuron_sparsity_percent |   test_accuracy_percent |   accuracy_drop_percent_points |   trainable_parameters |   deployable_parameters |   parameter_reduction_percent |   effective_macs |   mac_reduction_percent |   checkpoint_size_mb |   batch_1_mean_ms |   batch_32_mean_ms |
|:-----------------|:----------------------------------|---------------------------------:|------------------------:|-------------------------------:|-----------------------:|------------------------:|------------------------------:|-----------------:|------------------------:|---------------------:|------------------:|-------------------:|
| Dense            | 3072 -> 2048 -> 1024 -> 512 -> 10 |                              nan |                   52.21 |                           0    |               17846794 |                 8928778 |                        0      |          8918016 |                  0      |             107.104  |          4.05356  |           21.4083  |
| Unstructured 60% | 3072 -> 2048 -> 1024 -> 512 -> 10 |                              nan |                   51.84 |                           0.44 |               17846794 |                 8928778 |                        0      |          3567206 |                 60      |             107.104  |         12.9301   |           35.0343  |
| Structured 20%   | 3072 -> 1639 -> 820 -> 410 -> 10  |                               20 |                   52.89 |                          -0.61 |                6727905 |                 6727905 |                       24.6492 |          6719288 |                 24.6549 |              26.9858 |          1.14064  |            6.55786 |
| Structured 40%   | 3072 -> 1229 -> 615 -> 308 -> 10  |                               40 |                   52.72 |                          -0.44 |                4730289 |                 4730289 |                       47.022  |          4723823 |                 47.0306 |              18.9879 |          0.646763 |            4.2202  |
| Structured 60%   | 3072 -> 820 -> 410 -> 205 -> 10   |                               60 |                   52.05 |                           0.23 |                2945655 |                 2945655 |                       67.0094 |          2941340 |                 67.018  |              11.8416 |          0.506517 |            2.61267 |

## Layer-wise neuron pruning

| model          |   layer | layer_type   |   original_neurons |   remaining_neurons |   pruned_neurons |   pruning_percent |
|:---------------|--------:|:-------------|-------------------:|--------------------:|-----------------:|------------------:|
| Structured 20% |       1 | hidden       |               2048 |                1639 |              409 |           19.9707 |
| Structured 20% |       2 | hidden       |               1024 |                 820 |              204 |           19.9219 |
| Structured 20% |       3 | hidden       |                512 |                 410 |              102 |           19.9219 |
| Structured 20% |       4 | output       |                 10 |                  10 |                0 |            0      |
| Structured 40% |       1 | hidden       |               2048 |                1229 |              819 |           39.9902 |
| Structured 40% |       2 | hidden       |               1024 |                 615 |              409 |           39.9414 |
| Structured 40% |       3 | hidden       |                512 |                 308 |              204 |           39.8438 |
| Structured 40% |       4 | output       |                 10 |                  10 |                0 |            0      |
| Structured 60% |       1 | hidden       |               2048 |                 820 |             1228 |           59.9609 |
| Structured 60% |       2 | hidden       |               1024 |                 410 |              614 |           59.9609 |
| Structured 60% |       3 | hidden       |                512 |                 205 |              307 |           59.9609 |
| Structured 60% |       4 | output       |                 10 |                  10 |                0 |            0      |

## Interpretation

The Dense and Unstructured 60% rows are existing validated checkpoints. Unstructured MAC reduction is an ideal masked estimate, while Structured MAC reduction is calculated from physically smaller dense matrices. Latency and checkpoint sizes are measured in this run.
