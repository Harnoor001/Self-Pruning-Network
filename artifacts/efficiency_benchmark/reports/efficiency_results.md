# Efficiency and Compression Benchmark

This report benchmarks the validated final checkpoints without retraining. Connectivity and MAC values are calculated from checkpoint masks; latency and checkpoint sizes are measured on this execution environment.

## Environment and protocol

```json
{
  "benchmark_timestamp_utc": "2026-08-28T16:51:23.053217+00:00",
  "benchmark_dir": "artifacts\\final_benchmark",
  "seed": 42,
  "device": "cpu",
  "device_name": "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel",
  "torch_version": "2.11.0+cpu",
  "python_version": "3.11.9",
  "platform": "Windows-10-10.0.26200-SP0",
  "torch_num_threads": 1,
  "batch_sizes": [
    1,
    32
  ],
  "warmup_iterations": 20,
  "measurement_iterations": 50,
  "accuracy_source": "artifacts\\final_benchmark\\reports\\results.csv",
  "accuracy_sparsity_pareto_frontier": [
    "Soft",
    "Hard 20%",
    "Hard 40%",
    "Hard 60%",
    "Hard 80%"
  ],
  "latency_accuracy_pareto_frontier_batch_1": [
    "Dense",
    "Soft"
  ],
  "models": [
    "Dense",
    "Soft",
    "Hard 20%",
    "Hard 40%",
    "Hard 60%",
    "Hard 80%"
  ],
  "layer_report": "layer_efficiency.csv"
}
```

`estimated_effective_macs` and `theoretical_mac_reduction_percent` assume ideal sparse execution. The benchmark models still use dense PyTorch tensors with binary masks, so these values are not measured runtime speedups.

## Results

| model    |   lambda |   target_sparsity_percent |   sparsity_percent |   density_percent |   test_accuracy_percent |   accuracy_drop_percent_points |   active_connections |   pruned_connections |   estimated_effective_macs |   theoretical_mac_reduction_percent |   checkpoint_size_bytes |   checkpoint_size_mb |   checkpoint_size_mib |   batch_1_mean_ms |   batch_1_p50_ms |   batch_1_p95_ms |   batch_1_latency_reduction_vs_dense_percent |   batch_32_mean_ms |   batch_32_p50_ms |   batch_32_p95_ms |   batch_32_latency_reduction_vs_dense_percent |
|:---------|---------:|--------------------------:|-------------------:|------------------:|------------------------:|-------------------------------:|---------------------:|---------------------:|---------------------------:|------------------------------------:|------------------------:|---------------------:|----------------------:|------------------:|-----------------:|-----------------:|---------------------------------------------:|-------------------:|------------------:|------------------:|----------------------------------------------:|
| Dense    |      nan |                         0 |                  0 |               100 |                   52.21 |                           0    |              8918016 |                    0 |                    8918016 |                                   0 |               107104166 |              107.104 |               102.142 |           3.44323 |           3.3126 |          4.24244 |                                        0     |            21.4313 |           21.0695 |           23.8298 |                                        0      |
| Soft     |        0 |                         0 |                  0 |               100 |                   52.28 |                          -0.07 |              8918016 |                    0 |                    8918016 |                                   0 |               107104125 |              107.104 |               102.142 |          29.4522  |          29.4728 |         32.1737  |                                     -755.366 |            54.0272 |           52.8449 |           58.0642 |                                     -152.095  |
| Hard 20% |        0 |                        20 |                 20 |                80 |                   52.08 |                           0.2  |              7134413 |              1783603 |                    7134413 |                                  20 |               107103805 |              107.104 |               102.142 |          13.2855  |          13.2094 |         14.832   |                                     -285.844 |            30.3946 |           29.7854 |           32.3635 |                                      -41.8237 |
| Hard 40% |        0 |                        40 |                 40 |                60 |                   51.87 |                           0.41 |              5350810 |              3567206 |                    5350810 |                                  40 |               107103805 |              107.104 |               102.142 |          14.2498  |          13.7504 |         16.2722  |                                     -313.85  |            35.9673 |           35.825  |           38.7332 |                                      -67.8264 |
| Hard 60% |        0 |                        60 |                 60 |                40 |                   51.84 |                           0.44 |              3567206 |              5350810 |                    3567206 |                                  60 |               107103805 |              107.104 |               102.142 |          13.6661  |          13.1026 |         14.9089  |                                     -296.897 |            30.4597 |           29.7564 |           36.4922 |                                      -42.1277 |
| Hard 80% |        0 |                        80 |                 80 |                20 |                   49.11 |                           3.17 |              1783603 |              7134413 |                    1783603 |                                  80 |               107103805 |              107.104 |               102.142 |          13.6941  |          13.4721 |         16.3438  |                                     -297.71  |            30.6548 |           29.5883 |           37.4945 |                                      -43.038  |

Latency reduction is measured relative to Dense at the same batch size; negative values mean the masked dense model was slower. Checkpoint size is the actual file size on disk, not a compressed sparse representation.

## Accuracy/sparsity Pareto frontier

Soft, Hard 20%, Hard 40%, Hard 60%, Hard 80%

## Latency/accuracy frontier (batch size 1)

Dense, Soft

## Layer-wise connectivity and MAC accounting

| model    | layer_name   |   layer_index |   input_features |   output_features |   total_weights |   active_weights |   pruned_weights |   sparsity_percent |   density_percent |   dense_macs |   estimated_effective_macs |
|:---------|:-------------|--------------:|-----------------:|------------------:|----------------:|-----------------:|-----------------:|-------------------:|------------------:|-------------:|---------------------------:|
| Dense    | layer1       |             0 |             3072 |              2048 |         6291456 |          6291456 |                0 |             0      |         100       |      6291456 |                    6291456 |
| Dense    | layer2       |             1 |             2048 |              1024 |         2097152 |          2097152 |                0 |             0      |         100       |      2097152 |                    2097152 |
| Dense    | layer3       |             2 |             1024 |               512 |          524288 |           524288 |                0 |             0      |         100       |       524288 |                     524288 |
| Dense    | layer4       |             3 |              512 |                10 |            5120 |             5120 |                0 |             0      |         100       |         5120 |                       5120 |
| Soft     | layer1       |             0 |             3072 |              2048 |         6291456 |          6291456 |                0 |             0      |         100       |      6291456 |                    6291456 |
| Soft     | layer2       |             1 |             2048 |              1024 |         2097152 |          2097152 |                0 |             0      |         100       |      2097152 |                    2097152 |
| Soft     | layer3       |             2 |             1024 |               512 |          524288 |           524288 |                0 |             0      |         100       |       524288 |                     524288 |
| Soft     | layer4       |             3 |              512 |                10 |            5120 |             5120 |                0 |             0      |         100       |         5120 |                       5120 |
| Hard 20% | layer1       |             0 |             3072 |              2048 |         6291456 |          5049858 |          1241598 |            19.7347 |          80.2653  |      6291456 |                    5049858 |
| Hard 20% | layer2       |             1 |             2048 |              1024 |         2097152 |          1664543 |           432609 |            20.6284 |          79.3716  |      2097152 |                    1664543 |
| Hard 20% | layer3       |             2 |             1024 |               512 |          524288 |           417244 |           107044 |            20.417  |          79.583   |       524288 |                     417244 |
| Hard 20% | layer4       |             3 |              512 |                10 |            5120 |             2768 |             2352 |            45.9375 |          54.0625  |         5120 |                       2768 |
| Hard 40% | layer1       |             0 |             3072 |              2048 |         6291456 |          3771836 |          2519620 |            40.0483 |          59.9517  |      6291456 |                    3771836 |
| Hard 40% | layer2       |             1 |             2048 |              1024 |         2097152 |          1255605 |           841547 |            40.1281 |          59.8719  |      2097152 |                    1255605 |
| Hard 40% | layer3       |             2 |             1024 |               512 |          524288 |           321408 |           202880 |            38.6963 |          61.3037  |       524288 |                     321408 |
| Hard 40% | layer4       |             3 |              512 |                10 |            5120 |             1961 |             3159 |            61.6992 |          38.3008  |         5120 |                       1961 |
| Hard 60% | layer1       |             0 |             3072 |              2048 |         6291456 |          2476666 |          3814790 |            60.6345 |          39.3655  |      6291456 |                    2476666 |
| Hard 60% | layer2       |             1 |             2048 |              1024 |         2097152 |           863508 |          1233644 |            58.8247 |          41.1753  |      2097152 |                     863508 |
| Hard 60% | layer3       |             2 |             1024 |               512 |          524288 |           225847 |           298441 |            56.9231 |          43.0769  |       524288 |                     225847 |
| Hard 60% | layer4       |             3 |              512 |                10 |            5120 |             1185 |             3935 |            76.8555 |          23.1445  |         5120 |                       1185 |
| Hard 80% | layer1       |             0 |             3072 |              2048 |         6291456 |          1227795 |          5063661 |            80.4847 |          19.5153  |      6291456 |                    1227795 |
| Hard 80% | layer2       |             1 |             2048 |              1024 |         2097152 |           439789 |          1657363 |            79.0292 |          20.9708  |      2097152 |                     439789 |
| Hard 80% | layer3       |             2 |             1024 |               512 |          524288 |           115568 |           408720 |            77.9572 |          22.0428  |       524288 |                     115568 |
| Hard 80% | layer4       |             3 |              512 |                10 |            5120 |              451 |             4669 |            91.1914 |           8.80859 |         5120 |                        451 |

The complete unrounded layer metrics are also available in `layer_efficiency.csv`.

## Interpretation

Logical sparsity and ideal MAC reduction are not physical parameter removal. Dense tensor shapes remain allocated and ordinary dense kernels may not skip zero entries. Actual storage reduction or acceleration requires a sparse representation, structured pruning, sparse kernels, or hardware/runtime support.
