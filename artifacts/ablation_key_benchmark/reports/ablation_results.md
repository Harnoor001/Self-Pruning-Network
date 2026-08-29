# Ablation Benchmark

Learned and random pruning start from the same unpruned soft-gated checkpoint for each seed. They use the same global target sparsity, one fixed mask during fine-tuning, and the same data/evaluation protocol.

## Per-run results

|   seed | method               |   lambda |   target_sparsity_percent |   actual_sparsity_percent |   validation_accuracy_percent |   test_accuracy_percent |   accuracy_drop_percent_points |   pre_finetune_test_accuracy_percent |   fine_tune_recovery_percent_points |   active_connections |   pruned_connections |
|-------:|:---------------------|---------:|--------------------------:|--------------------------:|------------------------------:|------------------------:|-------------------------------:|-------------------------------------:|------------------------------------:|---------------------:|---------------------:|
|     42 | dense                |  nan     |                         0 |                         0 |                         51.66 |                   52.21 |                           0    |                               nan    |                              nan    |              8918016 |                    0 |
|     42 | learned              |    0     |                        60 |                        60 |                         51.6  |                   51.82 |                           0.46 |                                49.8  |                                2.02 |              3567206 |              5350810 |
|     42 | random               |    0     |                        60 |                        60 |                         50.56 |                   50.84 |                           1.44 |                                13.65 |                               37.19 |              3567206 |              5350810 |
|     42 | soft_lambda_0        |    0     |                         0 |                         0 |                         51.74 |                   52.28 |                          -0.07 |                               nan    |                              nan    |              8918016 |                    0 |
|     42 | soft_lambda_positive |    0.001 |                         0 |                         0 |                         51.5  |                   52.27 |                          -0.06 |                               nan    |                              nan    |              8918016 |                    0 |
|    123 | dense                |  nan     |                         0 |                         0 |                         50.52 |                   51.8  |                           0    |                               nan    |                              nan    |              8918016 |                    0 |
|    123 | learned              |    0     |                        60 |                        60 |                         50.38 |                   51.74 |                           0.07 |                                46.9  |                                4.84 |              3567206 |              5350810 |
|    123 | random               |    0     |                        60 |                        60 |                         50.52 |                   51.06 |                           0.75 |                                12.45 |                               38.61 |              3567206 |              5350810 |
|    123 | soft_lambda_0        |    0     |                         0 |                         0 |                         50.7  |                   51.81 |                          -0.01 |                               nan    |                              nan    |              8918016 |                    0 |
|   2024 | dense                |  nan     |                         0 |                         0 |                         50.46 |                   52.01 |                           0    |                               nan    |                              nan    |              8918016 |                    0 |
|   2024 | learned              |    0     |                        60 |                        60 |                         50.48 |                   51.46 |                           0.23 |                                49.33 |                                2.13 |              3567206 |              5350810 |
|   2024 | random               |    0     |                        60 |                        60 |                         49.42 |                   50.5  |                           1.19 |                                11.8  |                               38.7  |              3567206 |              5350810 |
|   2024 | soft_lambda_0        |    0     |                         0 |                         0 |                         50.4  |                   51.69 |                           0.32 |                               nan    |                              nan    |              8918016 |                    0 |

## Learned advantage over random

|   seed |   target_sparsity_percent |   learned_test_accuracy_percent |   random_test_accuracy_percent |   learned_advantage_percent_points |
|-------:|--------------------------:|--------------------------------:|-------------------------------:|-----------------------------------:|
|     42 |                        60 |                           51.82 |                          50.84 |                               0.98 |
|    123 |                        60 |                           51.74 |                          51.06 |                               0.68 |
|   2024 |                        60 |                           51.46 |                          50.5  |                               0.96 |

## Mean learned advantage over random

|   target_sparsity_percent |   mean_learned_advantage_percent_points |   std_learned_advantage_percent_points |   seed_count |
|--------------------------:|----------------------------------------:|---------------------------------------:|-------------:|
|                        60 |                                0.873333 |                                0.16773 |            3 |

## Accuracy/sparsity Pareto configurations

learned seed42 60%

## Interpretation

The learned-vs-random difference is descriptive for the evaluated seeds. Three seeds are not sufficient for a strong statistical-significance claim. Logical sparsity and identical active counts are enforced by construction; accuracy differences measure the selection strategy and fine-tuning outcome.
