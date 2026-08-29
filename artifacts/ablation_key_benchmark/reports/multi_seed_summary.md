# Multi-Seed Summary

Mean and sample standard deviation are computed from the available seeds. A standard deviation of zero indicates only one seed was available for that target.

| method        |   target_sparsity_percent |   mean_test_accuracy |   std_test_accuracy |   mean_accuracy_drop |   std_accuracy_drop |   mean_actual_sparsity |   std_actual_sparsity |   seed_count |
|:--------------|--------------------------:|---------------------:|--------------------:|---------------------:|--------------------:|-----------------------:|----------------------:|-------------:|
| dense         |                         0 |              52.0067 |            0.20502  |             0        |            0        |                      0 |                     0 |            3 |
| learned       |                        60 |              51.6733 |            0.189033 |             0.253333 |            0.196044 |                     60 |                     0 |            3 |
| random        |                        60 |              50.8    |            0.282135 |             1.12667  |            0.349333 |                     60 |                     0 |            3 |
| soft_lambda_0 |                         0 |              51.9267 |            0.311823 |             0.08     |            0.21     |                      0 |                     0 |            3 |
