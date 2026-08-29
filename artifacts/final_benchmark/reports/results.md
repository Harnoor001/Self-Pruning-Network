# Self-Pruning Neural Network Report

## Summary

This project trains a feed-forward classifier with learnable gates attached to every weight.
Each gate is produced by a sigmoid-transformed parameter and multiplied element-wise with the corresponding weight.
An L1 penalty over the gate values encourages the optimizer to reduce the number of active connections.

## Why L1 Regularization Helps

The L1 penalty adds a direct cost for keeping gates open.
Since the gates are bounded in `(0, 1)`, the optimizer can lower the total loss by pushing many gate values toward zero.
That creates a sparse network where only useful connections remain active enough to justify their cost.

## Method

The model learns `W' = W * sigmoid(S)` jointly with classification. When pruning is enabled, gate importance is converted to a persistent binary mask. A hard-pruned forward pass uses `W_hard = W * M`, and fine-tuning reapplies `M` after every optimizer step.

Logical sparsity is the fraction of masked connections. The tensors are still stored densely, so these results do not claim proportional storage savings or hardware speedup.

## Dense reference

Separately trained dense-mode test accuracy: **0.522100**. Accuracy drops for soft models are measured against this dense reference; hard-pruned drops are measured against the corresponding unpruned soft model.

## Experiment configuration

```json
{
  "dataset":"CIFAR-10",
  "data_dir":"data",
  "train_subset":null,
  "test_subset":null,
  "validation_ratio":0.1,
  "seed":42,
  "train_samples":45000,
  "validation_samples":5000,
  "test_samples":10000,
  "architecture":{
    "input_dim":3072,
    "hidden_dims":[
      2048,
      1024,
      512
    ],
    "num_classes":10,
    "dropout":0.3,
    "use_batchnorm":true
  },
  "epochs":5,
  "batch_size":128,
  "learning_rate":0.0008,
  "optimizer":"AdamW",
  "weight_decay":0.0001,
  "scheduler":"CosineAnnealingLR",
  "label_smoothing":0.05,
  "fine_tune_epochs":1,
  "fine_tune_learning_rate":0.0002,
  "enabled":true,
  "strategy":"target",
  "threshold":0.1,
  "target_sparsities_percent":[
    20.0,
    40.0,
    60.0,
    80.0
  ],
  "selected_lambda":0.0
}
```

## Results

Best lambda: `0.0`

Best test accuracy: `0.5228`

Best soft-model mask sparsity: `0.00%`

Pruning enabled: `True`

| model_type   |   lambda |   target_sparsity_percent |   validation_accuracy |   test_accuracy |   sparsity_percent |   density_percent |   total_weights |   active_weights |   pruned_weights | reference_label   |   reference_test_accuracy |   accuracy_drop_percent_points |   pre_finetune_test_accuracy | checkpoint_path                                              |
|:-------------|---------:|--------------------------:|----------------------:|----------------:|-------------------:|------------------:|----------------:|-----------------:|-----------------:|:------------------|--------------------------:|-------------------------------:|-----------------------------:|:-------------------------------------------------------------|
| dense        |  nan     |                         0 |                0.5166 |          0.5221 |                  0 |               100 |         8918016 |          8918016 |                0 | dense             |                    0.5221 |                           0    |                     nan      | artifacts\final_benchmark\checkpoints\dense_lambda_0.0000.pt |
| soft         |    0     |                       nan |                0.5174 |          0.5228 |                  0 |               100 |         8918016 |          8918016 |                0 | dense             |                    0.5221 |                          -0.07 |                     nan      | artifacts\final_benchmark\checkpoints\soft_lambda_0.0000.pt  |
| soft         |    0.001 |                       nan |                0.515  |          0.5227 |                  0 |               100 |         8918016 |          8918016 |                0 | dense             |                    0.5221 |                          -0.06 |                     nan      | artifacts\final_benchmark\checkpoints\soft_lambda_0.0010.pt  |
| hard         |    0     |                         0 |                0.5174 |          0.521  |                  0 |               100 |         8918016 |          8918016 |                0 | soft_lambda_0.0   |                    0.5228 |                           0.18 |                       0.507  | artifacts\final_benchmark\checkpoints\hard_target_0.0000.pt  |
| hard         |    0     |                        20 |                0.5252 |          0.5208 |                 20 |                80 |         8918016 |          7134413 |          1783603 | soft_lambda_0.0   |                    0.5228 |                           0.2  |                       0.5061 | artifacts\final_benchmark\checkpoints\hard_target_0.2000.pt  |
| hard         |    0     |                        40 |                0.5176 |          0.5187 |                 40 |                60 |         8918016 |          5350810 |          3567206 | soft_lambda_0.0   |                    0.5228 |                           0.41 |                       0.5065 | artifacts\final_benchmark\checkpoints\hard_target_0.4000.pt  |
| hard         |    0     |                        60 |                0.515  |          0.5184 |                 60 |                40 |         8918016 |          3567206 |          5350810 | soft_lambda_0.0   |                    0.5228 |                           0.44 |                       0.498  | artifacts\final_benchmark\checkpoints\hard_target_0.6000.pt  |
| hard         |    0     |                        80 |                0.4854 |          0.4911 |                 80 |                20 |         8918016 |          1783603 |          7134413 | soft_lambda_0.0   |                    0.5228 |                           3.17 |                       0.4015 | artifacts\final_benchmark\checkpoints\hard_target_0.8000.pt  |

## Accuracy/sparsity Pareto frontier

The non-dominated hard-pruned configurations are listed below. A configuration is non-dominated when no evaluated hard-pruned result has both higher or equal accuracy and higher or equal sparsity with one strict improvement.

| model_type   |   lambda |   target_sparsity_percent |   validation_accuracy |   test_accuracy |   sparsity_percent |   density_percent |   total_weights |   active_weights |   pruned_weights | reference_label   |   reference_test_accuracy |   accuracy_drop_percent_points |   pre_finetune_test_accuracy | checkpoint_path                                             |
|:-------------|---------:|--------------------------:|----------------------:|----------------:|-------------------:|------------------:|----------------:|-----------------:|-----------------:|:------------------|--------------------------:|-------------------------------:|-----------------------------:|:------------------------------------------------------------|
| hard         |        0 |                         0 |                0.5174 |          0.521  |                  0 |               100 |         8918016 |          8918016 |                0 | soft_lambda_0.0   |                    0.5228 |                           0.18 |                       0.507  | artifacts\final_benchmark\checkpoints\hard_target_0.0000.pt |
| hard         |        0 |                        20 |                0.5252 |          0.5208 |                 20 |                80 |         8918016 |          7134413 |          1783603 | soft_lambda_0.0   |                    0.5228 |                           0.2  |                       0.5061 | artifacts\final_benchmark\checkpoints\hard_target_0.2000.pt |
| hard         |        0 |                        40 |                0.5176 |          0.5187 |                 40 |                60 |         8918016 |          5350810 |          3567206 | soft_lambda_0.0   |                    0.5228 |                           0.41 |                       0.5065 | artifacts\final_benchmark\checkpoints\hard_target_0.4000.pt |
| hard         |        0 |                        60 |                0.515  |          0.5184 |                 60 |                40 |         8918016 |          3567206 |          5350810 | soft_lambda_0.0   |                    0.5228 |                           0.44 |                       0.498  | artifacts\final_benchmark\checkpoints\hard_target_0.6000.pt |
| hard         |        0 |                        80 |                0.4854 |          0.4911 |                 80 |                20 |         8918016 |          1783603 |          7134413 | soft_lambda_0.0   |                    0.5228 |                           3.17 |                       0.4015 | artifacts\final_benchmark\checkpoints\hard_target_0.8000.pt |

## Fine-tuning recovery

The table below records test accuracy immediately after applying the hard mask and after the configured fine-tuning stage. Accuracy-drop values in the main table use the post-fine-tuning accuracy.

|   target_sparsity_percent |   pre_finetune_test_accuracy |   test_accuracy |
|--------------------------:|-----------------------------:|----------------:|
|                         0 |                       0.507  |          0.521  |
|                        20 |                       0.5061 |          0.5208 |
|                        40 |                       0.5065 |          0.5187 |
|                        60 |                       0.498  |          0.5184 |
|                        80 |                       0.4015 |          0.4911 |
