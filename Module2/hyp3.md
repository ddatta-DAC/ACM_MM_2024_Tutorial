## Effect of varying the `epochs` in training the ML model


| Model                    | Epochs | Train Batch Size | Learning Rate | Weight Decay | Train time (seconds) | Accuracy           | F1 score  |
|--------------------------|--------|------------------|---------------|--------------|----------------------|--------------------|-----------|
| distilbert-base-uncased  | 5      | 64               | 0.0001        | 0.01         | 77.58615303039551  | 0.938     |0.936082474226804 |
| distilbert-base-uncased  | 4      | 64               | 0.0001        | 0.01         | 62.806885957717896 | 0.926     |0.9227557411273486|
| distilbert-base-uncased  | 3      | 64               | 0.0001        | 0.01         | 47.86117625236511  | 0.932     |0.9282700421940928 | 
| distilbert-base-uncased  | 2      | 64               | 0.0001        | 0.01         | 33.35843515396118  | 0.91      |0.9087221095334684 |
| distilbert-base-uncased  | 1      | 64               | 0.0001        | 0.01         | 18.711568355560303 | 0.888     |0.8833333333333333 |
