## Effect of varying the `epochs` in training the ML model


| Model                   | Epochs | Train Batch Size | Learning Rate | Weight Decay | Train time (seconds) | Accuracy |
|-------------------------|--------|------------------|---------------|--------------|----------------------|----------|
| distilbert-base-uncased |5|64| 0.0001        | 0.01         |78.19880843162537| 0.932    | 
| distilbert-base-uncased |4|64| 0.0005        | 0.01         |77.11167120933533| 0.838    | 
| distilbert-base-uncased |3|64| 0.0010        | 0.01         |77.36311459541321| 0.518    | 
| distilbert-base-uncased |2|64| 0.0015        | 0.01         |77.06836485862732| 0.482    | 
| distilbert-base-uncased |1|64| 0.0020        | 0.01         |76.92143416404724| 0.482    |
