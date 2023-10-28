


##  Varying `weight decay`, and observing the effect on model performance


```
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.010 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.005 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.250 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.500 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
```


| Model                   | Epochs | Train Batch Size | Learning Rate | Gradient Accumulation Steps | Weight Decay | Train time (seconds) | Accuracy | F1-score |
|-------------------------|--------|------------------|---------------|-----------------------------|--------------|----------------------|----------|----------|
| distilbert-base-uncased | 4      | 32               | 0.0002        | 1                           | 0.005        | 100.36               | 0.932    | 0.92827  |
| distilbert-base-uncased | 4      | 32               | 0.0002        | 1                           | 0.01         | 101.55               | 0.926    | 0.92178  |
| distilbert-base-uncased | 4      | 32               | 0.0002        | 1                           | 0.25         | 100.85               | 0.916    | 0.91139  |
| distilbert-base-uncased | 4      | 32               | 0.0002        | 1                           | 0.5          | 101.39               | 0.916    | 0.91429  |     


-----

