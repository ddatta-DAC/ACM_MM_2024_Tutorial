##   See the effect of different `models` , with a different set of not carefully chosen hyperparameters.

```
python3 textclf.py --model_id roberta-base            --gradient_accumulation_steps 1  --weight_decay 0.10  --learning_rate 0.05    --train_batch_size 16  --num_epochs 2
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.01 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 10
python3 textclf.py --model_id bert-base-uncased       --gradient_accumulation_steps 1  --weight_decay 0.01 --learning_rate 0.0001   --train_batch_size 16  --num_epochs 2
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 2  --weight_decay 0.05 --learning_rate 0.01     --train_batch_size 64  --num_epochs 5
python3 textclf.py --model_id microsoft/deberta-base  --gradient_accumulation_steps 2  --weight_decay 0.01 --learning_rate 0.00005   --train_batch_size 8  --num_epochs 10
```

-----


| Model                   | Epochs | Train Batch Size | Learning Rate | Gradient Accumulation Steps | Weight Decay | Train time (seconds) | Accuracy | F1-score |
|-------------------------|--------|------------------|---------------|-----------------------------|--------------|----------------------|----------|----------|
| roberta-base            | 2      | 16               | 0.05          | 1                           | 0.1          | 266.00               | 0.518    | 0.0      | 
| distilbert-base-uncased | 10     | 32               | 0.0002        | 1                           | 0.01         | 100.16               | 0.92     | 0.91935  | 
| bert-base-uncased       | 2      | 16               | 0.0001        | 1                           | 0.01         | 251.22               | 0.932    | 0.93061  | 
| distilbert-base-uncased | 5      | 64               | 0.01          | 2                           | 0.05         | 76.42                | 0.482    | 0.65047  | 
| microsoft/deberta-base  | 10     | 8                | 5e-05         | 2                           | 0.01         | 558.60               | 0.966    | 0.96523  | 



----

We vary 
 - `epochs`  between `2` and `10`
 - `training batch size' bwtween `8` and `64`
 - `learning rate` between `0.0001` and `0.01`
 - `weight decay` between `0.05` and `0.1`

----
