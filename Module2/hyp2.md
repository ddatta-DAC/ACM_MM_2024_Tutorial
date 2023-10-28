<style>
    .table_style {
        width: 100%;
        text-align: center;
    }
    .table_style th {
        background: grey;
        word-wrap: break-word;
        text-align: center;
    }
    .table_style tr:nth-child(0) { background: rgba(10, 10, 10, 1.0);}
    .table_style tr:nth-child(1) { background: rgba(10, 10, 190, 0.2);}
    .table_style tr:nth-child(2) { background: rgba(200, 70, 10, 0.2); }
    .table_style tr:nth-child(3) { background: rgba(40, 170, 51, 0.2)  }
    .table_style tr:nth-child(4) { background: rgba(190, 10, 250, 0.1); }
</style>

##  Varying `weight decay`, and observing the effect of 


```
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.010 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.005 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.250 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.500 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
```
<div class="table_style">

| Model                 | Epochs | Train Batch Size | Learning Rate | Gradient Accumulation Steps | Weight Decay | Train time (seconds) | Accuracy | F1-score           |
|-----------------------|--------|------------------|---------------|-----------------------------|--------------|----------------------|----------|--------------------|
| distilbert-base-uncased | 4      | 32               | 0.0002        | 1                           | 0.005        | 100.36822247505188   | 0.932    | 0.9282700421940928 |
| distilbert-base-uncased | 4      | 32               | 0.0002        | 1                           | 0.01         | 101.55510210990906   | 0.926    | 0.9217758985200846 |
| distilbert-base-uncased | 4      | 32               | 0.0002        | 1                           | 0.25         | 100.8534562587738    | 0.916    | 0.9113924050632912 |
| distilbert-base-uncased | 4      | 32               | 0.0002        | 1                           | 0.5          | 101.38943672180176   | 0.916    | 0.9142857142857144 |     

</div>
-----

