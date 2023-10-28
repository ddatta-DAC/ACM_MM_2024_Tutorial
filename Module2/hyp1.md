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

##   See the effect of different `models`


```
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.010 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.005 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.250 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
python3 textclf.py --model_id distilbert-base-uncased --gradient_accumulation_steps 1  --weight_decay 0.500 --learning_rate 0.0002   --train_batch_size 32  --num_epochs 4
```