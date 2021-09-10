# Generator
The source code of Neutral Expression Generator and Affective Expression Generator

Our code is adapted from the [CDial-GPT](https://github.com/thu-coai/CDial-GPT)

## 1. Requirements
```
python==3.6.8
pytorch==1.3.0
```

## 2. Dataset

Train a sentiment classifier using manual annotation data **Sentimental Douban Conversation Corpus** to annotate the [Douban Conversation Corpus](https://github.com/MarkWuNLP/MultiTurnResponseSelection) automatically.

## 3. Run

### Train

- Neutral Expression Generator

```
export SENTI="[NEU]"
export TRAIN_PATH=""
export VALID_PATH=""

python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --senti $SENTI \
    --config_file "config.json" \
    --n_emd 768 \
    --n_epochs 10 \
    --train_path $TRAIN_PATH \
    --valid_path $VALID_PATH \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --lr 1e-4 \
    --valid_steps -1 \
    --weight_decay 1e-2 \
    --scheduler "linear_warmup" \
    --warmup_steps 10000 \
    --gradient_accumulation_steps 2
```

- Affective Expression Generator
```
export SENTI="[SENTI-NEU-SEP]"
export TRAIN_PATH=""
export VALID_PATH=""

python -m torch.distributed.launch --nproc_per_node=4 train.py \
    --senti $SENTI \
    --config_file "config.json" \
    --n_emd 768 \
    --n_epochs 10 \
    --train_path $TRAIN_PATH \
    --valid_path $VALID_PATH \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --lr 1e-4 \
    --valid_steps -1 \
    --weight_decay 1e-2 \
    --scheduler "linear_warmup" \
    --warmup_steps 10000 \
    --gradient_accumulation_steps 2
```

### Infer

- Neutral Expression Generator

```
export SENTI="[NEU]"
export MODEL_PATH=""
export TEST_PATH=""
export OUT_PATH=""  

python -u infer.py \
    --model_checkpoint $MODEL_PATH \
    --senti $SENTI \
    --target_senti "[NEU]" \
    --datapath $TEST_PATH \
    --out_path $OUT_PATH \
    --batch_size 256 \
    --max_length 64 \
    --temperature 0.7 \
    --top_p 0.9 
```

- Affective Expression Generator
```
export SENTI="[SENTI-NEU-SEP]"
export MODEL_PATH=""
export TEST_PATH=""
export OUT_PATH=""  

python -u infer.py \
    --model_checkpoint $MODEL_PATH \
    --senti $SENTI \
    --target_senti "[POS]" \
    --datapath $TEST_PATH \
    --out_path $OUT_PATH \
    --batch_size 256 \
    --max_length 64 \
    --temperature 0.7 \
    --top_p 0.9 
```