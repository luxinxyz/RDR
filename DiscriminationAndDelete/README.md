# Discrimination And Delete
The source code of Discrimination and Delete Module

Our code is based on the [Huggingface/Transformers](https://github.com/huggingface/transformers)

## 1. Requirements
```
python==3.6.8
pytorch==1.3.0
transformers==2.3.0
```

## 2. Dataset

### Douban Conversation Corpus

Download the [dataset](https://github.com/MarkWuNLP/MultiTurnResponseSelection) and put them into the `data/DoubanConversationCorpus/` directory

### Sentimental Douban Conversation Corpus

We annotate the Douban Conversation Corpus in terms of sentiment polarity, and obtain a new Sentimental Douban Conversation Corpus.

## 3. Run

### train
```
export CUDA_VISIBLE_DEVICES=0
export BERT_PATH="RoBERTa/chinese_roberta_wwm_large_ext_pytorch"
export DATA_DIR="data/"
export OUT_DIR="output/"
export TASK_NAME=douban_dialog

python run.py \
    --model_type bert \
    --model_name_or_path $BERT_PATH \
    --task_name $TASK_NAME \
    --do_train \
    --data_dir $DATA_DIR \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --output_dir $OUT_DIR \
    --evaluate_during_training \
    --save_steps 50 \
    --logging_steps 50 \
    --max_steps -1 \
    --warmup_proportion 0.1 \
    --weight_decay 0.0001 \
    --gradient_accumulation_steps 1 \
    --seed 100
```

### test
```
export CUDA_VISIBLE_DEVICES=0
export BERT_PATH="RoBERTa/chinese_roberta_wwm_large_ext_pytorch"
export DATA_DIR="data/"
export OUT_DIR="output/checkpoint-xxx"
export TASK_NAME=douban_dialog

python run.py \
    --model_type bert \
    --model_name_or_path $BERT_PATH \
    --task_name $TASK_NAME \
    --do_test \
    --data_dir $DATA_DIR \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=64   \
    --output_dir $OUT_DIR
```
