export CUDA_VISIBLE_DEVICES=0
export BERT_PATH="RoBERTa/chinese_roberta_wwm_large_ext_pytorch"
export DATA_DIR="data/"
export OUT_DIR="output/checkpoint-xxx"
export TASK_NAME=douban_dialog

python run.py \
    --model_type bert \
    --model_name_or_path $BERT_PATH \
    --task_name $TASK_NAME \
    --do_eval \
    --data_dir $DATA_DIR \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=128   \
    --output_dir $OUT_DIR