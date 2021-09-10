# export CUDA_VISIBLE_DEVICES=0,1,2,3

# 1. train Neutral Expression Generator
# export SENTI="[NEU]"
# export TRAIN_PATH=""
# export VALID_PATH=""

# 2. train Affective Expression Generator
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

