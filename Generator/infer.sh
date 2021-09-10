# export CUDA_VISIBLE_DEVICES=0

# Neutral Expression Generator: test
export SENTI="[NEU]"
export MODEL_PATH=""
export TEST_PATH=""
export OUT_PATH=""  

# Affective Expression Generator: test
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
