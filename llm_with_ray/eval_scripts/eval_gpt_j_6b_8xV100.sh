
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
mkdir -p $OUTPUT

python eval_prompting.py \
    --model_name_or_path eleutherai/gpt-j-6B \
    --num_gpus_per_actor 1 \
    --batch_size_per_actor 1 \
    --max_new_tokens 256 \
    --attention_block_name GPTJBlock \
    --data_path Dahoas/rm-static \
    --split test \
    --num_partitions 2 \
    --output_path $OUTPUT
    