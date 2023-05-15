


CUDA_VISIBLE_DEVICES=0 python \
    rllib/examples/mpt_repo/scripts/eval_instruct.py \
    --name_or_path mosaicml/mpt-7b \
    --temperature 0.5 \
    --top_p 0.92 \
    --top_k 0 \
    --seed 1 \
    --use_cache True \
    --max_new_tokens 512 \
    --do_sample True \
    --repetition_penalty 1.1 \
    --prompts \
      "Teach me some cuss words in Spanish." 
      # "How best should I travel from London to Edinburgh, UK?" \
      # "Write a short story about a robot that has a nice day." \