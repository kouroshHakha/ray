


CUDA_VISIBLE_DEVICES=0 python \
    rllib/examples/mpt_repo/scripts/eval_instruct.py \
    --name_or_path /mnt/shared_storage/kourosh/ray_mpt_checkpoints/mpt_instruct \
    --temperature 0.5 \
    --top_p 0.92 \
    --top_k 0 \
    --seed 1 \
    --use_cache True \
    --max_new_tokens 512 \
    --do_sample True \
    --repetition_penalty 1.1 \
    --is_mpt \
    --prompts \
      "How best should I travel from London to Edinburgh, UK?" \
      "Tell me whether these are states or countries: Canada, South Carolina, New York, New Jersey, Japan, Germany, Australia, USA, Georgia, United Kingdom." \
      "Classify the musical genres of the following bands: Metallica, AC/DC, Aerosmith, Madonna" \
      "Which is a bird or fish: Red snapper or Red kite"

    