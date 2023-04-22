
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

from huggingface_hub import snapshot_download
import pprint
from pathlib import Path



def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        # choices=[
        #     "decapoda-research/llama-7b-hf",
        #     "databricks/dolly-v1-6b",
        #     "<path>/lmsys-vicuna-7b-delta-v1.1"
        # ]
    )
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16", "int8"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()
    return args



def main():
    args = parse_arguments()
    model_name = args.model_name
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens

    if not Path(model_name).exists():
        print("Downloading the model from huggingface hub ...")
        time_download_s = time.time()
        snapshot_download(model_name)
        time_download_e = time.time()
        print(f"Downloading done in {time_download_e - time_download_s} seconds.")

    time_load_from_pretarined_s = time.time()
    print("Loading the model into GPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        load_in_8bit=args.dtype == "int8",
    )
    time_load_from_pretarined_e = time.time()
    print(f"Time to load the model from pretrained ckpt: {time_load_from_pretarined_e - time_load_from_pretarined_s} seconds.")
    print(f"device map: {model.hf_device_map}")
    
    print("Loading the tokenizer ...")
    time_tokenizer_load_s = time.time()
    if "llama" in args.model_name or "vicuna" in args.model_name:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name,
            truncation_side="left",
            padding_side="left"
        )
        tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            truncation_side="left",
            padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token

    time_tokenizer_load_e = time.time()
    print(f"Tokenizer loaded in {time_tokenizer_load_e - time_tokenizer_load_s} seconds.")

    if "vicuna" in args.model_name:
        # some special processing for vincuna 1.1 to match the fine-tuning process
        from fastchat.conversation import Conversation, SeparatorStyle

        default_conv = Conversation(
            system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
            roles=["USER", "ASSISTANT"],
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        conv = default_conv.copy()
        conv.append_message("USER", "Can you tell me about your understanding of the meaning of life?")
        conv.append_message("ASSISTANT", None)

        prompts = [
            conv.get_prompt() for _ in range(batch_size)
        ]
    elif "dolly" in args.model_name:
        prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        Can you tell me about your understanding of the meaning of life?

        ### Response:
        """
        prompts = [prompt for _ in range(batch_size)]
    else:
        prompts = [
            "Human: Can you tell me about your understanding of the meaning of life?\n\nAI Assistant: " for _ in range(batch_size)
        ]

    prompt_tensors = tokenizer(
        prompts, 
        return_tensors="pt", 
        max_length=model.config.max_position_embeddings, 
        truncation=True, 
        padding=True
    )

    generation_time_s = time.time()
    print("Generating ...")
    output_ids = model.generate(
        input_ids=prompt_tensors["input_ids"].to("cuda"),
        attention_mask=prompt_tensors["attention_mask"].to("cuda"),
        max_new_tokens=max_new_tokens,
        # we set min_length to force the models to generate a fix number of tokens, 
        # otherwise they will reside back to the default stopping criteria of model.
        # generation_config() which may do early stopping.
        min_length=max_new_tokens + prompt_tensors["input_ids"].shape[1],
    )
    generation_time_e = time.time()

    total_generated_tokens = (output_ids.shape[1] - prompt_tensors["input_ids"].shape[1]) * output_ids.shape[0]
    total_gen_time = generation_time_e - generation_time_s
    print("time the generation took: ", total_gen_time)
    print("shape of input: ", prompt_tensors["input_ids"].shape)
    print("shape of output: ", output_ids.shape)
    print("number of generated tokens: ", total_generated_tokens)
    print(f"throughput: {total_generated_tokens / total_gen_time} tokens/second")

    decoded_output_text = tokenizer.batch_decode(output_ids)
    
    print("decoded output text: ", decoded_output_text[0])


if __name__ == "__main__":
    main()