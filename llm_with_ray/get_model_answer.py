from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

from huggingface_hub import snapshot_download
import pprint
from pathlib import Path
import ray
import json
import pandas as pd
from ray.data import ActorPoolStrategy

from transformers import StoppingCriteriaList, StoppingCriteria


class StopOnEncounteringWord(StoppingCriteria):

    def __init__(self, stops, tokenizer):
        super().__init__()
        self.stops = stops
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            input_text = self.tokenizer.decode(input_ids[0])
            if stop in input_text[-len(stop):]:
                return True

        return False
    
class Prompter:

    def __call__(self, records: pd.DataFrame) -> str:
        return records

class VicunaPrompter(Prompter):

    def __init__(self) -> None:
        # some special processing for vincuna 1.1 to match the fine-tuning process
        from fastchat.conversation import Conversation, SeparatorStyle

        self.default_conv = Conversation(
            system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
            roles=["USER", "ASSISTANT"],
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
    

    def __call__(self, records: pd.DataFrame) -> str:
        
        for row in records.iterrows():
            conversation = row["conversation"]
            conv = self.default_conv.copy()
            for turn in conversation:
                if turn["role"] == "user":
                    conv.append_message("USER", turn["text"])
                elif turn["role"] == "assistant":
                    conv.append_message("ASSISTANT", turn["text"])
            conv.append_message("ASSISTANT", None)
            row["prompt"] = conv.get_prompt()
        
        return records

def get_tokenizer(model_name_or_path: str):
    if "llama" in model_name_or_path or "vicuna" in model_name_or_path:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path,
            truncation_side="left",
            padding_side="left"
        )
        tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            truncation_side="left",
            padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


class Predictor:    

    def __init__(
        self, 
        model_name_or_path: str,
        model_dtype: str = "fp16",
        attention_block_name: str = None,
        generate_kwargs: dict = None,
    ) -> None:
        """
        
        Args:
            model_name_or_path: The name of the model to use.
            checkpoint_path: The path to the checkpoint to load. This is pytorch_model.bin path from hf_hub_download.
            hf_home: You can optionally specify the path to a cluster storage directory for huggingface hub downloads.
            attention_block_name: The name of the attention block to use for checkpoint splitting. For example GPTJBlock for GPT-J.
            generate_kwargs: The kwargs to pass to the generate method.
        """
        
        if generate_kwargs is None:
            generate_kwargs = {}
        self.generate_kwargs = generate_kwargs

        if not Path(model_name_or_path).exists():
            print("Downloading the model from huggingface hub ...")
            time_download_s = time.time()
            snapshot_download(model_name_or_path)
            time_download_e = time.time()
            print(f"Downloading done in {time_download_e - time_download_s} seconds.")
    
        # tokenizer
        print("Loading the tokenizer ...")
        time_tokenizer_load_s = time.time()
        self.tokenizer = get_tokenizer(model_name_or_path)
        time_tokenizer_load_e = time.time()
        print(f"Tokenizer loaded in {time_tokenizer_load_e - time_tokenizer_load_s} seconds.")

        no_split_module_classes = None
        if attention_block_name is not None:
            no_split_module_classes = [attention_block_name]

        time_load_from_pretarined_s = time.time()
        print("Loading the model into GPU ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if model_dtype == "fp16" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
            # no_split_module_classes=no_split_module_classes,
            load_in_8bit=model_dtype == "int8",
        )
        # if we add a pad token, we need to resize the embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        time_load_from_pretarined_e = time.time()
        print(f"Time to load the model from pretrained ckpt: {time_load_from_pretarined_e - time_load_from_pretarined_s} seconds.")
        self.model.eval()

        # Creating the prompter
        if "vicuna" in model_name_or_path:
            self.prompter = VicunaPrompter()
        else:
            self.prompter = Prompter()

    def tokenize(self, batch: pd.DataFrame) -> dict:
        return dict(
            self.tokenizer(
                list(batch["prompt"]),
                return_tensors="pt",
                # fix to 1024 max length for now
                max_length=1024,
                truncation=True,
                padding=True,
            )
        )
    

    def __call__(
        self, 
        batch: pd.DataFrame, 
    ) -> dict:
        
        print("Creating Prompt ...")
        batch = self.prompter(batch)
        print("Prompt created.")
        print("Tokenizing ...")
        prompt_tensors = self.tokenize(batch)
        print("Tokenization done.")

        if "max_new_tokens" not in self.generate_kwargs:
            self.generate_kwargs["max_new_tokens"] = 1024 - prompt_tensors["input_ids"].shape[1]

        if "do_sample" not in self.generate_kwargs:
            self.generate_kwargs["do_sample"] = True

        if "top_k" not in self.generate_kwargs:
            self.generate_kwargs["top_k"] = 50
        
        if "top_p" not in self.generate_kwargs:
            self.generate_kwargs["top_p"] = 0.95
    
        generation_time_s = time.time()
        print("Generating ...")
        output_ids = self.model.generate(
            input_ids=prompt_tensors["input_ids"].to("cuda"),
            attention_mask=prompt_tensors["attention_mask"].to("cuda"),
            # eos_token_id=self.tokenizer.eos_token_id,
            # pad_token_id=self.tokenizer.pad_token_id,
            **self.generate_kwargs
        )
        generation_time_e = time.time()

        total_generated_tokens = (output_ids.shape[1] - prompt_tensors["input_ids"].shape[1]) * output_ids.shape[0]
        total_gen_time = generation_time_e - generation_time_s
        print("time the generation took: ", total_gen_time)
        print("total generated tokens: ", total_generated_tokens)
        print("shape of input: ", prompt_tensors["input_ids"].shape)
        print("shape of output: ", output_ids.shape)


        response_ids = output_ids[:, prompt_tensors["input_ids"].shape[1]:]
        decoded_output_text = self.tokenizer.batch_decode(response_ids, skip_special_tokens=False)

        return pd.DataFrame({
            "conv_id": batch["conv_id"],
            "conversations": batch["conversations"],
            "prompt": batch["prompt"],
            "response": decoded_output_text,
            "num_input_tokens": [len(prompt_tensors["input_ids"][i]) for i in range(prompt_tensors["input_ids"].shape[0])],
            "num_generated_tokens": [len(response_ids[i]) for i in range(response_ids.shape[0])],
        })

def parse_arguments():
    import argparse


    # Custom function to convert the string argument to the desired format
    def newline_converter(string):
        return string.replace('\\n', '\n')

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
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
    )
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16", "int8"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus_per_actor", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # Add the stop_word argument with a custom type
    parser.add_argument('--stop_word', type=newline_converter, default='\n\n',
                        help='Specify the stop word as a string, e.g., "\\n\\n" for two newline characters.')

    args = parser.parse_args()
    return args



def main():
    args = parse_arguments()

    with open(args.data_path, "r") as f:
        list_of_convs = json.load(f)
    
    
    df = pd.DataFrame(list_of_convs)
    ray_dataset = ray.data.from_pandas(df)
    ray_dataset = ray_dataset.repartition(128)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
    }

    if args.stop_word:
        # stopping criteria
        tokenizer = get_tokenizer(args.model_name)
        stop_words = [args.stop_word]
        stopping_critera = StoppingCriteriaList([
            StopOnEncounteringWord(stop_words, tokenizer=tokenizer)
        ])
        generate_kwargs["stopping_criteria"] = stopping_critera

    time_batch_inference_s = time.time()
    output = ray_dataset.map_batches(
        Predictor,
        batch_format="pandas", 
        batch_size=1,
        num_gpus=args.num_gpus_per_actor,
        fn_constructor_kwargs={
            "model_name_or_path": args.model_name,
            "generate_kwargs": generate_kwargs,
            "model_dtype": args.dtype,
        },
        compute=ActorPoolStrategy(min_size=1, max_size=8)
    ).fully_executed()
    time_batch_infernence_e = time.time()
    
    print(f"Time to run batch inference: {time_batch_infernence_e - time_batch_inference_s} seconds.")
    odf = output.to_pandas()

    # save to json
    model_name_flat = args.model_name.replace("/", "-")
    fname_stem = Path(args.data_path).stem
    output_path = Path(args.data_path).parent.parent / "model_outputs" / model_name_flat 
    
    output_path.mkdir(parents=True, exist_ok=True)
    odf.to_json(output_path / f"output_{fname_stem}.json", orient="records", lines=True)
    print(f"Saved output to {output_path}")

    # save the statistics
    total_time = time_batch_infernence_e - time_batch_inference_s
    num_gen_tokens = odf["num_generated_tokens"].sum()
    num_input_tokens = odf["num_input_tokens"].sum()
    total_num_tokens = num_gen_tokens + num_input_tokens
    logs_dict = {
        "run_time": float(total_time),
        "num_input_tokens": int(num_input_tokens),
        "num_generated_tokens": int(num_gen_tokens),
        "total_num_tokens": int(total_num_tokens),
        "mean_num_input_tokens": float(odf["num_input_tokens"].mean()),
        "mean_num_generated_tokens": float(odf["num_generated_tokens"].mean()),
        "num_conversations": int(len(odf)),
        "num_tokens_per_sec": float(total_num_tokens / total_time),
        "requests_per_sec": float(len(odf) / total_time),
    }

    pprint.pprint(logs_dict)

    logs_path = Path(args.data_path).parent.parent / "model_outputs" / model_name_flat / f"log_{fname_stem}.txt"

    with open(logs_path, "w") as f:
        for k, v in logs_dict.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
