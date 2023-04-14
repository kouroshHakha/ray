from argparse import ArgumentParser
from typing import Union
import os
import time
import pandas as pd
import datasets
import sys
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from ray.data import ActorPoolStrategy
import functools
import torch
from pathlib import Path
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download

import ray


class Predictor:    

    def __init__(
        self, 
        model_name_or_path: str,
        checkpoint_path: Union[str, os.PathLike],
        hf_home: Union[str, os.PathLike] = None,
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
    
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            padding_side="left",
            truncation_side="left",
            cache_dir=hf_home,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create a model and initialize it with empty weights
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=hf_home)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        no_split_module_classes = None
        if attention_block_name is not None:
            no_split_module_classes = [attention_block_name]

        self.model = load_checkpoint_and_dispatch(
            model,
            checkpoint_path,
            device_map="auto",
            no_split_module_classes=no_split_module_classes,
            dtype=torch.bfloat16,
        )
        self.model.eval()

    def tokenize(self, batch: pd.DataFrame, block_size: int) -> dict:
        return dict(
            self.tokenizer(
                list(batch["prompt"]),
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_tensors="pt"
            )
        )
    

    def __call__(
        self, 
        batch: pd.DataFrame, 
        block_size: int = sys.maxsize,
    ) -> dict:
        
        # TODO (Kourosh): prompt should be half of the max context length for now. The 
        # smarter thing to do is to not generate the same number of tokens for all 
        # elements. But how do we do that in batch mode efficiently?
        block_size = min(block_size, self.tokenizer.model_max_length // 2)

        input_tensors = self.tokenize(batch, block_size)
        print("Tokenization done.")

        if "max_new_tokens" not in self.generate_kwargs:
            self.generate_kwargs["max_new_tokens"] = block_size

        if "do_sample" not in self.generate_kwargs:
            self.generate_kwargs["do_sample"] = True

        if "top_k" not in self.generate_kwargs:
            self.generate_kwargs["top_k"] = 50
        
        if "top_p" not in self.generate_kwargs:
            self.generate_kwargs["top_p"] = 0.95

        output = self.model.generate(
            input_ids=input_tensors["input_ids"].to("cuda"),
            attention_mask=input_tensors["attention_mask"].to("cuda"),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **self.generate_kwargs
        )

        print("output shape = ", output.shape)

        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return pd.DataFrame({
            "prompt": batch["prompt"],
            "response": output_text,
        })

        
def parse_args():

    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--num_gpus_per_actor", type=float, default=1)
    parser.add_argument("--batch_size_per_actor", type=int, default=16)
    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help="Path to the training dataset. "
        "The given datasets should have key for "
        "``prompt``. Accepted formats:"
        "1) a single data path, 2) multiple datasets in the "
        "form: dataset1-path dataset2-path ...",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset to use for evaluation."
    )

    parser.add_argument(
        "--num_partitions",
        type=int,
        default=100,
        help="Number of partitions to split the dataset into (for parallelization)."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs",
        help="Path to save the output."
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate."
    )

    parser.add_argument(
        "--attention_block_name",
        type=str,
        default=None,
        help="Name of the attention block to keep on a single gpu."
    )

    parser.add_argument(
        "--hf_home",
        type=str,
        default="/mnt/shared_storage/kourosh/hf_home",
        help="Path to the cache directory for checkpoints. This should be cluster storage."
    )


    return parser.parse_args()

def main():
    _args = parse_args()

    if len(_args.data_path) == 1:
        hf_dataset = datasets.load_dataset(_args.data_path[0], split=_args.split, cache_dir=_args.hf_home)
        ray_dataset = ray.data.from_huggingface(hf_dataset)
        ray_dataset = ray_dataset.repartition(_args.num_partitions)
    else:
        raise ValueError("Multiple datasets are not supported yet.")

    ray_dataset = ray_dataset.limit(_args.batch_size_per_actor)
    print("The dataset loaded successfully.")

    print("Schema of the dataset: ", ray_dataset.schema())
    print("Number of examples: ", ray_dataset.count())

    # Make sure the required components are downloaded from driver and are put in HF_HOME (which should be on cluster storage)
    # We only donwload the weights and not do model.from_pretrained() because that will 
    # load the model on the driver which can take a long time.
    
    AutoConfig.from_pretrained(_args.model_name_or_path, cache_dir=_args.hf_home)
    AutoTokenizer.from_pretrained(_args.model_name_or_path, cache_dir=_args.hf_home)
    checkpoint_path = hf_hub_download(
        _args.model_name_or_path, 
        "pytorch_model.bin", 
        cache_dir=_args.hf_home
    )
        
    generate_kwargs = {
        "max_new_tokens": _args.max_new_tokens,
    }

    predictor = Predictor(
        **{       
            "model_name_or_path": _args.model_name_or_path,
            "checkpoint_path": checkpoint_path,
            "hf_home": _args.hf_home,
            "attention_block_name": _args.attention_block_name,
            "generate_kwargs": generate_kwargs,
        }
    )
    breakpoint()

    s = time.time()
    output = ray_dataset.map_batches(
        Predictor,
        batch_format="pandas", 
        batch_size=_args.batch_size_per_actor,
        num_gpus=_args.num_gpus_per_actor,
        # TODO (Kourosh): args or kwargs is not working. Fix it.
        # fn_kwargs={"generate_kwargs": generate_kwargs},
        fn_constructor_kwargs={
            "model_name_or_path": _args.model_name_or_path,
            "checkpoint_path": checkpoint_path,
            "hf_home": _args.hf_home,
            "attention_block_name": _args.attention_block_name,
            "generate_kwargs": generate_kwargs,
        },
        compute=ActorPoolStrategy(min_size=1, max_size=8)
    ).fully_executed()
    e = time.time()
    
    print(f"Time taken to generate: {e - s}")
    print(f"Number of tokens generated: {output.count() * _args.max_new_tokens}")
    print(f"Throughput: {output.count() * _args.max_new_tokens / (e - s)} tokens/sec")
    df = output.to_pandas()
    output_path = Path(_args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path / "out.json", orient='records', lines=True)

    


if __name__ == "__main__":
    main()