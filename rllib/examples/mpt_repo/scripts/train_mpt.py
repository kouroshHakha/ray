import argparse

import evaluate
import torch
import datasets
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup, set_seed

import ray
from accelerate import Accelerator, DistributedType
import tree
import time
import tqdm
import os
from ray import air
from ray.air import session

from torch.profiler import profile, record_function, ProfilerActivity
from accelerate.utils.dataclasses import DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler

from ray.train.huggingface.accelerate import AccelerateTrainer 
from ray.air.integrations.wandb import WandbLoggerCallback
from transformers.models.gptj import GPTJForCausalLM

from mpt.module.configuration_mpt import MPTConfig
from mpt.module.modeling_mpt import MPTForCausalLM, MPTModel, MPTBlock

EVAL_BATCH_SIZE = 32
# MODEL = "EleutherAI/gpt-j-6b"
MODEL = "mosaicml/mpt-7b"
BLOCK_SIZE = 512
OVERLAP_LENGTH = 128


# Custom DictDataset class
class DictDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.data_dict.items()}

    def __len__(self):
        return len(next(iter(self.data_dict.values())))
    
def get_ds(split, tokenizer):
    ds = load_dataset("tiny_shakespeare")[split]
    text = ds["text"][0]
    input_tokens = tokenizer.encode(text, return_tensors="np")[0]


    # Split input tokens into chunks
    num_tokens = len(input_tokens)
    start_idx = 0
    end_idx = BLOCK_SIZE
    token_chunks = []
    while start_idx < num_tokens:
        token_chunk = input_tokens[start_idx:end_idx]
        token_chunks.append(token_chunk)
        start_idx += BLOCK_SIZE - OVERLAP_LENGTH
        end_idx += BLOCK_SIZE - OVERLAP_LENGTH

    # Join token chunks back into strings
    text_chunks = []
    for chunk in token_chunks:
        text_chunks.append(tokenizer.decode(chunk))


    tokenizer_kwargs = {
        "padding": True,
        "max_length": BLOCK_SIZE,
        "truncation": True,
        "return_tensors": "pt"
    }

    output_ds = tokenizer(text_chunks, **tokenizer_kwargs)
    output_ds["labels"] = output_ds["input_ids"].clone()
    return DictDataset(output_ds)
    
def get_dataloaders(accelerator: Accelerator, batch_size: int = 16):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    # tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    with accelerator.main_process_first():
        train_ds = get_ds(split="train", tokenizer=tokenizer)
        valid_ds = get_ds(split="validation", tokenizer=tokenizer)


    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        train_ds,
        shuffle=True, 
        batch_size=batch_size, 
        drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_ds,
        shuffle=False,
        batch_size=EVAL_BATCH_SIZE,
        drop_last=(accelerator.mixed_precision == "fp8"),
    )

    return train_dataloader, valid_dataloader


def training_function(kwargs: dict):
    print("training_function called")

    config = kwargs["config"]
    args = argparse.Namespace(**kwargs["args"])

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    gradient_accumulation_steps = int(config["gradient_accumulation_steps"])

    
    ds_plugin = DeepSpeedPlugin(hf_ds_config=args.ds_config)


    # Initialize accelerator
    accelerator = Accelerator(
        deepspeed_plugin=ds_plugin,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )


    set_seed(seed)
    train_dataloader, valid_dataloader = get_dataloaders(accelerator, batch_size)
    print("Datasets created.")
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)

    print("Loading model")
    # config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    config = MPTConfig()
    config.attn_config['attn_impl'] = 'torch'

    # if BLOCK_SIZE > config.max_seq_len:
    #     config.update({"max_seq_len": BLOCK_SIZE})
    # model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = MPTForCausalLM(config)
    print("Done loading model")
    

    
    print("Model initialized with pretrained weights. Training starting...")

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    # model = model.to(accelerator.device)
    # Instantiate optimizer

    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=lr)


    # Instantiate scheduler

    # Creates Dummy Scheduler if `scheduler` was spcified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps, warmup_num_steps=100
        )


    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.

    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    # Now we train the model
    if accelerator.is_main_process:
        print("Starting training")
        print("number of batches", len(train_dataloader))
    avg_fwd_time, avg_bwd_time, avg_opt_step_time = 0, 0, 0
    s_epoch = time.time()
    for epoch in range(num_epochs):
        model.train()
        for step, batch in tqdm.tqdm(enumerate(train_dataloader)):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            # batch = tree.map_structure(lambda x: x.to(accelerator.device), batch)
            with accelerator.accumulate(model):
                s_fwd = time.time()
                outputs = model(**batch)
                loss = outputs.loss
                e_fwd = time.time()
                avg_fwd_time += e_fwd - s_fwd
                s_bwd = time.time()
                accelerator.backward(loss)
                e_bwd = time.time()
                avg_bwd_time += e_bwd - s_bwd

                s_opt_step = time.time()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                e_opt_step = time.time()
                avg_opt_step_time += e_opt_step - s_opt_step

        print("avg fwd time: ", avg_fwd_time / (step + 1))
        print("avg bwd time: ", avg_bwd_time / (step + 1))
        print("avg opt step time: ", avg_opt_step_time / (step + 1))

    e_epoch = time.time()
    print("time per epoch: ", e_epoch - s_epoch)

    session.report({
        "len of train loader": len(train_dataloader),
        "number of iterations": step + 1,
        "time per epoch": e_epoch - s_epoch,
        "avg fwd time": avg_fwd_time / (step + 1),
        "avg bwd time": avg_bwd_time / (step + 1),
    })
        # model.eval()
        # for step, batch in enumerate(valid_dataloader):
        #     # We could avoid this line since we set the accelerator with `device_placement=True`.

        #     batch = tree.map_structure(lambda x: x.to(accelerator.device), batch)
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #     add_metrics_from_preds(outputs, batch)
        
        # eval_metric = metric.compute()
        # # Use accelerator.print to print only on the main process.
        # accelerator.print(f"epoch {epoch}:", eval_metric)


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUS to use for DDP.")
    parser.add_argument("--bs", type=int, default=16, help="Batch size to use.")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps.")

    parser.add_argument("--grad_ckpt", action="store_true", help="If passed, will use gradient checkpointing.")

    parser.add_argument("--ds_config", type=str, help="Path to deepspeed config file. This should be on shared storage accessible by all nodes.")

    parser.add_argument("--exp-name", type=str, help="Name of experiment to log to wandb.")
    args = parser.parse_args()
    config = {
        "lr": 2e-5, 
        "num_epochs": 1, 
        "seed": 42, 
        "batch_size": args.bs, 
        "gradient_accumulation_steps": args.grad_accum,
        "model_name": MODEL,
        "block_size": BLOCK_SIZE,
        "overlap": OVERLAP_LENGTH,
        "eval_batch_size": EVAL_BATCH_SIZE,
    }


    ray.init()
    trainer = AccelerateTrainer(
        training_function,
        train_loop_config={"config": config, "args": vars(args)},
        accelerate_config={}, # set to None to grab the default config
        scaling_config=air.ScalingConfig(
            num_workers=args.num_gpus, 
            use_gpu=True,  
        ),
        run_config=air.RunConfig(
            callbacks=[
                WandbLoggerCallback(
                    project="test_accelerate_sft",
                    name=args.exp_name,
                    api_key_file="/mnt/shared_storage/kourosh/wandb_key",
                )
            ]
        ),
    )

    results = trainer.fit()



if __name__ == "__main__":
    main()
