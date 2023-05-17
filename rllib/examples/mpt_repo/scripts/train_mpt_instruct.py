import argparse

import evaluate
import torch
import datasets
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup, set_seed
import functools

import ray
from accelerate import Accelerator, DistributedType
import tree
import time
import tqdm
import os
from ray import air
from ray.air import session
from pathlib import Path

from torch.profiler import profile, record_function, ProfilerActivity
from accelerate.utils.dataclasses import DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler

from ray.train.huggingface.accelerate import AccelerateTrainer 
from ray.air.integrations.wandb import WandbLoggerCallback

from ray.rllib.examples.mpt_repo.mpt.module.configuration_mpt import MPTConfig
from ray.rllib.examples.mpt_repo.mpt.module.modeling_mpt import MPTForCausalLM, MPTModel, MPTBlock
import numpy as np

EVAL_BATCH_SIZE = 32
# MODEL = "EleutherAI/gpt-j-6b"
MODEL = "mosaicml/mpt-7b"
DATASET = "mosaicml/dolly_hhrlhf"
BLOCK_SIZE = 2048
OVERLAP_LENGTH = 128
OPTIM_BETAS = (0.9, 0.999)
OPTIM_EPS = 1e-8
OPTIM_WEIGHT_DECAY = 0.


class MPTTextDataset(Dataset):

    def __init__(self, split, smoke_test: bool = False) -> None:
        super().__init__()

        self.ds = load_dataset(DATASET)[split]
        if smoke_test:
            self.ds = self.ds.select(np.arange(1000))

    def __getitem__(self, index):
        return self.ds[index]["prompt"] + self.ds[index]["response"]
    
    def __len__(self):
        return len(self.ds)


def collate_fn(batch, tokenizer):
    out_batch = tokenizer(
        batch, 
        padding="longest", 
        max_length=BLOCK_SIZE, 
        truncation=True, 
        return_tensors="pt"    
    )
    out_batch["labels"] = out_batch["input_ids"].clone()
    return out_batch


def get_ds(split, smoke_test: bool = False):
    return MPTTextDataset(split, smoke_test)
    
# New Code #
def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        print(f"Success {status_msg}")
    else:
        print(f"Failure {status_msg}")
    return

def get_dataloaders(tokenizer, batch_size: int = 16, smoke_test: bool = False):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """

    
    train_ds = get_ds(split="train", smoke_test=smoke_test)
    valid_ds = get_ds(split="test", smoke_test=smoke_test)

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        train_ds,
        shuffle=True, 
        batch_size=batch_size, 
        # if drop_last is False, I don't know what the expected behavior is but if the 
        # total number of samples have to be divisible by num_gpus there is no 
        # ambiguity.
        drop_last=True,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer)
    )
    valid_dataloader = DataLoader(
        valid_ds,
        shuffle=False,
        batch_size=EVAL_BATCH_SIZE,
        # drop_last=(accelerator.mixed_precision == "fp8"),
        drop_last=True,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer)
    )

    return train_dataloader, valid_dataloader


def training_function(kwargs: dict):
    print("training_function called")

    config = kwargs["config"]
    args = argparse.Namespace(**kwargs["args"])

    if not args.output_dir:
        raise ValueError("--output_dir must be specified")
    
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

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
        mixed_precision=args.mx,
    )

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token

    set_seed(seed)
    with accelerator.main_process_first():
        print("Loading datasets")
        train_dataloader, valid_dataloader = get_dataloaders(tokenizer, batch_size, smoke_test=args.smoke_test)

    if accelerator.is_main_process:
        print("Datasets created.")
        print("len(train_dataloader): ", len(train_dataloader))
        print("len(valid_dataloader): ", len(valid_dataloader))

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    print("Loading model ...")
    # config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    config = MPTConfig.from_pretrained(MODEL)
    config.attn_config['attn_impl'] = args.attn_impl
    config.attn_config['alibi'] = args.use_alibi

    if accelerator.is_main_process:
        print("Saving tokenizer and config.")
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    # if BLOCK_SIZE > config.max_seq_len:
    #     config.update({"max_seq_len": BLOCK_SIZE})
    # model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # TODO (Kourosh): Load from the pretrained checkpoint
    s = time.time()
    model = MPTForCausalLM.from_pretrained(MODEL, config=config)
    # model = MPTForCausalLM(config=config)
    print(f"Done loading model in {time.time() - s} seconds.")
    

    print("Model initialized with pretrained weights. Training starting...")
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(
        model.parameters(), 
        lr=lr, 
        betas=OPTIM_BETAS, 
        weight_decay=OPTIM_WEIGHT_DECAY, 
        eps=OPTIM_EPS
    )


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

    s = time.time()
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader, lr_scheduler)
    print(f"Prepare done in {time.time() - s} seconds.")

    # Now we train the model
    if accelerator.is_main_process:
        print("Starting training ...")
        print("number of batches on main process", len(train_dataloader))

    avg_fwd_time, avg_bwd_time, avg_opt_step_time = 0, 0, 0
    for epoch in range(num_epochs):
        s_epoch = time.time()
        model.train()
        loss_sum = torch.tensor(0.0).to(accelerator.device)
        for step, batch in tqdm.tqdm(enumerate(train_dataloader)):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch = tree.map_structure(lambda x: x.to(accelerator.device), batch)
            with accelerator.accumulate(model):
                s_fwd = time.time()
                outputs = model(**batch)
                loss = outputs.loss
                loss_sum += loss
                e_fwd = time.time()
                avg_fwd_time += e_fwd - s_fwd
                s_bwd = time.time()
                accelerator.backward(loss)
                e_bwd = time.time()
                avg_bwd_time += e_bwd - s_bwd

                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(model.parameters(), 1.0)

                s_opt_step = time.time()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                e_opt_step = time.time()
                avg_opt_step_time += e_opt_step - s_opt_step
                
            if accelerator.is_main_process:
                accelerator.print(f"[epoch {epoch} step {step}] loss: {loss.item()}")

        e_epoch = time.time()
        accelerator.print("Train time per epoch: ", e_epoch - s_epoch)

        eval_s_epoch = time.time()
        # model.eval()
        # eval_loss_sum = torch.tensor(0.0).to(accelerator.device)
        # for eval_step, batch in tqdm.tqdm(enumerate(valid_dataloader)):
        #     batch = tree.map_structure(lambda x: x.to(accelerator.device), batch)
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #         eval_loss_sum += outputs.loss
        #         if accelerator.is_main_process:
        #             accelerator.print(f"[epoch {epoch} eval_step {eval_step}] loss: {loss.item()}")
        
        eval_e_epoch = time.time()
        accelerator.print("Eval time per epoch: ", eval_e_epoch - eval_s_epoch)

        accelerator.print("avg fwd time: ", avg_fwd_time / (step + 1))
        accelerator.print("avg bwd time: ", avg_bwd_time / (step + 1))
        accelerator.print("avg opt step time: ", avg_opt_step_time / (step + 1))


        session.report(
            {
                "epoch": epoch,
                "train_loss": loss_sum.item() / len(train_dataloader),
                # "eval_loss": eval_loss_sum.item() / len(valid_dataloader),
                "number of iterations": step + 1,
                "Train time per epoch": e_epoch - s_epoch,
                "Eval time per epoch": eval_e_epoch - eval_s_epoch,
                "avg fwd time": avg_fwd_time / (step + 1),
                "avg bwd time": avg_bwd_time / (step + 1),
            },
        )
    
    #     checkpoint_model(args.output_dir, epoch, model, epoch, step + 1)



    # if args.output_dir is not None:
    #     accelerator.print(f"Saving bin version of model in {args.output_dir}")
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(
    #         args.output_dir,
    #         is_main_process=accelerator.is_main_process,
    #         save_function=accelerator.save,
    #         state_dict=accelerator.get_state_dict(model),
    #     )


def print_dataset_stats():

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token


    tloader, vloader = get_dataloaders(tokenizer, batch_size=16)    

    for mode, dloader in zip(["train", "test"], [tloader, vloader]):

        num_samples = 0
        num_tokens = 0
        sample_tokens = []
        for batch in dloader:
            num_samples += batch["input_ids"].shape[0]
            num_tokens += batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
            sample_tokens += list(batch["attention_mask"].sum(-1))

        print("-"*50)
        print(f"[{mode}] num samples: {num_samples}, num tokens: {num_tokens/1e6:.1f}M")
        print(f"Total number of useful tokens: {sum(sample_tokens)/1e6:.1f}M")
        print(f"AVG tokens per sample: {np.mean(sample_tokens)}")
        print(f"STD tokens per sample: {np.std(sample_tokens)}")
        print(f"MIN tokens per sample: {np.min(sample_tokens)}")
        print(f"MAX tokens per sample: {np.max(sample_tokens)}")
        print(f"AVG tokens per batch: {num_tokens / num_samples}")
    
    breakpoint()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mx",
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

    parser.add_argument("--attn-impl", type=str, help="Attention implementation to use. Choose between `torch`, `triton`, or `flash`.", default="torch")
    parser.add_argument("--use-alibi", action="store_true", help="Use AliBi")

    parser.add_argument("--output_dir", type=str, help="Path to output directory.")

    parser.add_argument("--ds_config", type=str, help="Path to deepspeed config file. This should be on shared storage accessible by all nodes.")

    parser.add_argument("--exp-name", type=str, help="Name of experiment to log to wandb.")

    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--smoke-test", action="store_true", help="To run a smoke test on first 1000 samples.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate to use.")
    args = parser.parse_args()
    config = {
        "lr": args.lr, 
        "num_epochs": args.num_epochs, 
        "seed": 42, 
        "batch_size": args.bs, 
        "gradient_accumulation_steps": args.grad_accum,
        "model_name": MODEL,
        "block_size": BLOCK_SIZE,
        "overlap": OVERLAP_LENGTH,
        "eval_batch_size": EVAL_BATCH_SIZE,
    }


    ray.init()
    from ray.train.torch import TorchConfig
    trainer = AccelerateTrainer(
        training_function,
        train_loop_config={"config": config, "args": vars(args)},
        accelerate_config={}, # set to None to grab the default config
        scaling_config=air.ScalingConfig(
            num_workers=args.num_gpus, 
            use_gpu=True,  
        ),
        # torch_config=TorchConfig(timeout_s=60),
        run_config=air.RunConfig(
            callbacks=[
                WandbLoggerCallback(
                    project="test_accelerate_sft",
                    name=args.exp_name,
                    api_key_file="/mnt/shared_storage/kourosh/wandb_key",
                )
            ],
        )
    )

    results = trainer.fit()
    
    breakpoint()



if __name__ == "__main__":
    main()
