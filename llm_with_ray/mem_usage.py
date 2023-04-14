import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pprint import pprint
import time
import gc
import matplotlib.pyplot as plt
import numpy as np


model_base = "databricks/dolly-v1-6b"
batch_size = 32
mode = "inference"
dtype = torch.float16

class DummyContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def get_memory_usage(device):
    mem_alloc = torch.cuda.max_memory_allocated(device)
    # mem_alloc = torch.cuda.memory_allocated(device)
    mem_reserved = torch.cuda.max_memory_reserved(device)
    # mem_reserved = torch.cuda.memory_reserved(device)
    print(mem_alloc, mem_reserved)
    return mem_alloc, mem_reserved

def memory_footprint_gpu(model_base, batch_size, data_type=torch.float32, device=torch.device('cuda'), mode="train", input_text=None):
    assert mode in ("train", "inference")

    model = AutoModelForCausalLM.from_pretrained(model_base)
    model.to(data_type).to(device)

    # Perform a forward pass to estimate activations size
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    if input_text is None:
        input_text = "This is a sample test."
    # this is one batch
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    num_input_tokens = input_ids.shape[1]
    input_ids = input_ids.repeat(batch_size, 1)

    # input_ids = torch.randn(batch_size, 128, dtype=data_type).to(device)
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(128, 1024),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(1024, 1),
    # ).to(data_type).to(device)


    # Calculate the size of the model parameters
    num_parameters = sum(p.numel() for p in model.parameters())
    dtype_size = torch.tensor(1, dtype=data_type).element_size()
    parameter_memory = num_parameters * dtype_size

    context = torch.no_grad() if mode == "inference" else DummyContext()
    with context:

        # Clear any existing CUDA cache
        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.reset_peak_memory_stats()
        # Monitor memory usage before and after running the model
        torch.cuda.synchronize()
        mem_alloc_1, mem_reserved_1 = get_memory_usage(device)
        with torch.cuda.profiler.profile():
            for _ in range(100):
                model(input_ids)
        torch.cuda.synchronize()
        mem_alloc_2, mem_reserved_2 = get_memory_usage(device)

        # Calculate the size of activations
        # activations_memory = (mem_reserved_2 - mem_reserved_1)
        activations_memory = (mem_alloc_2 - mem_alloc_1)# * batch_size
        # activations_memory = (
        #     mem_alloc_2 - mem_alloc_1 + mem_reserved_2 - mem_reserved_1
        # )

    # Calculate the total memory footprint
    total_memory = parameter_memory + activations_memory
    infos = {
        "parameter_memory": parameter_memory, 
        "activation_memory": activations_memory, 
        "num_input_tokens": num_input_tokens,
        "activation_mem_per_token": activations_memory / num_input_tokens
    }
    return num_parameters, total_memory, infos


def main():
    nparams, mem_ftp, infos = memory_footprint_gpu(model_base, batch_size, data_type=dtype, mode=mode)
    print(f"Num parameters: {nparams / 1e9:.2f}B, Mem footprint for {mode} with bsize = {batch_size}: { mem_ftp / 1e9 :.2f} GiB")
    pprint(infos)
    
    # activation_memory = []
    # batch_sizes = [2, 4, 8, 16, 32]
    # for batch_size in batch_sizes:
    #     nparams, mem_ftp, infos = memory_footprint_gpu(model_base, batch_size, data_type=dtype, mode=mode)

    #     activation_memory.append(infos["activation_memory"])

    # plt.scatter(batch_sizes, activation_memory)
    # plt.plot(batch_sizes, np.concatenate([activation_memory[:1], np.array(batch_sizes[1:]) / batch_sizes[0] * activation_memory[0]]), "--")
    # # plt.xscale("log")
    # plt.savefig(f"llm_benchmarks/{model_base}_{mode}_batch.png")

    # # let's change it with number of tokens
    # input_text = "This is "
    # ntokens = []
    # activation_memory = []
    # for i in range(5):
    #     nparams, mem_ftp, infos = memory_footprint_gpu(model_base, batch_size, data_type=dtype, mode="inference", input_text=input_text)
    #     ntokens.append(infos["num_input_tokens"])
    #     activation_memory.append(infos["activation_memory"])
    #     # let's roughly double it 
    #     input_text += input_text

    # plt.scatter(ntokens, activation_memory)
    # plt.plot(ntokens, np.concatenate([activation_memory[:1], np.array(ntokens[1:]) / ntokens[0] * activation_memory[0]]), "--")
    # plt.savefig(f"llm_benchmarks/{model_base}_{mode}_ntokens.png")


if __name__ == "__main__":

    main()