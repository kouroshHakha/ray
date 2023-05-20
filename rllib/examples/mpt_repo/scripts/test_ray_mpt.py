
from ray.rllib.examples.mpt_repo.mpt.module.configuration_mpt import MPTConfig
from ray.rllib.examples.mpt_repo.mpt.module.modeling_mpt import MPTForCausalLM, MPTModel, MPTForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import ray
import torch

import torch.nn.functional as F
import torch.nn as nn

import xformers.ops as xops
import time
import os

import torch.utils.benchmark as benchmark
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.deterministic = True

CHECKPOINT_PATH = "/mnt/shared_storage/kourosh/hf_home/hub/models--mosaicml--mpt-7b/snapshots/d8304854d4877849c3c0a78f3469512a84419e84/"


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    # f(*args, **kwargs)
    # t0 = benchmark.Timer(
    #     stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    # )
    # return t0.blocked_autorange().mean * 1e6
    # warm up step
    f(*args, **kwargs)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) * 1e3



# @ray.remote(num_gpus=1)
class Actor:


    def __init__(self) -> None:
        print("In init")

    def foo(self):
        print("In foo")
        
        config = MPTConfig.from_pretrained("mosaicml/mpt-7b")
        # config.attn_config["attn_impl"] = "torch_anyscale"
        config.attn_config["attn_impl"] = "torch"
        config.attn_config["alibi"] = False

        print("Loading model")
        model = MPTForCausalLM.from_pretrained(
            "mosaicml/mpt-7b",
            config=config,    
        ).half().cuda()

        print("Done loading model")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(["hello my name is kourosh? "], return_tensors="pt", max_length=16, padding="max_length", truncation=True)
        print(f"inputs_shape = {inputs['input_ids'].shape}")
        # move inputs to gpu
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()


        def fn():
            for p in model.parameters():
                p.grad = None
            with torch.no_grad():
                output = model.forward(**inputs)
            print(f"Loss = {output.loss}")

        print(f"The {config.attn_config['attn_impl']} implementation runs in {benchmark_torch_function_in_microseconds(fn):.3f} microseconds")



        # return output


if __name__ == "__main__":
    # ray.init()
    # actors = [Actor.remote() for _ in range(1)]
    # out = ray.get([a.foo.remote() for a in actors])
    out = Actor().foo()









# # Lets define the hyper-parameters of our input
# batch_size = 16
# max_sequence_len = 1024
# num_heads = 16
# embed_dimension = 128

# dtype = torch.float16
# device = torch.device("cuda")

# query = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
# key = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
# value = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)

# attn_mask = torch.ones((batch_size, num_heads, max_sequence_len, max_sequence_len), device=device, dtype=torch.bool)

# attn_mask = torch.zeros_like(attn_mask).masked_fill(~attn_mask, float("-inf"))

# print("q.shape = ", query.shape)

# print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

# # Lets explore the speed of each of the 3 implementations
# from torch.backends.cuda import sdp_kernel, SDPBackend

# # Helpful arguments mapper
# backend_map = {
#     SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
#     SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
#     SDPBackend.EFFICIENT_ATTENTION: {
#         "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
# }

# # with sdp_kernel(**backend_map[SDPBackend.MATH]):
# #     print(f"The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value, attn_mask=attn_mask):.3f} microseconds")


# # with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
# #     try:
# #         print(f"The flash attention implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value, attn_mask=attn_mask):.3f} microseconds")
# #     except RuntimeError:
# #         print("FlashAttention is not supported. See warnings for reasons.")

# # with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
# #     try:
# #         print(f"The memory efficient implementation runs in {benchmark_torch_function_in_microseconds(xops.memory_efficient_attention, query, key, value, attn_bias=attn_mask):.3f} microseconds")
# #     except RuntimeError:
# #         print("EfficientAttention is not supported. See warnings for reasons.")


# import torch
# import xformers.ops as xops

# torch.manual_seed(42)
# device = "cuda:0"
# dtype = torch.float16


# B = 16
# N = 16
# H = 16
# D = 128

# q = torch.rand(B, H, N, D, dtype=dtype, device=device)
# k = torch.rand(B, H, N, D, dtype=dtype, device=device)
# v = torch.rand(B, H, N, D, dtype=dtype, device=device)



# attn_mask = torch.ones((B, H, N, N), dtype=torch.bool, device=device).triu(1)
# attn_bias = attn_mask.to(dtype).masked_fill(attn_mask, float("-inf"))


# print(f"The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, q, k, v, attn_mask=attn_mask):.3f} microseconds")

# # xformers attention expects shape B, N, H, D instead of B, H, N, D
# q = q.transpose(1, 2)
# k = k.transpose(1, 2)
# v = v.transpose(1, 2)

# print(f"The xformer implementation runs in {benchmark_torch_function_in_microseconds(xops.memory_efficient_attention, q, k, v, attn_bias=attn_bias):.3f} microseconds")

# out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)

# out = out.transpose(1, 2)
# breakpoint()

###############################################################################

# seed=42
# torch.manual_seed(seed)
# device = "cuda:0"
# dtype = torch.float16

# B = 4
# N = 512
# H = 8
# D = 128

# q = torch.rand(B, N, H * D, dtype=dtype, device=device)
# k = torch.rand(B, N, H * D, dtype=dtype, device=device)
# v = torch.rand(B, N, H * D, dtype=dtype, device=device)
# x = torch.rand(B, N, H * D, dtype=dtype, device=device)

# # key_padding_mask = torch.ones((B, N), dtype=torch.bool, device=device)
# key_padding_mask = (torch.rand((B, N)) > 0.5).to(dtype=torch.bool, device=device)


# from ray.rllib.examples.mpt_repo.mpt.module.attention import scaled_multihead_dot_product_attention, xformer_multihead_dot_product_attention, MultiheadAttention

###############################################################################


# t_math = benchmark_torch_function_in_microseconds(scaled_multihead_dot_product_attention, q, k, v, H, key_padding_mask=key_padding_mask)

# t_xformer = benchmark_torch_function_in_microseconds(xformer_multihead_dot_product_attention, q, k, v, H, key_padding_mask=key_padding_mask)


# print(f"The math implementation runs in {t_math:.3f} microseconds")
# print(f"The xformer implementation runs in {t_xformer:.3f} microseconds")

# out_math, _ = scaled_multihead_dot_product_attention(q, k, v, H, key_padding_mask=key_padding_mask, is_causal=True)

# out_xformer, _ = xformer_multihead_dot_product_attention(q, k, v, H, key_padding_mask=key_padding_mask, is_causal=True)

# out_math = out_math[~out_xformer.isnan()]
# out_xformer = out_xformer[~out_xformer.isnan()]


# print(torch.allclose(out_math, out_xformer, atol=1e-3))

# breakpoint()

# ###############################################################################

# mha_math = MultiheadAttention(
#     d_model=H * D,
#     n_heads=H,
#     attn_impl="torch",
# ).to(dtype).to(device)

# mha_xformer = MultiheadAttention(
#     d_model=H * D,
#     n_heads=H,
#     attn_impl="torch_anyscale",
# ).to(dtype).to(device)

# mha_xformer.load_state_dict(mha_math.state_dict())


# t_math = benchmark_torch_function_in_microseconds(mha_math, x, attention_mask=key_padding_mask, is_causal=True)
# out_math, _, _ = mha_math(x, attention_mask=key_padding_mask, is_causal=True)
# print(f"The math implementation runs in {t_math:.3f} microseconds")

# t_xformer = benchmark_torch_function_in_microseconds(mha_xformer, x, attention_mask=key_padding_mask, is_causal=True)
# out_xformer, _, _ = mha_xformer(x, attention_mask=key_padding_mask, is_causal=True)
# print(f"The xformer implementation runs in {t_xformer:.3f} microseconds")

# out_math = out_math[~out_xformer.isnan()]
# out_xformer = out_xformer[~out_xformer.isnan()]

# print(torch.allclose(out_math, out_xformer, atol=1e-3))


###############################################################################


# from ray.rllib.examples.mpt_repo.mpt.module.blocks import MPTBlock

# torch.manual_seed(seed)
# mpt_block_math = MPTBlock(
#     d_model=H * D,
#     n_heads=H,
#     expansion_ratio=4,
#     attn_config={
#         'attn_type': 'multihead_attention', 
#         'attn_pdrop': 0.0, 
#         'attn_impl': 'torch', 
#         'qk_ln': False, 
#         'clip_qkv': None, 
#         'softmax_scale': None, 
#         'prefix_lm': False, 
#         'attn_uses_sequence_id': False, 
#         'alibi': False, 
#         'alibi_bias_max': 8
#     },
#     norm_type="layernorm",
# ).to(dtype).to(device)
# mpt_block_math.eval()

# torch.manual_seed(seed)
# mpt_block_xformer = MPTBlock(
#     d_model=H * D,
#     n_heads=H,
#     expansion_ratio=4,
#     attn_config={
#         'attn_type': 'multihead_attention', 
#         'attn_pdrop': 0.0, 
#         'attn_impl': 'torch_anyscale', 
#         'qk_ln': False, 
#         'clip_qkv': None, 
#         'softmax_scale': None, 
#         'prefix_lm': False, 
#         'attn_uses_sequence_id': False, 
#         'alibi': False, 
#         'alibi_bias_max': 8
#     },
#     norm_type="layernorm",
# ).to(dtype).to(device)
# mpt_block_xformer.eval()

# mpt_block_xformer.load_state_dict(mpt_block_math.state_dict())


# with torch.no_grad():
#     t_math = benchmark_torch_function_in_microseconds(mpt_block_math, x, attention_mask=key_padding_mask, is_causal=True)
#     out_math, _ = mpt_block_math(x, attention_mask=key_padding_mask, is_causal=True)
#     print(f"The math implementation runs in {t_math:.3f} microseconds")

#     t_xformer = benchmark_torch_function_in_microseconds(mpt_block_xformer, x, attention_mask=key_padding_mask, is_causal=True)
#     out_xformer, _ = mpt_block_xformer(x, attention_mask=key_padding_mask, is_causal=True)
#     print(f"The xformer implementation runs in {t_xformer:.3f} microseconds")

# out_math = out_math[~out_xformer.isnan()]
# out_xformer = out_xformer[~out_xformer.isnan()]
# print(torch.allclose(out_math, out_xformer, atol=1e-2))


# # inds = torch.where((out_math - out_xformer).abs() > 1e-3)[0]
# # print(out_math[inds])
# # print(out_xformer[inds])

# breakpoint()


# ###############################################################################

# n_layers = 5
# VOCAB_SIZE = 50400

# config_math = MPTConfig(
#     d_model=H * D,
#     n_heads=H,
#     n_layers=n_layers,
#     expansion_ratio=4,
#     max_seq_len=2048,
#     vocab_size=VOCAB_SIZE,
#     attn_config={
#         'attn_type': 'multihead_attention', 
#         'attn_pdrop': 0.0, 
#         'attn_impl': 'torch', 
#         'qk_ln': False, 
#         'clip_qkv': None, 
#         'softmax_scale': None, 
#         'prefix_lm': False, 
#         'attn_uses_sequence_id': False, 
#         'alibi': False, 
#         'alibi_bias_max': 8
#     },
#     init_device=device,
# )

# config_xformer = MPTConfig(
#     d_model=H * D,
#     n_heads=H,
#     n_layers=n_layers,
#     expansion_ratio=4,
#     max_seq_len=2048,
#     vocab_size=VOCAB_SIZE,
#     attn_config={
#         'attn_type': 'multihead_attention', 
#         'attn_pdrop': 0.0, 
#         'attn_impl': 'torch_anyscale', 
#         'qk_ln': False, 
#         'clip_qkv': None, 
#         'softmax_scale': None, 
#         'prefix_lm': False, 
#         'attn_uses_sequence_id': False, 
#         'alibi': False, 
#         'alibi_bias_max': 8
#     },
#     init_device=device,
# )

# model_math = MPTForCausalLM(config_math).to(dtype).to(device)
# model_math.eval()

# model_xformer = MPTForCausalLM(config_xformer).to(dtype).to(device)
# model_xformer.load_state_dict(model_math.state_dict())
# model_xformer.eval()


# # random input_ids and attention mask
# input_ids = torch.randint(0, VOCAB_SIZE, (B, N)).to(device)
# attention_mask = torch.ones((B, N)).to(device)


# t_xformer = benchmark_torch_function_in_microseconds(model_xformer, input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
# out_xformer = model_xformer(input_ids, attention_mask=attention_mask, labels=input_ids)
# logits_xformer = out_xformer["logits"]
# loss_xformer = out_xformer["loss"]
# print(f"The xformer implementation runs in {t_xformer:.3f} microseconds")  


# t_math = benchmark_torch_function_in_microseconds(model_math, input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
# out_math = model_math(input_ids, attention_mask=attention_mask, labels=input_ids)
# logits_math = out_math["logits"]
# loss_math = out_math["loss"]
# print(f"The math implementation runs in {t_math:.3f} microseconds")

# print(torch.allclose(loss_math, loss_xformer, atol=1e-1))


# breakpoint()
