import torch
import torch.nn.functional as F
import unittest
import xformers.ops as xops
import math
import time

MAX_ITER = 100


SDP_BACKENDS = {
    "math": dict(enable_math=True, enable_flash=False, enable_mem_efficient=False),
    "flash": dict(enable_math=False, enable_flash=True, enable_mem_efficient=False),
    "mem_efficient": dict(enable_math=False, enable_flash=False, enable_mem_efficient=True),
}

def attn_fn(q, k, v, attn_mask=None, scale=None, dropout_p=0., is_causal=False, is_training=False):
    # Converted C++ implementation of pytorch to python. 
    # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/attention.cpp#L639

    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    scale_factor = math.sqrt(1 / math.sqrt(q.size(-1)) if scale is None else scale)

    if is_causal:
        if attn_mask is not None:
            raise ValueError("Explicit attn_mask should not be set when is_causal=True")

        # Replace attn_mask with causal mask; lower triangular elements take part in attention.
        L, S = q.size(-2), k.size(-2)
        attn_mask = torch.ones([L, S], dtype=torch.bool).tril()

    if attn_mask is not None:
        # Convert boolean mask to additive mask; need to invert mask to indicate what to mask *out*.
        new_attn_mask = torch.zeros_like(attn_mask).to(q)
        new_attn_mask = new_attn_mask.masked_fill(~attn_mask, -float('inf'))
        attn_mask = new_attn_mask
    
    query_ = q * scale_factor
    key_ = k * scale_factor
    attn = query_ @ key_.transpose(-2, -1)    
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)

    if dropout_p > 0:
        attn = torch.dropout(attn, dropout_p, train=is_training)

    return attn @ v



class TestAttention(unittest.TestCase):

    def test_attention(self):
        torch.manual_seed(42)
        device = "cuda:0"
        dtype = torch.float16

        NEG_INF = torch.tensor(float("-inf")).to(dtype).to(device)


        B = 16
        N = 1024
        H = 16
        D = 128

        q = torch.rand(B, H, N, D, dtype=dtype, device=device)
        k = torch.rand(B, H, N, D, dtype=dtype, device=device)
        v = torch.rand(B, H, N, D, dtype=dtype, device=device)

        attn_mask = torch.ones((B, H, N, N), dtype=torch.bool, device=device).tril()
        # attn_mask = None

        # warm up step
        attn_fn(q, k, v, attn_mask=attn_mask)

        s_math = time.time()
        for _ in range(MAX_ITER):
            out_math = attn_fn(q, k, v, attn_mask=attn_mask)
        e_math = time.time()

        for _ in range(MAX_ITER):
            s_torch = time.time()
            # with torch.backends.cuda.sdp_kernel(**SDP_BACKENDS["math"]):
            out_torch = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            e_torch = time.time()
            print(f"The torch implementation runs in {(e_torch - s_torch) * 1e6:.3f} microseconds")

        if attn_mask is not None:
            attn_bias = attn_mask.to(q.dtype).masked_fill(~attn_mask, NEG_INF)
        else:
            attn_bias = None

        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)

        # start_event.record()
        s_xformer = time.time()
        for _ in range(MAX_ITER):
            s_xformer = time.time()
            # if attn_mask is not None:
            #     attn_bias = attn_mask.to(q.dtype).masked_fill(~attn_mask, NEG_INF)
            # else:
            #     attn_bias = None
                
            out_xformer_ = xops.memory_efficient_attention(q_, k_, v_, attn_bias=attn_bias)
        e_xformer = time.time()
        # end_event.record()
        # torch.cuda.synchronize()
        # elapsed_time_ms = start_event.elapsed_time(end_event)
        # print(elapsed_time_ms / MAX_ITER * 1e3)

        out_xformer = out_xformer_.transpose(1, 2)

        print(f"The math implementation runs in {(e_math - s_math) / MAX_ITER * 1e6:.3f} microseconds")
        print(f"The torch implementation runs in {(e_torch - s_torch) / MAX_ITER * 1e6:.3f} microseconds")
        # print(f"The xformer implementation runs in {(e_xformer - s_xformer) / MAX_ITER * 1e6:.3f} microseconds.")
        print(f"The xformer implementation runs in {(e_xformer - s_xformer) * 1e6:.3f} microseconds.")


        self.assertTrue(torch.allclose(out_math, out_torch, atol=1e-3))
        self.assertTrue(torch.allclose(out_math, out_xformer, atol=1e-3))

if __name__ == '__main__':
    unittest.main()