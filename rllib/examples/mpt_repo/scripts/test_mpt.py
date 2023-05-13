from mpt.module.configuration_mpt import MPTConfig
from mpt.module.modeling_mpt import MPTForCausalLM, MPTModel, MPTBlock
from transformers import AutoTokenizer
from fairscale.nn.checkpoint import checkpoint_wrapper
from torch.utils.checkpoint import checkpoint
from functools import partial

from transformers.models.gptj import GPTJModel

config = MPTConfig()
config.attn_config["attn_impl"] = "torch"

print("Loading model")
model = MPTForCausalLM(config)
print("Done loading model")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

inputs = tokenizer(["hello my name is kourosh? "], return_tensors="pt")

model.gradient_checkpointing_enable()

output = model.forward(**inputs)


breakpoint()