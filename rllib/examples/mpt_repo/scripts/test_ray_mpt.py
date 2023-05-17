
from ray.rllib.examples.mpt_repo.mpt.module.configuration_mpt import MPTConfig
from ray.rllib.examples.mpt_repo.mpt.module.modeling_mpt import MPTForCausalLM
from transformers import AutoTokenizer
import ray
import torch

# @ray.remote(num_gpus=1)
class Actor:


    def __init__(self) -> None:
        print("In init")

    def foo(self):
        print("In foo")
        config = MPTConfig()
        config.attn_config["attn_impl"] = "triton"
        config.attn_config["alibi"] = False

        print("Loading model")
        model = MPTForCausalLM(config).half().cuda()
        print("Done loading model")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        inputs = tokenizer(["hello my name is kourosh? "], return_tensors="pt")
        # move inputs to gpu
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()


        output = model.forward(**inputs)
        output.loss.backward()


        return output




if __name__ == "__main__":
    # ray.init()
    # actors = [Actor.remote() for _ in range(1)]
    # out = ray.get([a.foo.remote() for a in actors])
    out = Actor().foo()
    breakpoint()


