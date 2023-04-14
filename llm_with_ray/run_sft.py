

import ray.data

import datasets



# what if we want to mix in multiple-datasets?
dataset = datasets.load_dataset("Dahoas/rm-static")
hf_dataset = ray.data.from_huggingface(dataset)
breakpoint()




