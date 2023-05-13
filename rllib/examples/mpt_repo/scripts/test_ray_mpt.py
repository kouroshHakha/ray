
from ray.rllib.examples.mpt_repo.mpt.module.configuration_mpt import MPTConfig
import ray

@ray.remote(num_gpus=1)
class Actor:


    def __init__(self) -> None:
        print("In init")
        config = MPTConfig()
        # import sys
        # print(sys.path)

    def foo(self):
        print("In foo")




if __name__ == "__main__":
    ray.init()
    actors = [Actor.remote() for _ in range(16)]
    ray.get([a.foo.remote() for a in actors])


