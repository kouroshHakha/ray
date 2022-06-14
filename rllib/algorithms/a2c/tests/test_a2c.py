import unittest

import ray
import ray.rllib.algorithms.a2c as a2c
from ray.rllib.utils.test_utils import (
    add_gpu_if_necessary,
    check_compute_single_action,
    check_train_results,
    framework_iterator,
)


class TestA2C(unittest.TestCase):
    """Sanity tests for A2C exec impl."""

    def setUp(self):
        ray.init()

    def tearDown(self):
        ray.shutdown()

    def test_a2c_compilation(self):
        """Test whether an A2C can be built with both frameworks."""
        config = a2c.A2CConfig()\
            .rollouts(num_rollout_workers=2, num_envs_per_worker=2)\
            .training(model={
                "fcnet_hiddens": [16],  # keep it simple
            })
        config = add_gpu_if_necessary(config)
        num_iterations = 1

        # Test against all frameworks.
        for _ in framework_iterator(config, with_eager_tracing=True):
            for env in ["CartPole-v0", "Pendulum-v1", "PongDeterministic-v0"]:
                print(f"env={env}")
                algo = config.build(env=env)
                for i in range(num_iterations):
                    results = algo.train()
                    check_train_results(results)
                    print(results)
                check_compute_single_action(algo)
                algo.stop()

    def test_a2c_exec_impl(self):
        config = (
            a2c.A2CConfig()
            .training(model={
                "fcnet_hiddens": [16],  # keep it simple
            })
            .environment(env="CartPole-v0")
            .reporting(min_time_s_per_iteration=0)
        )

        for _ in framework_iterator(config):
            algo = config.build()
            results = algo.train()
            check_train_results(results)
            print(results)
            check_compute_single_action(algo)
            algo.stop()

    def test_a2c_exec_impl_microbatch(self):
        config = (
            a2c.A2CConfig()
            .training(model={
                "fcnet_hiddens": [16],  # keep it simple
            })
            .environment(env="CartPole-v0")
            .reporting(min_time_s_per_iteration=0)
            .training(microbatch_size=10)
        )

        for _ in framework_iterator(config):
            algo = config.build()
            results = algo.train()
            check_train_results(results)
            print(results)
            check_compute_single_action(algo)
            algo.stop()


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
