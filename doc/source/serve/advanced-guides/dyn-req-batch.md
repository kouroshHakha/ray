(serve-performance-batching-requests)=
# Dynamic Request Batching

Serve offers a request batching feature that can improve your service throughput without sacrificing latency. This improvement is possible because ML models can utilize efficient vectorized computation to process a batch of requests at a time. Batching is also necessary when your model is expensive to use and you want to maximize the utilization of hardware.

Machine Learning (ML) frameworks such as Tensorflow, PyTorch, and Scikit-Learn support evaluating multiple samples at the same time.
Ray Serve allows you to take advantage of this feature with dynamic request batching.
When a request arrives, Serve puts the request in a queue. This queue buffers the requests to form a batch. The deployment picks up the batch and evaluates it. After the evaluation, Ray Serve
splits up the resulting batch, and returns each response individually.

## Enable batching for your deployment
You can enable batching by using the {mod}`ray.serve.batch` decorator. The following simple example modifies the `Model` class to accept a batch:
```{literalinclude} ../doc_code/batching_guide.py
---
start-after: __single_sample_begin__
end-before: __single_sample_end__
---
```

The batching decorators expect you to make the following changes in your method signature:
- Declare the method as an async method because the decorator batches in asyncio event loop.
- Modify the method to accept a list of its original input types as input. For example, `arg1: int, arg2: str` should be changed to `arg1: List[int], arg2: List[str]`.
- Modify the method to return a list. The length of the return list and the input list must be of equal lengths for the decorator to split the output evenly and return a corresponding response back to its respective request.

```{literalinclude} ../doc_code/batching_guide.py
---
start-after: __batch_begin__
end-before: __batch_end__
emphasize-lines: 11-12
---
```

You can supply 3 optional parameters to the decorators.
- `batch_wait_timeout_s` controls how long Serve should wait for a batch once the first request arrives.
- `max_batch_size` controls the size of the batch.
- `max_concurrent_batches` maximum number of batches that can run concurrently.

Once the first request arrives, the batching decorator waits for a full batch (up to `max_batch_size`) until `batch_wait_timeout_s` is reached. If the timeout is reached, the Serve sends the batch to the model regardless the batch size.

:::{tip}
You can reconfigure your `batch_wait_timeout_s` and `max_batch_size` parameters using the `set_batch_wait_timeout_s` and `set_max_batch_size` methods:

```{literalinclude} ../doc_code/batching_guide.py
---
start-after: __batch_params_update_begin__
end-before: __batch_params_update_end__
---
```

Use these methods in the constructor or the `reconfigure` [method](serve-user-config) to control the `@serve.batch` parameters through your Serve configuration file.
:::

(serve-streaming-batched-requests-guide)=

## Streaming batched requests

Use an async generator to stream the outputs from your batched requests. The following example converts the `StreamingResponder` class to accept a batch.

```{literalinclude} ../doc_code/batching_guide.py
---
start-after: __single_stream_begin__
end-before: __single_stream_end__
---
```

Decorate async generator functions with the {mod}`ray.serve.batch` decorator. Similar to non-streaming methods, the function takes in a `List` of inputs and in each iteration it `yield`s an iterable of outputs with the same length as the input batch size.

```{literalinclude} ../doc_code/batching_guide.py
---
start-after: __batch_stream_begin__
end-before: __batch_stream_end__
---
```

Calling the `serve.batch`-decorated function returns an async generator that you can `await` to receive results.

Some inputs within a batch may generate fewer outputs than others. When a particular input has nothing left to yield, pass a `StopIteration` object into the output iterable. This action terminates the generator that Serve returns when it calls the `serve.batch` function with that input. When `serve.batch`-decorated functions return streaming generators over HTTP, this action allows the end client's connection to terminate once its call is done, instead of waiting until the entire batch is done.

## Tips for fine-tuning batching parameters

`max_batch_size` ideally should be a power of 2 (2, 4, 8, 16, ...) because CPUs and GPUs are both optimized for data of these shapes. Large batch sizes incur a high memory cost as well as latency penalty for the first few requests.

Set `batch_wait_timeout_s` considering the end to end latency SLO (Service Level Objective). For example, if your latency target is 150ms, and the model takes 100ms to evaluate the batch, set the `batch_wait_timeout_s` to a value much lower than 150ms - 100ms = 50ms.

When using batching in a Serve Deployment Graph, the relationship between an upstream node and a downstream node might affect the performance as well. Consider a chain of two models where first model sets `max_batch_size=8` and second model sets `max_batch_size=6`. In this scenario, when the first model finishes a full batch of 8, the second model finishes one batch of 6 and then to fill the next batch, which Serve initially only partially fills with 8 - 6 = 2 requests, leads to incurring latency costs. The batch size of downstream models should ideally be multiples or divisors of the upstream models to ensure the batches work optimally together.

