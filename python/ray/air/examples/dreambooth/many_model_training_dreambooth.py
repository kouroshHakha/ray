from typing import List

from accelerate_dreambooth import train_fn
import ray
from ray.util.multiprocessing import Pool
import time

# 1. Create a task that defines a parameterized training job (e.g train a model with a given set of images from s3)
@ray.remote(num_gpus=1)
def train_model(s3_bucket):

    # Get data from s3 and train model
    # This takes 8-12 min to run and in the end the model is stored on S3
    config = {}
    # train_fn(config)
    
    # testing the GPU stuff
    import torch
    import os
    print("cuda_is_available: ", torch.cuda.is_available())
    print("cuda device count: ", torch.cuda.device_count())
    print("cuda visible devices: ", os.environ["CUDA_VISIBLE_DEVICES"])
    
    # 1e5x1e5 = 1e10 = 10GB
    size = (100_000, 100_000)
    foo = torch.ones(size, dtype=torch.int8, device="cuda")
    
    time.sleep(10)
    

class TaskManager:
    
    def __init__(self, max_num_tasks_in_flight: int):
        pass
    
    def submit_task(self, task):
        pass
    
    def is_ready(self) -> bool:
        # will return True if the number of tasks in flight is less than max_num_tasks
        pass
    
    def get_inflight_tasks(self) -> List[ray.ObjectRef]:
        # returns a list of ray remote object refs for those tasks that are in flight
        pass
    
    def get_completed_tasks(self) -> List[ray.ObjectRef]:
        # returns a list of ray remote object refs for those tasks that are completed
        pass
    
    def get_failed_taks(self) -> List[ray.ObjectRef]:
        # returns a list of ray remote object refs for those tasks that have failed
        pass
    
    
def main():
    # Skeleton of what this would look like as a repeated loop
    # + We can submit new similar jobs as we want to scale horizontally (more resources get allocated to this, different clouds etc.). We can increase the redundancy of the service to ensure fault tolerance in case of failure on driver nodes.
    # Note: We need some locking mechanism to ensure competing services do not serve the same request at the same time
    
    pending_requests = []
    task_manager = TaskManager(max_num_tasks_in_flight=10)
    MAX_NUM_RETRIES = 3
    db = None # some wrapper around an external db that stores the requests for this service

    while True: 
        
        # Poll requests that either have been not scheduled from the external 
        # database of requests for this service
        requests = db.get_unscheduled_requests()
        
        # If the task manager is ready to accept more tasks, submit the next one
        # and update their status in the external database to "in progress"
        if task_manager.is_ready():
            r = pending_requests.pop(0)
            task = train_model.remote(r.image_path)
            task_manager.submit_task(task)
            db.update_status(r, "in progress")
            
        
        # In each iteration check if any of the tasks have completed
        # For those that have completed, remove them from the task manager and post status update to the external database
        completed_tasks = task_manager.get_completed_tasks()
        for ctask in completed_tasks:
            task_manager.remove_task(ctask)
            db.update_status(ctask, "completed")
            
        # In each iteration check if any of the tasks have failed
        # For those that have failed, remove them from the task manager and add them to the end of the queue (as if they we resubmitted, do this max N times)
        failed_tasks = task_manager.get_failed_tasks()
        for ftask in failed_tasks:
            task_manager.remove_task(ftask)
            if ftask.num_retries < MAX_NUM_RETRIES:
                db.update_status(ftask, "submitted")
            else:
                db.update_status(ftask, "failed")
        
        
        # sleep for a bit before checking again
        time.sleep(1)
    
        

s3_buckets = ["foo", "bar"] * 20
train_model_futures = [train_model.remote(s3) for s3 in s3_buckets]
results = ray.get(train_model_futures)


    


