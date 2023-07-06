from typing import List

import time
import requests

URL = ""

def main():
    # Skeleton of what this would look like as a repeated loop
    # + We can submit new similar jobs as we want to scale horizontally (more resources get allocated to this, different clouds etc.). We can increase the redundancy of the service to ensure fault tolerance in case of failure on driver nodes.
    # Note: We need some locking mechanism to ensure competing services do not serve the same request at the same time
    
    pending_requests = []
    MAX_NUM_RETRIES = 3
    db = None # some wrapper around an external db that stores the requests for this service

    while True: 
        
        # Poll requests that either have been not scheduled from the external 
        # database of requests for this service
        requests = db.get_unscheduled_requests()
        
        # If the task manager is ready to accept more tasks, submit the next one
        # and update their status in the external database to "in progress"
        if pending_requests:
            r = pending_requests.pop(0)
            resp = requests.post(
                URL + "/create", 
                json={"image_path": r.image_path}
            )
        
            if resp.ok:
                db.update_status(r, "in progress")
            
        
        tasks = requests.get(URL + "/list")
        for task in tasks:
            # In each iteration check if any of the tasks have completed
            # For those that have completed, post status update to the external database
            status = requests.get(URL + "/status", json={"task_id": task.task_id})
            if status == "completed":
                db.update_status(ctask, "completed")
                
            # In each iteration check if any of the tasks have failed
            # For those that have failed, add them to the end of the queue (as if they we resubmitted, do this max N times)
            if status == "failed":
                if task.num_retries < MAX_NUM_RETRIES:
                    db.update_status(ftask, "submitted")
                else:
                    db.update_status(ftask, "failed")
        
        
        # sleep for a bit before checking again
        time.sleep(1)

    


