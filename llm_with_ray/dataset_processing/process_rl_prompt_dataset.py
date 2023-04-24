import json
from typing import List, Dict
from uuid import uuid4
from datasets import load_dataset
import enum
from pathlib import Path
import tqdm


dataset_output_dir = "/mnt/shared_storage/kourosh/prompt_datasets/rl-prompt-dataset"

class Role(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"

# Load Hugging Face dataset
dataset = load_dataset("Dahoas/rl-prompt-dataset")
data = dataset["test"]  # You can change this to the dataset split you want to process

def process_conversations(prompt: str) -> Dict[str, List[Dict[str, str]]]:
    if "\n\nHuman: " not in prompt:
        raise ValueError(f"Invalid prompt format on {prompt}")

    segments = [prompt]
    role = Role.USER
    conversations = []


    while segments:
        segment = segments.pop(0)
        segment = segment.strip()
        if not segment:
            continue

        if role is Role.USER:
            segments = segment.split("\n\nAssistant:", 1)
            text = segments.pop(0).split("Human:", 1)[-1].strip()
        elif role is Role.ASSISTANT:
            segments = segment.split("\n\nHuman:", 1)
            text = segments.pop(0).strip()

        conversations.append({"role": role.value, "text": text})
        role = Role.ASSISTANT if role is Role.USER else Role.USER

    return conversations

output_data = []

for record in tqdm.tqdm(data):
    prompt = record["prompt"]
    conv_id = str(uuid4())
    conversations = process_conversations(prompt)

    output_data.append(
        {"conv_id": conv_id, "conversations": conversations, "prompt": prompt}
    )

# Save output as a JSON file
Path(dataset_output_dir).mkdir(parents=True, exist_ok=True)

fpath = Path(dataset_output_dir) / "rl-prompt-dataset.json"
with open(fpath, "w") as outfile:
    json.dump(output_data, outfile, indent=2)