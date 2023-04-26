"""

Notes:

- You can run this script on ~200 cpus. 200 cpus allows the process of max. 200 tasks at a time, if some tasks hit their exponential backing off strategy the other tasks won't get started until at least some tasks are finished. This helps with the rate limit.

"""
import json
import ray
import openai
import argparse
import re
import time
from pathlib import Path
import pprint
import pandas as pd
from tqdm import tqdm

MAX_API_RETRY = 2
model_name = "gpt-3.5-turbo"
unit_price = 0.002


def get_conversation(conversation):
    conv_str = ""
    for turn in conversation:
        conv_str += f"{turn['role']}: {turn['text']}\n\n"
    conv_str += f"assistant: "
    return conv_str

def get_prompt(prompt, answer1, answer2):

    system_prompt = "As a language model judge, your task is to evaluate two AI responses to a given conversation and give each a score from 1 to 10 in form of ```Response 1 => score: S, Response 2 => score: P```. Look for helpful and factual responses, avoiding irrelevant or hallucinatory information. DO NOT explain your answer, just follow the given format so that I can parse your answer.\n\n"

    user_prompt = f"[Conversation]\n\n{prompt}\n\n[Response 1]\n\n{answer1}\n\n[Response 2]\n\n{answer2}\n\n[Answer]\n\n"

    return system_prompt, user_prompt

def create_comp_table(resp_file_1, resp_file_2):
    with open(resp_file_1, "r") as f:
        resp_1 = [json.loads(line) for line in f]
    
    with open(resp_file_2, "r") as f:
        resp_2 = [json.loads(line) for line in f]

    response_table = {}

    for record in resp_1:
        conv_id = record["conv_id"]
        response_table[conv_id] = {
            "conversation": get_conversation(record["conversations"]),
            "answer1": record["response"],
        }

    not_found = []
    for record in resp_2:
        conv_id = record["conv_id"]
        if conv_id in response_table:
            response_table[conv_id]["answer2"] = record["response"]
        else:
            not_found.append(conv_id)

    assert len(not_found) == 0, f"Conv_id not found: {not_found}"

    limit = len(response_table)
    response_table = {k: v for k, v in list(response_table.items())[:limit]}

    return response_table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resp_file_1", type=str, required=True)
    parser.add_argument("--resp_file_2", type=str, required=True)
    return parser.parse_args()


@ray.remote(num_cpus=1)
def get_answer_with_retry(system_prompt, user_prompt):
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
    )

    # there is a rate limit of 80k tokens per minute, so we should wait at least 60 
    # seconds between retries. More info: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    @retry(wait=wait_exponential(min=60, max=60*20), stop=stop_after_attempt(20))
    def completion_with_backoff(system_prompt, user_prompt):
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
        )

        return resp

    for i in range(MAX_API_RETRY):
        try: 
            resp = completion_with_backoff(system_prompt, user_prompt)
            
            num_tokens = resp["usage"]["total_tokens"]
            gpt_answer = resp["choices"][0]["message"]["content"]
            return (gpt_answer, num_tokens)
        except Exception as e:
            print(e)
            time.sleep(1)
            continue

def extract_scores(text):
    pattern = r'(?i)score\s*:\s*(\d+)'
    scores = re.findall(pattern, text)
    
    if len(scores) == 2:
        return tuple(int(score) for score in scores)
    else:
        return None
    
def progress_bar(obj_refs):
    ready = []
    with tqdm(total=len(obj_refs)) as pbar:
        while len(obj_refs) > 0:
            new_ready, obj_refs = ray.wait(obj_refs, num_returns=1)
            pbar.update(len(new_ready))
            ready.extend(new_ready)
    return ready

def main():
    args = parse_args()
    response_table = create_comp_table(args.resp_file_1, args.resp_file_2)

    handles = []
    inputs = []
    for conv_id, record in response_table.items():
        system_prompt, user_prompt = get_prompt(
            record["conversation"], 
            record["answer1"], 
            record["answer2"]
        )
        inputs.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        handles.append(get_answer_with_retry.remote(system_prompt, user_prompt))

    responses = ray.get(progress_bar(handles))


    for conv_id, resp in zip(response_table, responses):
        if resp is None:
            continue

        gpt_answer, num_tokens = resp

        scores = extract_scores(gpt_answer)
        if scores is None:
            print(f"Could not extract scores from: {gpt_answer}, Skipping...")
            continue
            
        response_table[conv_id]["answer1_score"] = scores[0]
        response_table[conv_id]["answer2_score"] = scores[1]
        response_table[conv_id]["gpt_answer"] = gpt_answer
        response_table[conv_id]["num_tokens"] = num_tokens
        
    df = pd.DataFrame.from_dict(response_table, orient="index")
    avg_tokens = df["num_tokens"].mean()
    total_cost = df["num_tokens"].sum() / 1000 * unit_price
    print("Average number of tokens used:", avg_tokens)
    print("Total API cost: $", total_cost)

    winrate_1_over_2 = (df["answer1_score"] > df["answer2_score"]).mean()
    print("Winrate of answer 1 over answer 2:", winrate_1_over_2)

    id_1 = Path(args.resp_file_1).parent.stem
    id_2 = Path(args.resp_file_2).parent.stem
    fname = f"{id_1}_vs_{id_2}_judge_{model_name}"

    # save the dataset answers and gpt answers
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / f"{fname}.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Winrate of {id_1} over {id_2}: {winrate_1_over_2}\n")
        f.write(f"Total API cost: ${total_cost}\n")
        f.write(f"Average number of tokens used: {avg_tokens}\n")
        f.write(f"Total number of tokens used: {df['num_tokens'].sum()}\n")
        f.write(f"Total Number of failed queries: {df['gpt_answer'].isnull().sum()} / {len(df)}\n")

    df.to_json(output_dir / f"{fname}.json", orient="records", lines=True)
            
if __name__ == "__main__":
    main()