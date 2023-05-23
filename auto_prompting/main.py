from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain, ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import re
import openai



import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# TUNER_SYSTEM_PROMPT = """
# You are an AI Prompt Engineer for a target LLM. Here's the process to fine-tune prompts for a target LLM:

# 1. You'll get an initial prompt.
# 2. You'll get examples where this prompt failed in some way.
# 3. Your task is to analyze why the prompt failed to generate the correct response for this example.
# 4. You will then suggest a better revised prompt. 

# You should iterate this process until the target responses are satisfactory. Follow these steps when revising the prompt:

# 1. If the model's response is incorrect, understand the reason behind choosing its answer. For doing so you can update the prompt to force the model to generate the reasoning behind its answer.
# 2. Reflect on its reasoning and see if you can come up with a better prompt that will help correct the target model's answers. 
# 3. If the model's responses are still incorrect after a 1 or 2 attempts, consider providing counter-examples in the prompt that will highlight the failure cases. DO NOT use the example or response in your revised prompt.
# 4. The output should only include your analysis and the revise prompt enclosed in ````. Example:
# Analysis: <Analysis>
# Revised prompt:
# ```
# <Revised prompt>
# ```
# """


TUNER_SYSTEM_PROMPT = """
I want you to act as a LLM prompt engineer. I will give you an initial prompt. This prompt will be used on a target LLM to have it do some task. I will also give you some of its failed output response. Your job is to iteratively come up with a better instruction that will nudge the model to behave as expected. 

If feedback is not provided analyze what is wrong first, and generate some hypothesises. Then, add explicit prohibitive conditions to the instruction that will counter the model from repeating the same mistake based on your hypothesises. 

Return the new prompt enclosed in tripple brackets (```). Example:

Revised prompt:
```
<Revised prompt>
"""

TUNER_HUMAN_TEMPLATE = """
============== Trial {trial_number} ===============

Initial Prompt:
```
{current_prompt_fmt}
```

Failed Examples:
```
Input: {example}
Output: {response}
```

Feedback:
```
{feedback}
```
"""

# TARGET_SYSTEM_PROMPT = "You are a helpful assistant."
initial_prompt_fmt = """
I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show.
"""
# initial_prompt_fmt = """
# You are a math assistant. I want you to answer the following questions. Put the final answer in a json format with result key enclosed in triplet ticks.
# """



target_model = ChatOpenAI(model_name="gpt-3.5-turbo")
# tuner_model = ChatOpenAI(model_name="gpt-4", callbacks=[StreamingStdOutCallbackHandler()], streaming=True)
tuner_model = ChatOpenAI(model_name="gpt-4", callbacks=[StreamingStdOutCallbackHandler()], streaming=True, temperature=1.0)
    
failed_output_example = {
    "text": "pwd",
    # "text":  "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "response": None,
    "score": None,
    "feedback": None,
}


tuner_messages = [
    SystemMessage(content=TUNER_SYSTEM_PROMPT),
]



satisfied = False
current_prompt_fmt = initial_prompt_fmt
turn = 0

while not satisfied:
    # Create the prompt for target model and get its response
    # target_messages = [
    #     HumanMessagePromptTemplate.from_template(current_prompt_fmt)
    # ]
    target_messages = [
        SystemMessage(content=current_prompt_fmt),
        HumanMessage(content=failed_output_example["text"])
    ]
    target_model_prompt = ChatPromptTemplate.from_messages(target_messages)
    target_chain = LLMChain(llm=target_model, prompt=target_model_prompt)

    cur_resp = target_chain.run(example=failed_output_example["text"], verbose=True)
    print(f"============== Trial {turn + 1} ===============")
    print("Current Prompt format:")
    print(current_prompt_fmt)
    print("Failed Example:")
    print("Input:")
    print(failed_output_example["text"])
    print("Output:")
    print(cur_resp)

    cur_satisfaction = ""
    while not cur_satisfaction:
        cur_satisfaction = input("Is this response satisfactory? (y/n)\n")
        if cur_satisfaction not in ["y", "n"]:
            print("Please enter y or n\n")
            cur_satisfaction = ""

    if cur_satisfaction == "y":
        print("Great! The model response is satisfactory!")
        exit(0)
    
    # score = ""
    # while not score.isdigit():
    #     score = input("How would you rate this response? (score 1-5)? (Press enter to continue)\n")
    #     if not score.isdigit() and score not in [1,2,3,4,5]:
    #         print("Please enter a number between 1 and 5\n")

    # score = int(score)
    # if score == 5:
    #     print("Great! The model response is satisfactory!")
    #     exit(0)

    feedback = input("[Optional] Specify what is wrong? (Press enter to continue)\n")
    if not feedback:
        feedback = "No feedback provided."

    # Create a decision prompt for gpt-4
    tuner_human_prompt = TUNER_HUMAN_TEMPLATE.format(
        trial_number=turn + 1,
        current_prompt_fmt=current_prompt_fmt,
        example=failed_output_example["text"],
        response=cur_resp,
        feedback=feedback,
    )
    tuner_messages.append(HumanMessage(content=tuner_human_prompt))

    tuner_prompt = ChatPromptTemplate.from_messages(tuner_messages)
    tuner_chain = LLMChain(llm=tuner_model, prompt=tuner_prompt, verbose=True)
    
    # print(tuner_prompt.format(example="{example}"))
    # breakpoint()
    tuner_resp = tuner_chain.run(example="{example}")
    tuner_messages.append(AIMessagePromptTemplate.from_template(tuner_resp))


    # assuming input_string is your string
    snippet_regex = re.compile(r'```(.*?)```', re.DOTALL)

    # match the pattern
    matches = re.findall(snippet_regex, tuner_resp)

    if matches:
        current_prompt_fmt = matches[0]
    else:
        print("Could not find the prompt in the response. Please try again.\n")
        print("Tuner Response: ")
        print(tuner_resp)
        breakpoint()

    breakpoint()
    turn += 1

