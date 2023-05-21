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

TUNER_SYSTEM_PROMPT = """
You are an AI Prompt Engineer for a target LLM. Here's the process to fine-tune prompts for a target LLM:

1. You'll get an initial prompt.
2. You'll get an example where this prompt fails on the target LLM.
3. Your task is to analyze why the prompt failed to generate the correct response for this example.
4. You will then suggest a better revised prompt. 

You should iterate this process until the target response is satisfactory. Follow these guidelines when revising the prompt:

1. If the model's response is incorrect, change your prompt in a way that helps you understand chose its answer by giving it a guideline to give you reasons behind its choice.
2. Reflect on its reasoning and see if you can come up with a better prompt that will help correct the target model's reasoning. 
3. If the model's response is still incorrect after a 1 or 2 attempts, consider providing counter-examples in the prompt that will highlight the failure case. DO NOT use the example or response in your revised prompt.
4. The output should only include your analysis and the revise prompt enclosed in ````. Example:
Analysis: <Analysis>
Revised prompt:
```
<Revised prompt>
```
"""


TUNER_HUMAN_TEMPLATE = """
============== Trial {trial_number} ===============

Prompt:
```
{current_prompt_fmt}
```

Example:
```
{example}
```

Response:
```
{response}
```

Feedback:
```
{feedback}
```
"""

TARGET_SYSTEM_PROMPT = "You are a helpful assistant."
# initial_prompt_fmt = """
# # Task
# Is the following text hate speech?
# # Output format
# Answer Yes or No as labels
# # Examples
# {example}
# """

# initial_prompt_fmt = """
# # Task
# Consider the following text and determine if it constitutes hate speech. Hate speech specifically targets individuals or groups to insult, offend, or threaten them based on attributes like race, religion, ethnic origin, sexual orientation, disability, or gender. When analyzing the text, differentiate between hate speech, strong emotions, critique, and sarcasm by considering the following questions:

# 1. Is the text directly targeting a specific group based on their attributes?
# 2. Is the text intending to insult, offend, or threaten the group?

# If the answer to both questions is yes, then it is hate speech. Otherwise, it is not.
# # Output format
# Answer Yes or No as labels
# # Examples
# {example}
# """

initial_prompt_fmt = """
# Task
Given the definition of hate speech as "a statement or behavior that is intended to insult, offend, or intimidate a person because of some trait (as race, religion, sexual orientation, national origin, or disability)", is the following text hate speech?
# Output format
Answer Yes or No as labels
# Examples
{example}
"""

target_model = ChatOpenAI(model_name="gpt-3.5-turbo")
# tuner_model = ChatOpenAI(model_name="gpt-4", callbacks=[StreamingStdOutCallbackHandler()], streaming=True)
tuner_model = ChatOpenAI(model_name="gpt-4", callbacks=[StreamingStdOutCallbackHandler()], streaming=True, temperature=0.7)
    
failed_output_example = {
    "text":  "Do you know why he is smiling because there is no `excretion law` in New Zealand! The max sentence he will receive from a judge is no more than 27 years in prison! Is this justice? Or because Muslims lives don't matter!??? :(((",
    "response": None,
    "score": None,
    "feedback": None,
}


tuner_messages = [
    SystemMessagePromptTemplate.from_template(TUNER_SYSTEM_PROMPT),
]

target_messages = [
    SystemMessagePromptTemplate.from_template(TARGET_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(initial_prompt_fmt)
]

satisfied = False
current_prompt_fmt = initial_prompt_fmt
turn = 0

while not satisfied:
    # Create the prompt for target model and get its response
    target_model_prompt = ChatPromptTemplate.from_messages(target_messages)
    target_chain = LLMChain(llm=target_model, prompt=target_model_prompt)

    cur_resp = target_chain.run(example=failed_output_example["text"])
    print(f"============== Trial {turn + 1} ===============")
    print("Current Prompt format:")
    print(current_prompt_fmt)
    print("Failed Example:")
    print(failed_output_example["text"])
    print("Response of the target model:")
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
        current_prompt_fmt=current_prompt_fmt.format(example="{example}"),
        example=failed_output_example["text"],
        response=cur_resp,
        # score=score,
        feedback=feedback,
    )
    tuner_messages.append(HumanMessagePromptTemplate.from_template(tuner_human_prompt))

    tuner_prompt = ChatPromptTemplate.from_messages(tuner_messages)
    tuner_chain = LLMChain(llm=tuner_model, prompt=tuner_prompt, verbose=True)
    
    print(tuner_prompt.format(example="{example}"))
    breakpoint()
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
        breakpoint()

    breakpoint()
    turn += 1

