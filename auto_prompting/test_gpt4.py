import openai 

SYSTEM_ID = "system"
USER_ID = "user"

test_input = {
    SYSTEM_ID:  """
You are an AI Prompt Engineer for LLMs. Help me refine my prompt for optimal results. Here's the process:

1. You'll get an initial prompt.
2. You'll get an example where this prompt fails.
3. Your task is to analyze why the prompt failed to generate the correct response for this example.
3. You will then suggest a better revised prompt.

We'll iterate this process until I'm satisfied with the model's responses. Ensure your instructions are succinct. Do not use the example or response in your revised prompt.
    """,
    USER_ID: """
Prompt:

```
# Task
Determine if the following text contains hate speech or not.
# Output format
Answer Yes or No
# Example
{example}
```

Example:

```
# Task
Determine if the following text contains hate speech or not.
# Output format
Answer Yes or No
# Example
Do you know why he is smiling because there is no `excretion law` in New Zealand! The max sentence he will receive from a judge is no more than 27 years in prison! Is this justice? Or because Muslims lives don't matter!??? :(((
```

Response:

```
Yes
```

Feedback:

```
No feedback.
```
"""
}

resp = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[{"role": role, "content": text} for role, text in test_input.items()],
  temperature=0.7
)

print(resp["choices"][0]["message"]["content"])
breakpoint()