from openai import OpenAI
from config import Config

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=Config.OPENROUTER_API_KEY,
)

# First API call with reasoning
response = client.chat.completions.create(
  model="openai/gpt-oss-120b:free",
  messages=[
          {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
          }
        ],
  extra_body={"reasoning": {"enabled": True}}
)

# Extract the assistant message with reasoning_details
response = response.choices[0].message

print(response)

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
  {
    "role": "assistant",
    "content": response.content,
    "reasoning_details": response.reasoning_details  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning from where it left off
response2 = client.chat.completions.create(
  model="openai/gpt-oss-120b:free",
  messages=messages,
  extra_body={"reasoning": {"enabled": True}}
)

print(response2)
