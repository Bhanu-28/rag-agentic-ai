from openai import OpenAI
from config import Config

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=Config.OPENROUTER_API_KEY,
)

completion = client.chat.completions.create(

  extra_body={},
  model="meta-llama/llama-3.3-70b-instruct:free",
  messages=[
    {
      "role": "user",
      "content": "dO INTERNET SEARCH AND TELL ME ABOUT THE CURRENT STOCK PRICE OF NVIDIA"
    }
  ]
)
print(completion.choices[0].message.content)