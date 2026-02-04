from openai import OpenAI
from config import Config

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=Config.OPENROUTER_API_KEY,
)

completion = client.chat.completions.create(
  extra_body={},
  model="google/gemma-3-27b-it:free",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://media.licdn.com/dms/image/v2/C4D03AQFmkCOW2f7M8A/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1661072970629?e=1772064000&v=beta&t=lfLBwqTeuuIqP81pHOALNyiCHyMAKLR0pKGdPWLtcew"
          }
        }
      ]
    }
  ]
)
print(completion.choices[0].message.content)