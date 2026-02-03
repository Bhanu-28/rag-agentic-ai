from config import Config
from langchain.llms import OpenAI

# Validate required keys
try:
    Config.validate_keys(["OPENAI_API_KEY"])
    print("✓ Configuration validated")
except ValueError as e:
    print(f"✗ Error: {e}")
    exit(1)

# Use the API key
# llm = OpenAI(organization="org-1nCOm3vMW1YRbJ9SO6CKMeRQ",api_key=Config.OPENAI_API_KEY)
llm = OpenAI(api_key=Config.OPENAI_API_KEY)
response = llm("Hello, world!")
print(response) 