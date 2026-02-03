# API Keys Setup Guide

This guide explains how to properly manage API keys in this RAG Agentic AI project.

## 🔐 Best Practices for API Key Management

We use **environment variables** stored in a `.env` file - this is the industry standard for managing API keys and secrets.

### Why This Approach?

- ✅ **Security**: Keys never get committed to version control
- ✅ **Flexibility**: Easy to switch between development/staging/production
- ✅ **Portability**: Works across different environments
- ✅ **Multi-Provider Support**: Can manage keys for multiple AI providers

---

## 📋 Setup Instructions

### Step 1: Install Dependencies

First, install the required package for loading environment variables:

```bash
pip install python-dotenv
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Create Your `.env` File

Copy the example file to create your own `.env`:

```bash
cp .env.example .env
```

### Step 3: Add Your API Keys

Open the `.env` file and replace the placeholder values with your actual API keys:

```bash
# Example - Replace with your actual keys
OPENAI_API_KEY=sk-proj-abc123xyz...
GOOGLE_API_KEY=AIzaSyD...
PINECONE_API_KEY=pcsk_...
```

### Step 4: Use the Configuration in Your Code

Import and use the `Config` class in your Python files:

```python
from config import Config

# Access API keys
openai_key = Config.OPENAI_API_KEY
google_key = Config.GOOGLE_API_KEY

# Validate required keys before running
Config.validate_keys(["OPENAI_API_KEY", "PINECONE_API_KEY"])
```

---

## 🔑 Supported API Providers

The `.env.example` file includes placeholders for:

### AI/LLM Providers
- **OpenAI** (GPT-4, GPT-3.5)
- **Anthropic** (Claude)
- **Google AI** (Gemini)
- **Cohere**
- **Hugging Face**

### Vector Databases
- **Pinecone**
- **Weaviate**
- **Qdrant**

### Monitoring & Tracing
- **LangSmith**

---

## 🛡️ Security Notes

1. **Never commit `.env` to Git** - It's already in `.gitignore`
2. **Don't share your `.env` file** with anyone
3. **Rotate keys regularly** if you suspect they're compromised
4. **Use different keys** for development and production
5. **Check `.env.example` into Git** - It helps other developers

---

## 🚨 Troubleshooting

### "Missing required API keys" error

Make sure you've:
1. Created the `.env` file (not just `.env.example`)
2. Added actual API keys (not placeholder text)
3. Removed any quotes around the key values

### Keys not loading

1. Ensure `.env` is in the project root directory
2. Check that `python-dotenv` is installed
3. Verify the file is named `.env` (not `.env.txt`)

---

## 📝 Example Usage

```python
# example_usage.py
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
llm = OpenAI(api_key=Config.OPENAI_API_KEY, model=Config.OPENAI_MODEL)
response = llm("Hello, world!")
print(response)
```

---

## 🔄 Multiple Environments

You can create environment-specific files:

- `.env` - Default (development)
- `.env.local` - Local overrides
- `.env.production` - Production keys (deploy separately)
- `.env.staging` - Staging environment

All these files are already in `.gitignore` to prevent accidental commits.
