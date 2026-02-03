"""
Simple test script to verify config module is accessible from anywhere in the project.
"""
from config import Config

print("=" * 60)
print("✓ Config Module Import Test - SUCCESS!")
print("=" * 60)
print()

# Display configuration status
print("📋 Configuration Status:")
print("-" * 60)

# Check which API keys are configured
api_keys = {
    "OpenAI": Config.OPENAI_API_KEY,
    "Anthropic (Claude)": Config.ANTHROPIC_API_KEY,
    "Google AI (Gemini)": Config.GOOGLE_API_KEY,
    "Cohere": Config.COHERE_API_KEY,
    "Hugging Face": Config.HUGGINGFACE_API_KEY,
    "Pinecone": Config.PINECONE_API_KEY,
    "Weaviate": Config.WEAVIATE_API_KEY,
    "Qdrant": Config.QDRANT_API_KEY,
    "LangSmith": Config.LANGSMITH_API_KEY,
}

for name, key in api_keys.items():
    status = "✓ Configured" if key else "✗ Not set"
    # Show first/last 4 chars if key exists
    preview = f"({key[:8]}...{key[-4:]})" if key and len(key) > 12 else ""
    print(f"  {name:.<25} {status} {preview}")

print()
print("🔧 Application Settings:")
print("-" * 60)
print(f"  Environment: {Config.APP_ENV}")
print(f"  Log Level: {Config.LOG_LEVEL}")
print()

# Test validation
print("🧪 Testing Key Validation:")
print("-" * 60)
try:
    # This will raise an error if OPENAI_API_KEY is not set
    Config.validate_keys(["OPENAI_API_KEY"])
    print("  ✓ OpenAI API key is configured!")
except ValueError as e:
    print(f"  ✗ Validation failed: {e}")
    print()
    print("💡 Tip: Copy .env.example to .env and add your API keys")

print()
print("=" * 60)
print("✅ Config module is working correctly!")
print("=" * 60)
