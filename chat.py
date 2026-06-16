import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load your environment variables from the .env file
load_dotenv()

# 2. Initialize the modern Gemini client
# It automatically picks up the "GEMINI_API_KEY" environment variable!
client = genai.Client()

# 3. Generate content using the exact model name string
response = client.models.generate_content(
    model="gemini-2.5-flash",  # Upgraded to the faster, current generation Flash model
    contents="What is RAG in AI? Explain in 3 sentences.",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant."
    )
)

# 4. Print the result
print(response.text)