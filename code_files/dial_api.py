import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

def get_dial_api_key():
    return os.getenv("OPENAI_API_KEY")

def get_azure_client():
    return AzureOpenAI(
        api_key=get_dial_api_key(),
        api_version="2024-02-01",
        azure_endpoint="https://ai-proxy.lab.epam.com"
    )

def dial_chat(messages, model="gpt-4o-mini-2024-07-18", max_tokens=512, temperature=0.0):
    client = get_azure_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        timeout= 10,
    )
    return response.choices[0].message.content

# def dial_embedding(texts, model="text-embedding-3-small-1"):
#     client = get_azure_client()
#     response = client.embeddings.create(
#         model=model,
#         input=texts,
#     )
#     return [r.embedding for r in response.data]
def dial_embedding(texts, model="text-embedding-3-small-1"):
    client = get_azure_client()
    response = client.embeddings.create(
        model=model,
        input=texts,
        timeout=10,
    )
    return [r.embedding for r in response.data]
