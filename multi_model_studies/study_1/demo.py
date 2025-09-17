import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Define the sentences to analyze
sentences = [
    'Is outgoing, sociable',
    'Extraverted',
    'Is original, comes up with new ideas'
]


def get_embeddings(texts):
    """Get embeddings for a list of texts using Azure OpenAI"""
    endpoint = "https://allmodelapi3225011299.openai.azure.com/"
    deployment = "text-embedding-3-large"
    api_version = "2023-12-01-preview"
    azure_key = os.getenv("AZURE_API_KEY")
    
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=azure_key,
    )
    
    response = client.embeddings.create(
        input=texts,
        model=deployment
    )
    
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)

def calculate_cosine_similarities(embeddings, texts):
    """Calculate and display cosine similarities between embeddings"""
    n = len(texts)
    
    print("Cosine Similarities:")
    print("-" * 60)
    
    for i in range(n):
        for j in range(i+1, n):
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            print(f"'{texts[i]}' <-> '{texts[j]}':")
            print(f"  Cosine Similarity: {similarity:.4f}")
            print()

def main():
    """Main demo function"""
    print("Sentences to analyze:")
    for i, text in enumerate(sentences, 1):
        print(f"{i}. {text}")
    print()
    
    print("Getting embeddings from Azure OpenAI...")
    embeddings = get_embeddings(sentences)
    print(f"Embeddings shape: {embeddings.shape}")
    print()
    
    calculate_cosine_similarities(embeddings, sentences)

if __name__ == "__main__":
    main()