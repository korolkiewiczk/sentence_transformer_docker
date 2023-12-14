import requests
from typing import List

class TextEmbeddingClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def encode_text(self, text_input, model_name='default-model-name'):
        endpoint = f"{self.base_url}/v1/embeddings"
        data = {"input": text_input, "model": model_name}
        
        response = requests.post(endpoint, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"Error: {response.status_code}")
            return None

    def generate_embedding(self, query):
        endpoint = f"{self.base_url}/generate-embedding?query={query}"
        
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"Error: {response.status_code}")
            return None

    def generate_multiple_embeddings(self, texts):
        endpoint = f"{self.base_url}/generate-embeddings"
        data = texts
        
        response = requests.post(endpoint, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"Error: {response.status_code}")
            return None
    
    def embed_query(self, query) -> List[float]:
        response = self.generate_embedding(query)
        
        if response and 'embedding' in response:
            return response['embedding']
        else:
            print("Error: Unable to retrieve embedding.")
            return []
