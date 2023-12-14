from text_embedding_client import TextEmbeddingClient

if __name__ == "__main__":
    BASE_URL = "http://localhost:5000"  # Replace with the actual URL of your Docker container
    client = TextEmbeddingClient(BASE_URL)
    
    # Example usage of the TextEmbeddingClient class
    text_input = "This is a sample text for encoding."
    query = "What is the meaning of life?"
    texts = ["Text 1", "Text 2", "Text 3"]
    
    # Send a POST request to /v1/embeddings
    encoded_text = client.encode_text(text_input)
    print("Encoded Text:")
    print(encoded_text)
    
    # Send a GET request to /generate-embedding
    embedding = client.generate_embedding(query)
    print("\nGenerated Embedding:")
    print(embedding)
    
    # Send a POST request to /generate-embeddings
    embeddings = client.generate_multiple_embeddings(texts)
    print("\nGenerated Embeddings for Multiple Texts:")
    print(embeddings)
