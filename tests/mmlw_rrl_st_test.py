import pytest
import requests
import numpy as np

# URL of your Flask application
BASE_URL = "http://localhost:5000"

@pytest.fixture
def app():
    app.testing = True
    return app

def test_get_embeddings(app):
    data = {"input": "Jak dożyć 100 lat?"}
    response = requests.post(f"{BASE_URL}/v1/embeddings", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "data" in result
    assert "embedding" in result["data"][0]
    assert "model" in result
    assert result["model"] == "default-model-name"
    assert "usage" in result
    assert "prompt_tokens" in result["usage"]
    assert "total_tokens" in result["usage"]

def test_generate_embedding(app):
    query = "Jak dożyć 100 lat?"
    response = requests.get(f"{BASE_URL}/generate-embedding?query={query}")
    assert response.status_code == 200
    result = response.json()
    assert "embedding" in result

def test_generate_multiple_embeddings(app):
    data = ["Jak dożyć 100 lat?", "Inny przykład tekstu", "Jak chciałbym dożyć aż 70 latek życia?"]
    response = requests.post(f"{BASE_URL}/generate-embeddings", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "embeddings" in result
    assert len(result["embeddings"]) == len(data)
    similarity1 = cosine_similarity(result["embeddings"][0], result["embeddings"][1])
    similarity2 = cosine_similarity(result["embeddings"][0], result["embeddings"][2])
    assert similarity1 < 0.62
    assert similarity2 > 0.81

def test_best_answer_selection(app):
    query = "Jak dożyć 100 lat?"
    response = requests.get(f"{BASE_URL}/generate-embedding?query={query}")
    assert response.status_code == 200
    result = response.json()
    query_embedding = result["embedding"]
    
    answers = [
        "Trzeba zdrowo się odżywiać i uprawiać sport.",
        "Trzeba nie dbać o siebie.",
        "Dzisiaj jest ładna pogoda.",
        "W zdrowym ciele zdrowy duch."
    ]
    
    # Calculate similarity scores for each answer
    answer_similarities = []
    for answer in answers:
        data = {"input": answer}
        response = requests.post(f"{BASE_URL}/v1/embeddings", json=data)
        assert response.status_code == 200
        result = response.json()
        answer_embedding = result["data"][0]["embedding"]
        similarity = cosine_similarity(query_embedding, answer_embedding)
        answer_similarities.append(similarity.item())

    # Find the index of the best answer
    best_answer_index = answer_similarities.index(max(answer_similarities))
    expected_best_answer = answers[best_answer_index]

    assert expected_best_answer == "Trzeba zdrowo się odżywiać i uprawiać sport."

def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.

    Args:
    vector_a (numpy.ndarray): A numpy array representing the first vector.
    vector_b (numpy.ndarray): A numpy array representing the second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)
    if magnitude_a == 0 or magnitude_b == 0:
        # Handling the case where one vector is zero
        return 0
    return dot_product / (magnitude_a * magnitude_b)

if __name__ == "__main__":
    pytest.main()
