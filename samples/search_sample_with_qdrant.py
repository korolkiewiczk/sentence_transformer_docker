from qdrant_client import QdrantClient, models
import requests

from text_embedding_client import TextEmbeddingClient

# data import
url = "https://unknow.news/archiwum.json"
response = requests.get(url)
data = response.json()

collection = []

for info in data[:50]:
    element = {}
    element['Title'] = info['title']
    element['URL'] = info['url']
    element['Info'] = info['info']
    element['Date'] = info['date']
    collection.append(element)

qdrant = QdrantClient(url="localhost", port=6333)
collectionName = "ai_devs_data"
result = qdrant.get_collections()
indexed = next((collection for collection in result.collections if collection.name == collectionName), None)

if indexed == None:
    qdrant.create_collection(collection_name=collectionName, vectors_config=
        models.VectorParams(size=1024, distance=models.Distance.COSINE, on_disk=True)
    )

BASE_URL = "http://localhost:5000"  # Replace with the actual URL of your Docker container
embeddings = TextEmbeddingClient(BASE_URL)

qdrant.upload_records(
    collection_name=collectionName,
    records=[
        models.Record(
            id=idx, vector = embeddings.embed_query(doc['Title'] + " " + doc['Info']), payload=doc
        )
        for idx, doc in enumerate(collection)
    ],
)

query="Kalendarze adwentowe"
qvec = embeddings.embed_query(query)

res = qdrant.search(collection_name=collectionName, query_vector=qvec, limit=1)
print(res[0])
print(res[0].payload['URL'])