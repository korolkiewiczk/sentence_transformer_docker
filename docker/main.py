from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer("sdadas/mmlw-retrieval-roberta-large")

def encode_text(texts):
    # Prefix each text with "zapytanie: "
    prefixed_texts = ["zapytanie: " + text for text in texts]
    
    # Encode the texts and convert to list
    embeddings = model.encode(prefixed_texts, convert_to_tensor=True, show_progress_bar=False)
    embeddings_list = embeddings.tolist()
    
    return embeddings_list

@app.route('/v1/embeddings', methods=['POST'])
def get_embeddings():
    content = request.json
    text_input = content['input']
    model_name = content.get('model', 'default-model-name')

    # Encode the input text
    embedding_list = encode_text([text_input])[0]

    response = {
        "data": [
            {
                "embedding": embedding_list,
                "index": 0,
                "object": "embedding"
            }
        ],
        "model": model_name,
        "object": "list",
        "usage": {
            "prompt_tokens": len(text_input.split()),
            "total_tokens": len(text_input.split())
        }
    }

    return jsonify(response)

@app.route('/generate-embedding', methods=['GET'])
def generate_embedding():
    query = request.args.get('query')
    
    # Encode the query
    embedding_list = encode_text([query])[0]

    return jsonify(embedding=embedding_list)

@app.route('/generate-embeddings', methods=['POST'])
def generate_multiple_embeddings():
    texts = request.json
    
    # Encode the multiple texts
    embeddings_list = encode_text(texts)

    return jsonify(embeddings=embeddings_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
