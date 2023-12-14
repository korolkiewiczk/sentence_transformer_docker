curl http://localhost:5000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer aaa" \
  -d '{
    "input": "Your text string goes here",
    "model": "custom-model"
  }'