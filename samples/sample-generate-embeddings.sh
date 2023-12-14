curl -X POST http://localhost:5000/generate-embeddings \
     -H "Content-Type: application/json" \
     -d '["tekst1", "tekst2", "tekst3"]'