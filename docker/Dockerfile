FROM python:3.10-slim

WORKDIR /app

RUN pip install transformers \
    && pip3 install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install sentence-transformers \
    && pip install Flask

COPY main.py /app/

RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sdadas/mmlw-retrieval-roberta-large')"

CMD ["python", "/app/main.py"]
