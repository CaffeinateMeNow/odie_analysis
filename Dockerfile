FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-alpine3.10

RUN pip install requests pandas numpy elasticsearch elasticsearch-dsl tqdm datasketch scattertext ujson pytz nltk python-dotenv
RUN python -m nltk.downloader stopwords

COPY ./app /app
