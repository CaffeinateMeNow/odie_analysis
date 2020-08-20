FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install requests pandas numpy elasticsearch elasticsearch-dsl tqdm datasketch scattertext ujson pytz nltk
RUN python -m nltk.downloader stopwords

COPY ./app /app
