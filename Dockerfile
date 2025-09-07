FROM python:3.11-slim AS parser-stage

WORKDIR /app

COPY parser-requirements.txt .
COPY parser.py .

RUN pip install --no-cache-dir -r parser-requirements.txt

RUN python parser.py

FROM python:3.11-slim 

WORKDIR /app

COPY prod-requirements.txt .
RUN pip install --no-cache-dir -r prod-requirements.txt
RUN python -m nltk.downloader averaged_perceptron_tagger_eng

COPY main.py .
COPY processors/ ./processors/

COPY --from=parser-stage /app/local_fs/ ./local_fs

ENTRYPOINT [ "python" ]

CMD [ "-u", "main.py" ]
