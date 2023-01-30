FROM python:3.8.3

RUN apt-get update
RUN pip install mlflow

COPY . /app
WORKDIR /app

RUN pip install -r build/requirements.txt
RUN chmod +x -R src/

CMD ./src/scripts/run_pipeline.sh