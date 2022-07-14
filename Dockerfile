FROM python:3.6-stretch

RUN  apt-get update

WORKDIR /home/logreg

COPY ./src /home/logreg/src
COPY ./volume /home/logreg/volume
COPY ./requirements.txt /home/logreg/requirements.txt
COPY ./params.yaml /home/logreg/params.yaml

RUN pip install -r requirements.txt
RUN chmod +x -R src/

CMD bash