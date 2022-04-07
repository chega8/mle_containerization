FROM python:3.6-stretch

RUN  apt-get update

WORKDIR /home/logreg

COPY ./src /home/logreg/src
COPY ./volume /home/logreg/volume
COPY ./requirements.txt /home/logreg/requirements.txt

RUN pip install -r requirements.txt

CMD bash