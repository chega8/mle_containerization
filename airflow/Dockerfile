FROM apache/airflow:2.5.3
USER root

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         vim \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* 

USER airflow

RUN pip install dvc
# Copy the Airflow configuration file
COPY airflow/airflow.cfg /opt/airflow/airflow.cfg

COPY ./dags /opt/airflow/dags
COPY ./src /opt/airflow/src

# Expose the default Airflow web server port
EXPOSE 8080

ENTRYPOINT ["bash", "-c", "airflow scheduler & airflow webserver"]