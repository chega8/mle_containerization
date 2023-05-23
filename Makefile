.PHONY: build
build:
	docker-compose build

.PHONY: run
run:
	docker-compose up -d
	docker exec -it airflow airflow db upgrade
	docker exec -ti airflow airflow users create \
		--username admin \
		--password admin \
		--firstname admin \
		--lastname admin \
		--role Admin \
		--email admin@example.com
	docker exec -u root -ti airflow chmod 777 /var/run/docker.sock
	docker exec -d airflow airflow scheduler

.PHONY: stop
stop:
	docker-compose down