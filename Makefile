help:
	@echo "run:             To run the pipeline."
	@echo "build:           To build the docker image."
	@echo "run_docker:      To run the pipeline inside a docker container."
run:
	python3 pipeline.py
build:
	docker build -t arahimi_task1 .
run_docker:
	docker run --rm -it arahimi_task1


