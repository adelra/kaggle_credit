help:
	@echo "run:             To run the pipeline."
	@echo "build:           To build the docker image."
	@echo "run_docker:      To run the pipeline inside a docker container."
run:
	jupyter notebook --Notebook.App.password='', --ip 0.0.0.0 --port 8889
build:
	docker build -t jupyter_notebook .
run_docker:
	docker run --rm -it jupyter_notebook


