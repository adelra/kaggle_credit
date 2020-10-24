help:
	@echo "run:             To run a Jupyter notebook on your machine."
	@echo "build:           To build the docker image for the environment."
	@echo "run_docker:      To run the Jupyter inside a docker container and expose the port so that you can use it."
run:
	jupyter notebook --Notebook.App.password='', --ip 0.0.0.0 --port 8765
build:
	docker build -t jupyter_notebook .
run_docker:
	docker run --rm -it -p 8765:8765 jupyter_notebook


