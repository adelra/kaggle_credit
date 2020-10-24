FROM ubuntu:18.04
RUN apt-get update
RUN apt-get --assume-yes install python3
RUN apt-get --assume-yes install python3-pip
RUN apt-get --assume-yes install vim
COPY . .
RUN pip3 install -r requirements.txt
ENTRYPOINT ["jupyter","notebook", "--NotebookApp.token=''", "--port=8765", "--NotebookApp.password=''", "--ip=0.0.0.0", "--no-browser", "--allow-root"]