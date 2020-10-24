FROM ubuntu:18.04
RUN apt-get update
RUN apt-get --assume-yes install python3
RUN apt-get --assume-yes install python3-pip
RUN apt-get --assume-yes install vim
COPY . .
RUN pip3 install -r requirements.txt
CMD python3 pipeline.py