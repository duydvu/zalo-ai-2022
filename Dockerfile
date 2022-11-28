FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

WORKDIR /code

RUN apt update && \
  apt-get -y install vim && \
  apt -y install python3-pip default-jdk

RUN pip install -r requirements.txt
