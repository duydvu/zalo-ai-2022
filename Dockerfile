FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

WORKDIR /code

RUN apt update && \
  apt-get -y install libgl1-mesa-glx libglib2.0-0 vim && \
  apt -y install python3-pip

RUN pip install -r requirements.txt