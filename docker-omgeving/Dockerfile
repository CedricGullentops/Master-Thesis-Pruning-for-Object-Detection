FROM pytorch/pytorch
WORKDIR /usr
RUN apt-get update && \
    apt-get upgrade -y && \
	pip install brambox && \
	pip install matplotlib && \
	pip install git+https://gitlab.com/eavise/lightnet
	
COPY Code /usr/Code

RUN cd Code && \
	python l2prune.py 95.0 30.0 -n Yolo -m hard