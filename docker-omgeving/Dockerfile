FROM pytorch/pytorch
WORKDIR /usr
RUN apt-get update && \
    apt-get upgrade -y && \
	pip install brambox && \
	pip install matplotlib && \
	pip install git+https://gitlab.com/eavise/lightnet
	
COPY Code /usr/Code

RUN cd Code && \
	python Pruning.py 30.0
