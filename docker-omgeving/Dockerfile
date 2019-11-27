FROM anibali/pytorch:cuda-10.0
WORKDIR /usr	
RUN	apt install git -y && \
	pip3 install torch && \
	pip3 install torchvision && \
	pip3 install matplotlib && \
	pip3 install tqdm && \
	pip3 install visdom && \
	pip3 install scipy && \
	pip3 install tables
	
RUN	pip3 install brambox && \
	pip3 install git+https://gitlab.com/eavise/lightnet@develop
	
COPY Code /usr/Code
COPY eavise-brugge_flir /usr/eavise-brugge_flir
