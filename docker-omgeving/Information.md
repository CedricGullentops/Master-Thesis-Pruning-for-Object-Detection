**Docker information:**

Base docker image:
pytorch/pytorch

Build docker container:
(sudo) docker build -t cedricgullentops/master-thesis:latest .

Run docker container:
(sudo) docker run -it cedricgullentops/master-thesis:latest

docker run --rm -it --init \
  --runtime=nvidia \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/app" \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  anibali/pytorch python3 Pruning.py 15.0 ../eavise-brugge_flir/backup/yolov2-640/final.pt -n ../eavise-brugge_flir/cfg/yolov2-640.py -c

Docker image available on dockerhub.

---
**Older version:**

FROM nvidia/cuda:10.1-base-ubuntu18.04
WORKDIR /usr
RUN apt-get update && \
    apt-get upgrade -y
	
RUN	apt install python3-pip -y && \
	apt install git -y && \
	pip3 install torch && \
	pip3 install torchvision && \
	pip3 install matplotlib && \
	pip3 install tqdm && \
	pip3 install visdom && \
	pip3 install tables
	
RUN	pip3 install brambox && \
	pip3 install git+https://gitlab.com/eavise/lightnet@develop
	
COPY Code /usr/Code
COPY eavise-brugge_flir /usr/eavise-brugge_flir

---

**Running example**:



