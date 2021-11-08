FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
		netbase \
		wget \
       	build-essential \
        	libopencv-dev\
        	libopenblas-dev \
        	nginx
RUN apt-get install -y git

ENV PYTHON_VERSION=3.6
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda install -y scipy pillow gunicorn==19.9.0 && \
    conda install -y gevent flask matplotlib && \
    conda install -y pytorch cudatoolkit=10.2  -c pytorch && \
    conda install -y -c conda-forge opencv

RUN pip install -U git+https://github.com/facebookresearch/fvcore.git
RUN pip install torchvision==0.10.0
	
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
# COPY files for training
COPY ../../../detectron2 /opt/program
COPY ../scripts /opt/program
COPY ../../../pepper/phenotypes_utils.py /opt/program
COPY ../../../pepper/pepper_segment_orient_pred.py /opt/program
COPY ../../../pepper/output/pretrain_weights.pth /opt/program



COPY ./lib /opt/program
COPY ./scripts /opt/program

RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve

WORKDIR /opt/program
# Install detectron2
RUN python -m pip install -e detectron2
#ENTRYPOINT ["python", "train", "--config-file", "configs/centermask/centermask_V_39_eSE_FPN_ms_3x_cuc.yaml"]


