FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN apt update \
    && apt install -y tmux \ 
    && apt install -y git \ 
    &&  apt clean

COPY . /world_machine

WORKDIR /world_machine
RUN python -m pip install .

WORKDIR /world_machine/experiments
RUN python -m pip install .

WORKDIR /

RUN rm -rf /world_machine

ENTRYPOINT ["/bin/bash"]