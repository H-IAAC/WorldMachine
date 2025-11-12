FROM nvcr.io/nvidia/pytorch:25.09-py3

RUN apt update \
    && apt install -y tmux \ 
    && apt install -y git \ 
    && apt install -y graphviz \
    &&  apt clean

RUN --mount=type=bind,source=.,target=/world_machine,rw \
    cd /world_machine \
    && python -m pip install . \
    && cd /world_machine/experiments \
    && python -m pip install . \
    && python -m pip cache purge

WORKDIR /

ENTRYPOINT ["/bin/bash"]