FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN apt update \
    && apt install tmux

RUN python -m pip install -e .
RUN python -m pip install -e ./experiments

ENTRYPOINT ["/bin/bash"]