FROM runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404

WORKDIR /workspace/minicpmo-listening-slack
COPY . .
RUN bash scripts/setup_runpod.sh

CMD ["bash"]
