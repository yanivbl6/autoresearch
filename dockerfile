FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /workspace

RUN ln -s /data/users/yanivbl /workspace/data

# Install uv globally to /usr/local/bin so every user (incl. yanivbl) finds it.
# Installer lands in /root/.local/bin by default; we copy the binary to /usr/local/bin.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && cp /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv --version

RUN useradd -ms /bin/bash yanivbl \
    && chown -R yanivbl:yanivbl /workspace

USER yanivbl

WORKDIR /workspace/


RUN git clone https://github.com/yanivbl6/autoresearch
RUN cd autoresearch && uv sync

# Run container with: docker run --gpus all -v $(pwd):/workspace/autoresearch ...
# The autoresearch repo is expected to be mounted (or COPY'd) at /workspace/autoresearch.
# Data file (all_data.pkl.gz) should sit at the repo root.
# Fetch instructions saved to getdata.txt for reference.
RUN echo "scp yanivbl@10.41.75.187:/local/users/tsufp/system-sw/SimulationsAndBenchmarks/CoreProfiler/oracle_replay_data_v2/all_data.pkl.gz ." > /workspace/getdata.txt

# Default to a single visible GPU; override at runtime with -e CUDA_VISIBLE_DEVICES=...
ENV CUDA_VISIBLE_DEVICES=0

RUN git config --global user.email "yanivblm6@gmail.com"
RUN git config --global user.name "Yaniv Blumenfeld"

CMD ["/bin/bash", "--login"]