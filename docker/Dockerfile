# Cosipy Dockerfile
# Author: Krishna Teja Vedula
# Based on the FermiBottle Dockerfile by Alex Reustle
# Original URL: https://github.com/fermi-lat/FermiBottle/blob/master/docker/Dockerfile
#
# This Dockerfile builds and runs Cosipy using a two-stage build process.

##########################################################################
#                             Stage 1: Build Stage
##########################################################################
# Base: Debian Bookworm Slim
# - Installs development tools, Mambaforge, and creates a Conda environment.
# - Clones the cosipy repository and installs cosipy .

FROM debian:bookworm-slim AS build

# Install necessary dev packages from apt
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    bzip2 \
    wget \
    ca-certificates &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Mambaforge (from conda-forge)
ENV MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download"
RUN wget "${MINIFORGE_URL}/Mambaforge-Linux-$(uname -m).sh" && \
    bash Mambaforge-Linux-$(uname -m).sh -b -p /opt/conda && \
    rm Mambaforge-Linux-$(uname -m).sh

# Set up Conda and Mamba
ENV PATH=/opt/conda/bin:$PATH
RUN conda init bash && conda config --set always_yes yes && \
    conda update conda && \
    conda install mamba -n base -c conda-forge

# Create conda environment and install dependencies
RUN mamba create -n cosipy python=3.10 pip jupyter h5py -c conda-forge && \
    mamba clean -a -y

# Clone the cosipy repository
WORKDIR /opt/cosipy
RUN git clone https://github.com/cositools/cosipy.git .

# Activate the environment and install cosipy
RUN /opt/conda/bin/conda run -n cosipy pip install -e . --verbose

# Cleanup the conda package cache and set appropriate permissions
ENV CONDA_PREFIX=/opt/conda
RUN rm -rf ${CONDA_PREFIX}/pkgs/* && \
    chmod -R g+rwx ${CONDA_PREFIX}

##########################################################################
#                             Stage 2: Runtime Stage
##########################################################################
# Base: Debian Bookworm Slim
# - Installs runtime packages and sets up a non-root user.
# - Copies Conda environment and Cosipy application from the build stage.
# - Configures environment variables, permissions, and entrypoint.

# Use Debian Bookworm Slim as the runtime base
FROM debian:bookworm-slim AS runtime

# Install necessary runtime packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    bzip2 \
    emacs-nox \
    gfortran \
    git \
    libbz2-1.0 \
    libncurses6 \
    libpng16-16 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxft2 \
    libxpm4 \
    libxrender1 \
    libxt6 \
    mesa-utils \
    ncurses-bin \
    openssl \
    perl \
    readline-common \
    sqlite3 \
    sudo \
    tar \
    vim \
    wget \
    x11-apps \
    zlib1g && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add a non-root user and create the wheel group
RUN groupadd wheel && \
    useradd -ms /bin/bash cosi && \
    usermod -aG sudo cosi && \
    usermod -aG wheel cosi && \
    echo '%sudo ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

# Copy the conda installation and cosipy application from the build stage and set ownership
COPY --from=build --chown=root:wheel /opt/conda /opt/conda
COPY --from=build --chown=root:wheel /opt/cosipy /opt/cosipy
COPY entrypoint.sh /opt/cosipy/entrypoint.sh

# make entrypoint.sh executable
RUN chmod +x /opt/cosipy/entrypoint.sh

# Set up the environment
ENV PATH=/opt/conda/bin:$PATH

# Create the /shared directory
RUN mkdir -p /shared

# Add the default volume and command
VOLUME ["/data"]
CMD ["/bin/bash"]

# Switch to the non-root user
USER cosi
WORKDIR /home/cosi

# Initialize Conda for bash
RUN conda init bash

# Set entrypoint
ENTRYPOINT ["/opt/cosipy/entrypoint.sh"]
