# Cosipy Dockerfile

This repository provides a Dockerfile for building and running Cospiy. Cosipy is [COSI](https://cosi.ssl.berkeley.edu)'s high-level analysis software.

## Overview

This Dockerfile sets up a two-stage Docker build to create an environment for running Cosipy. It uses Debian Bookworm Slim as the base image.

### What is Cosipy?

Cosipy is COSI's high-level analysis software. For more detailed installation and usage instructions, please refer to the main [Cosipy documentation](https://cositools.github.io/cosipy/) and the [Cosipy repository](https://github.com/cositools/cosipy)

### What is Docker?

Docker is a platform that enables developers to automate the deployment of applications inside lightweight, portable containers. Containers encapsulate an application and its dependencies, ensuring consistent behavior across various environments. For more information on Docker and how to install it, visit the [Docker Installation Page](https://docs.docker.com/engine/install/).

### Key Features

- **Two-Stage Build:** Utilizes a two-stage build process to keep the final image size small and optimize security by separating the build and runtime environments.
- **Debian Slim Base:** Uses Debian Bookworm Slim as the base image to maintain a small and stable footprint.
- **Mambaforge Integration:** Employs Mambaforge for efficient Conda package management, contributing to a compact and manageable image size.
- **Complete Runtime Environment:** Provides all necessary tools and dependencies for users to get started with Cosipy seamlessly.

### Repository Structure

- **Dockerfile:** The main Dockerfile for building the Cosipy environment.
- **entrypoint.sh:** The script used to initialize and run Cosipy in the runtime environment.

## Installation and usage : 

### Prepare Your Shared Directory

The shared directory is a folder on your host machine that will be accessible inside the container. This allows for easy file sharing between your local machine and the container environment. By default, we suggest creating the shared directory in your home folder, but you can customize the location if needed.

To create the shared directory, run:

```bash
mkdir $HOME/shared
```

### Build Instructions

To build the Docker image, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/krishnatejavedula/cosipy-docker.git
   cd cosipy-docker
   ```

2. **Build the Docker Image:**

   ```bash
   docker build -t cosipy-container .
   ```

The final Docker image is approximately 2.5 GB in size.


### Usage Instructions

To set up and run the Cosipy container, use the following commands:

1. **Create the container:**

   This command will create the Cosipy container without starting it immediately:

   ```bash
   docker create -it --init \
   -e HOST_USER_ID=$(id -u $USER) \
   -v "$HOME/shared":/shared \
   -p 8888:8888 \
   cosipy-container
   ```

   - The `--init` flag helps with proper signal handling.
   - The `-e HOST_USER_ID=$(id -u $USER)` flag sets the user ID inside the container to match the host user, ensuring correct file permissions.
   - The `-v "$HOME/shared:/shared"` flag mounts the shared directory between the host and container.
   - The `-p 8888:8888` flag maps port 8888 from the container to port 8888 on the host, useful for accessing services like Jupyter notebooks.

2. **Start the container:**

   After creating the container, start it with:

   ```bash
   docker start -i <container_id>
   ```

   Replace `<container_id>` with the actual ID of the created container. You can find the container ID by running:

   ```bash
   docker ps -a
   ```

### Using the Container

After creating your container, you'll see a long string of numbers and letters—this is your **CONTAINER ID**. You can use this ID or the container's **NAME** assigned by Docker to manage it. Follow the instructions below to interact with your container.

1. **Start and Enter the Container**  
   Use the following command to start the container, replacing `CONTAINER_ID_OR_NAME` with your container's ID or name:

   ```bash
   docker start CONTAINER_ID_OR_NAME
   ```

2. **Attach to the Container**  
   Attach to the running container’s shell using:

   ```bash
   docker attach CONTAINER_ID_OR_NAME
   ```

3. **Exiting the Container**  
   To shut down the container, simply type `exit` in the terminal. If you want to leave the container running in the background, press `Ctrl-p + Ctrl-q`.

4. **Reattaching to a Running Container**  
   If the container is running, reattach with:

   ```bash
   docker attach CONTAINER_ID_OR_NAME
   ```

5. **Restarting a Stopped Container**  
   To start a stopped container and immediately attach to it:

   ```bash
   docker start CONTAINER_ID_OR_NAME && docker attach CONTAINER_ID_OR_NAME
   ```

6. **Shared Directory**  
   The shared directory on your host machine, located at `$HOME/shared`, is mounted inside the container at `/shared`. Any work done within `/shared` will automatically be saved to your host machine.

### Bonus: Launching a Second Shell

You can open a second terminal to interact with the same running container. In a separate terminal window, run the following command, replacing `CONTAINER_ID_OR_NAME` with your container’s ID or name:

```bash
docker exec -it CONTAINER_ID_OR_NAME su - cosi
```

This allows multiple terminal sessions within the same container instance.

### The Shared Directory

The shared directory facilitates easy file sharing between your host and the container. Any files in the `$HOME/shared` folder on the host will be available inside the container under `/shared`. Both the host and container can read, write, and modify these files.

Make sure to perform all work in the container inside the `/shared` directory to ensure changes persist and are accessible on the host machine.

### Using Cosipy Tools

The Cosipy tools are set up in the container using the Conda package manager. By default, you will start in the `base` Conda environment, as indicated by the console prompt:

```bash
(base) cosi@72c1c0f435af:~$ 
```

where:
- `(base)` is the active Conda environment.
- `cosi` is the username.
- `72c1c0f435af` is the container ID.

### Activating the Cosipy Environment

To activate the `cosipy` environment, use the following command:

```bash
conda activate cosipy
```

Once activated, your prompt will change to:

```bash
(cosipy) cosi@72c1c0f435af:~
```

To deactivate the `cosipy` environment and return to the `base` environment, use:

```bash
conda deactivate
```

### Jupyter Notebooks

You can run a Jupyter notebook inside the container using the command:

```bash
notebook
```

This is an alias for:

```bash
jupyter notebook --ip 0.0.0.0 --no-browser
```

The notebook will be accessible from your host machine via the browser. If you need to run multiple Jupyter notebook servers, you can map additional ports when creating the container.

## Acknowledgements

This Dockerfile is based on the FermiBottle Dockerfile created by Alex Reustle. The original Dockerfile can be found at [https://github.com/fermi-lat/FermiBottle/blob/master/docker/Dockerfile](https://github.com/fermi-lat/FermiBottle/blob/master/docker/Dockerfile).



