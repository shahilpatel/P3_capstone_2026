# P3_capstone_2026

## Docker Quick Start

This repository includes a Docker setup for running all notebooks in a consistent environment.

### Prerequisites

- Docker installed and running

### Build the image

Run this from the repository root:

docker build -t p3-capstone-notebooks .

### Start JupyterLab

Run this from the repository root:

docker run --rm -it -p 8888:8888 -v "$(pwd)":/workspace p3-capstone-notebooks

Then open JupyterLab in your browser at:

http://localhost:8888

### Notes

- The container uses [Dockerfile](Dockerfile).
- The local repository is mounted into the container at /workspace.
- Jupyter token/password is disabled in the current Dockerfile for local convenience.