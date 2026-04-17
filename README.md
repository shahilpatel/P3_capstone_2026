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

## Docker Usage Guide

Use Docker when you want a reproducible environment for notebook development and batch runs.

### 1) Build or rebuild the image

```bash
docker build -t p3-capstone-notebooks .
```

Rebuild after dependency or [Dockerfile](Dockerfile) changes.

### 2) Run JupyterLab interactively

```bash
docker run --rm -it \
	-p 8888:8888 \
	-v "$(pwd)":/workspace \
	p3-capstone-notebooks
```

Then open http://localhost:8888.

### 3) Open a shell inside the container

```bash
docker run --rm -it \
	-v "$(pwd)":/workspace \
	p3-capstone-notebooks \
	bash
```

This is useful for running commands such as `papermill`, `jupyter nbconvert`, and `tmux` directly.

### 4) Execute a notebook headlessly in Docker

```bash
docker run --rm -it \
	-v "$(pwd)":/workspace \
	p3-capstone-notebooks \
	bash -lc "papermill ballflight_to_outcome/ball_make_characteristics_abhi.ipynb ballflight_to_outcome/ball_make_characteristics_abhi.executed.ipynb --log-output"
```

Outputs are persisted to your host because `/workspace` is mounted from your local repository.

### 5) Use tmux for long runs in Docker

```bash
docker run --rm -it \
	-v "$(pwd)":/workspace \
	p3-capstone-notebooks \
	bash

tmux new -s nb-run
cd /workspace
papermill ballflight_to_outcome/ball_make_characteristics_abhi.ipynb \
	ballflight_to_outcome/ball_make_characteristics_abhi.executed.ipynb \
	--log-output 2>&1 | tee ballflight_to_outcome/papermill_run.log
```

### 6) Common troubleshooting

- Port already in use: map a different host port, for example `-p 8890:8888`.
- Permission issues on mounted files: ensure your host user can write to the repository folder.
- Missing package after edits: rebuild image with `docker build` and rerun.

## Run Notebooks With Papermill

Papermill is recommended for long notebook runs because it provides better live progress and writes an executed notebook to disk.


### Execute a notebook end-to-end

From the repository root:

```bash
papermill ballflight_to_outcome/ball_make_characteristics_abhi.ipynb \
	ballflight_to_outcome/ball_make_characteristics_abhi.executed.ipynb \
	--log-output
```

This will:

- execute all cells in order,
- print cell logs/progress to the terminal,
- persist outputs in `ballflight_to_outcome/ball_make_characteristics_abhi.executed.ipynb`.

### Run in tmux for long jobs

```bash
tmux new -s nb-run
cd /workspace
papermill ballflight_to_outcome/ball_make_characteristics_abhi.ipynb \
	ballflight_to_outcome/ball_make_characteristics_abhi.executed.ipynb \
	--log-output 2>&1 | tee ballflight_to_outcome/papermill_run.log
```

Detach from tmux with `Ctrl+b` then `d`, and reattach with:

```bash
tmux attach -t nb-run
```

### Error behavior

- By default, papermill stops at the first cell error and returns a non-zero exit code.
- The log file (`ballflight_to_outcome/papermill_run.log`) captures traceback details.