# Use an official Python runtime as a parent image
FROM python:3.10-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml and poetry.lock files
COPY 2-data-ingestion/pyproject.toml 2-data-ingestion/poetry.lock* ./

# Install dependencies without installing the current project
RUN poetry install --no-root

# Copy the entire 2-data-ingestion and core directories into the image
COPY 2-data-ingestion/ /app/
COPY core/ /app/core/

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Command to run the script
CMD ["poetry", "run", "python", "/app/cdc.py"]
