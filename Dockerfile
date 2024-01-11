# Use an official Ubuntu as the base image
FROM ubuntu:latest

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary packages
RUN apt-get update -y \
    && apt-get install -y python3 python3-pip build-essential cmake git g++ gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (if any)
COPY ./requirements.txt /app
RUN pip3 install -r requirements.txt

COPY . /app
# Start an interactive shell when the container is run
CMD ["/bin/bash"]