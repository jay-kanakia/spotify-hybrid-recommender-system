#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 901619351636.dkr.ecr.us-east-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 901619351636.dkr.ecr.us-east-1.amazonaws.com/spotify-ecr:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=spotify-ecr)" ]; then
    echo "Stopping existing container..."
    docker stop spotify-ecr
fi

if [ "$(docker ps -aq -f name=spotify-ecr)" ]; then
    echo "Removing existing container..."
    docker rm spotify-ecr
fi

echo "Starting new container..."
docker run -d -p 80:8000 --name spotify-ecr 901619351636.dkr.ecr.us-east-1.amazonaws.com/spotify-ecr:latest

echo "Container started successfully."