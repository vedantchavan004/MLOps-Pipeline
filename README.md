# MLOps Pipeline for Automated Model Deployment
This project demonstrates an end-to-end MLOps pipeline for training and deploying a deep learning model. It includes containerized deployment (Docker), REST API serving (FastAPI), and CI/CD automation.
Key Features

✅ Deep Learning Model: Trained using PyTorch on the MNIST dataset.

✅ Containerized Deployment: Packaged with Docker for easy scalability and portability.

✅ REST API for Model Inference: Built using FastAPI to serve real-time predictions.

✅ CI/CD Automation: GitHub Actions automates testing and deployment.

✅ Docker Hub Integration: Pushes the Docker image for streamlined deployments.

🛠 How to Run Locally

1️⃣  Build and Run the Docker Container

docker build -t mlops-mnist:latest .
docker run -p 80:80 mlops-mnist:latest

2️⃣ Test the API using cURL

curl -X POST "http://localhost:80/predict/" -H "Content-Type: application/json" -d @input.json

