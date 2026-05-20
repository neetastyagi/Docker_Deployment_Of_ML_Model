# Docker_Deployment_Of_ML_Model

## Project Overview

This project demonstrates the deployment of a Generalized Linear Model (GLM) machine learning model as a REST API using Docker containerization. The solution provides an easy way to package, deploy, and serve predictions from a pre-trained GLM model through HTTP endpoints.

## Purpose

The primary goal of this project is to showcase how to:
- Deploy a machine learning model in a containerized environment using Docker
- Create a REST API for making predictions from a trained GLM model
- Make the API easily portable and deployable across different environments
- Provide a scalable solution for serving ML model predictions

## Project Structure

```
DockerDeployment/
├── Dockerfile              # Container configuration for building the Docker image
├── run_api.txt            # Shell script commands for building and running the Docker container
└── api_demo/              # Main application directory
    ├── GLM.pkl            # Pre-trained Generalized Linear Model (pickle format)
    ├── predict_api.py     # Flask application with prediction endpoints
    ├── requirements.txt   # Python dependencies (Flask, Flasgger)
    └── test_case.json     # Sample test data for API testing
```

## Key Components

### 1. **Dockerfile**
- Uses `continuumio/anaconda3` as the base image (includes Python and conda)
- Sets up the working directory at `/usr/local/python/`
- Installs required Python packages (pip, Cython, dependencies from requirements.txt)
- Exposes port 1313 for the API service
- Runs the Flask API application (`predict_api.py`)

### 2. **predict_api.py**
- Flask-based REST API for model predictions
- Loads the pre-trained GLM model from `GLM.pkl`
- Provides two main endpoints:
  - **`/predict`**: Accepts JSON query parameters and returns business outcome predictions (Event vs Not Event)
  - **`/predict_file`**: Processes test data from file and returns predictions
- Uses Flasgger for automatic Swagger/OpenAPI documentation
- Performs feature engineering on input data (categorical encoding, data transformation)
- Returns predictions with probability scores and business outcomes

### 3. **Model Input Features**
The API expects data with the following key features:
- `x5`: Day of week
- `x81`: Month
- `x31`: Geography (Japan, Asia, Germany)
- `x12`: Numeric value (currency-like format)
- `x44`, `x53`, `x56`, `x58`, `x62`, `x91`: Various numeric features

### 4. **Predictions**
- Model generates probability scores
- Classification threshold: 0.75
- Output classes: "Event" or "Not Event"
- Returns JSON format with prediction probabilities and model inputs

## Dependencies

- **Flask**: Web framework for building the REST API
- **Flasgger**: API documentation and Swagger UI
- **Pandas**: Data manipulation and processing
- **NumPy**: Numerical computations
- **Python 3.x** (via Anaconda)

## Getting Started

### Prerequisites
- Docker installed on your system

### Build and Run

Use the commands in `run_api.txt`:

```bash
# Build the Docker image
docker build -t glm-api .

# Run the Docker container
docker run -p 1313:1313 glm-api
```

This will:
1. Build a Docker image named `glm-api`
2. Start a container that exposes the API on port 1313
3. Load the pre-trained GLM model
4. Start the Flask API server

### API Access

Once running, you can access the API at:
- **Swagger UI**: `http://localhost:1313/apidocs`
- **Predictions endpoint**: `http://localhost:1313/predict?input_json=<JSON_DATA>`

## Use Case

This deployment pattern is ideal for:
- **Production ML Model Serving**: Containerized deployment ensures consistency across environments
- **Microservices Architecture**: Easy integration with larger systems
- **Scalability**: Docker containers can be orchestrated using Kubernetes or similar tools
- **Team Collaboration**: Reproducible environment for development and production

## Benefits

✅ **Portability**: Run the same container anywhere Docker is installed  
✅ **Isolation**: Model and dependencies isolated from system Python  
✅ **Reproducibility**: Exact same environment for development and production  
✅ **Scalability**: Easy to scale using container orchestration tools  
✅ **Easy API Access**: REST endpoints for programmatic prediction access  
✅ **Documentation**: Built-in Swagger UI for API exploration  

## Technology Stack

- **Language**: Python
- **Framework**: Flask
- **Containerization**: Docker
- **Base Image**: Anaconda3 (includes Python, pip, conda)
- **ML Model**: Generalized Linear Model (GLM)
- **API Documentation**: Flasgger/Swagger

---

**Author**: UNP (https://unp.education)  
**Language**: Python  
**License**: Not specified
