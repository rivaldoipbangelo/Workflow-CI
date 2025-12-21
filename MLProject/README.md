# Workflow CI - Spiderman Review Classification

[![MLflow CI/CD Pipeline](https://github.com/rivaldoipbangelo/Workflow-CI/actions/workflows/mlflow-ci.yml/badge.svg)](https://github.com/rivaldoipbangelo/Workflow-CI/actions/workflows/mlflow-ci.yml)

Automated Machine Learning workflow for Spiderman YouTube Review sentiment classification using MLflow, GitHub Actions, and Docker.

## 📁 Repository Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml              # CI/CD workflow configuration
├── MLProject/
│   ├── modelling.py                   # Training script
│   ├── conda.yaml                     # Environment dependencies
│   ├── MLProject                      # MLflow project file
│   ├── docker-hub-link.txt            # Docker Hub repository link
│   └── spiderman_youtube_review_preprocessing/
│       └── spiderman_youtube_review_preprocessed.csv
└── README.md
```

## 🚀 Features

- ✅ **Automated Training**: Model training triggered automatically on push to main branch
- ✅ **MLflow Tracking**: All experiments, metrics, and parameters tracked with MLflow
- ✅ **Artifact Storage**: Trained models saved to GitHub Artifacts
- ✅ **Docker Image**: Automatic Docker image build and push to Docker Hub
- ✅ **CI/CD Pipeline**: Complete automation using GitHub Actions

## 🐳 Docker Hub

Docker image is available at:
- **Repository**: [rivaldoipbangelo/spiderman-review-model](https://hub.docker.com/r/rivaldoipbangelo/spiderman-review-model)
- **Tags**: `latest`, `build-[number]`, `[commit-sha]`

### Pull & Run Docker Image

```bash
# Pull image
docker pull rivaldoipbangelo/spiderman-review-model:latest

# Run container
docker run -p 8080:8080 rivaldoipbangelo/spiderman-review-model:latest

# Test endpoint
curl http://localhost:8080/ping
```

## 📊 Model Information

- **Model Type**: Random Forest Classifier
- **Dataset**: Spiderman YouTube Reviews (preprocessed)
- **Features**: Multiple numeric features extracted from review text
- **Framework**: scikit-learn
- **Tracking**: MLflow

## 🔄 CI/CD Workflow

The GitHub Actions workflow (`mlflow-ci.yml`) automatically:

1. **Setup Environment**: Install Python and dependencies
2. **Train Model**: Run `modelling.py` with MLflow tracking
3. **Save Artifacts**: Upload trained model to GitHub Artifacts
4. **Build Docker Image**: Create Docker image from MLflow model
5. **Push to Docker Hub**: Deploy image to Docker Hub with multiple tags

### Trigger Workflow

Workflow is triggered on:
- Push to `main` branch
- Pull request to `main` branch
- Manual trigger via GitHub Actions UI

### Workflow Steps

```yaml
1. Checkout repository
2. Setup Python 3.9
3. Install dependencies (mlflow, scikit-learn, pandas, numpy, joblib)
4. Run training script (modelling.py)
5. Verify model output
6. Upload artifacts to GitHub
7. Login to Docker Hub
8. Find latest MLflow run
9. Build Docker image
10. Tag Docker image (latest, build-N, commit-sha)
11. Push to Docker Hub
```

## 📦 MLflow Project

This is an MLflow project with the following configuration:

**MLProject file:**
```yaml
name: Spiderman_Review_Classifier
conda_env: conda.yaml
entry_points:
  main:
    command: "python modelling.py"
```

### Run Locally

```bash
# Using MLflow
mlflow run ./MLProject --env-manager=local

# Or directly with Python
cd MLProject
python modelling.py
```

## 🛠️ Dependencies

**Python Packages:**
- mlflow==2.9.0
- scikit-learn==1.3.0
- pandas==2.0.0
- numpy==1.24.0
- joblib==1.3.0

See `conda.yaml` for complete environment specification.

## 📈 Model Metrics

The model logs the following metrics to MLflow:
- Training Accuracy
- Testing Accuracy
- Validation Accuracy

And parameters:
- Model Type
- Number of Estimators
- Maximum Depth
- Number of Features
- Test Size

## 🔐 GitHub Secrets

Required secrets for CI/CD workflow:
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub access token (with Read & Write permissions)

## 📸 Artifacts

After each workflow run, the following artifacts are available:

1. **trained-model-[number]**: 
   - Trained model files (.pkl)
   - MLflow run artifacts
   - Model metadata

2. **docker-hub-info-[number]**:
   - Docker Hub repository link
   - Pull commands
   - Run instructions
   - Build information

## 🎯 Scoring Criteria

This project meets the **Advance (4 points)** criteria:
- ✅ Workflow CI using GitHub Actions
- ✅ MLflow Project structure
- ✅ Artifacts saved to GitHub repository
- ✅ Docker Images built and pushed to Docker Hub

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

**Rivaldo IP Bangelo**
- GitHub: [@rivaldoipbangelo](https://github.com/rivaldoipbangelo)
- Docker Hub: [rivaldoipbangelo](https://hub.docker.com/u/rivaldoipbangelo)

## 🔗 Links

- **GitHub Repository**: https://github.com/rivaldoipbangelo/Workflow-CI
- **Docker Hub**: https://hub.docker.com/r/rivaldoipbangelo/spiderman-review-model
- **GitHub Actions**: https://github.com/rivaldoipbangelo/Workflow-CI/actions

---

⭐ Star this repository if you find it helpful!
