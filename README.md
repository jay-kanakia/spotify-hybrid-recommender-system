Overview
An end-to-end MLOps project that builds and deploys a Hybrid Song Recommendation System. The system manages 10 Million interaction rows, optimized from 60GB down to 31MB using SciPy Sparse matrices and Dask.

🛠️ The Production Pipeline
This project demonstrates a full CI/CD lifecycle for Machine Learning:

Data Versioning: Managed via DVC with AWS S3 as the remote storage.

CI/CD: GitHub Actions triggers on push to execute the DVC pipeline, run Pytest, and build Docker images.

Containerization: App is dockerized and pushed to Amazon ECR.

Deployment: Automated Rolling Updates to AWS EC2 via CodeDeploy using an Auto-Scaling Group behind an ALB.

📂 Updated Project Structure

├── .github
│   └── workflows
│       └── CI_CD.yml          <- GitHub Actions CI/CD Pipeline (DVC -> ECR -> CodeDeploy)
│
├── .dvc
│   └── config                <- DVC configuration (points to S3 remote storage)
│
├── app.py                    <- Main Streamlit Application UI
├── appspec.yml               <- AWS CodeDeploy configuration
├── Dockerfile                <- Container configuration for Amazon ECR
├── requirements.txt          <- Production-level dependencies
├── dvc.yaml                  <- DVC Pipeline definition (Stages: Data -> Features -> Matrix)
├── dvc.lock                  <- DVC state file (tracks data hashes for versioning)
│
├── data                      <- Data versioned by DVC (S3-backed)
│   ├── raw                   <- Original immutable dataset (10M rows)
│   ├── cleaned               <- Preprocessed/Cleaned data
│   ├── filtered              <- Intermediate filtered song/user mappings
│   └── processed             <- Final Sparse Matrices (31MB) & Pickled models
│
├── deploy                    <- Deployment orchestration
│   └── scripts           
│       ├── install_dependencies.sh  <- Script for CodeDeploy Agent (EC2 setup)
│       └── start_docker.sh          <- Script to pull from ECR and run container
│
├── src                       <- Source code module
│   ├── data                  <- Data fetching and S3 ingestion logic
│   ├── features              <- Feature Engineering & Sparse Matrix generation
│   └── models                <- Content/Collaborative hybrid logic
│
├── tests                     <- Automated Quality Assurance
│   └── test_app.py           <- Pytest script for CI/CD smoke tests
│
├── notebooks                 <- Research & Experimentation
│   ├── 01_EDA.ipynb          <- Initial exploratory data analysis
│   └── 02_Matrix_Sparsity.ipynb <- Prototyping SciPy Sparse logic
│
├── .dockerignore             <- Prevents heavy data/notebooks from bloating Docker images
├── .gitignore                <- Standard Git exclusions (ignores actual /data/ content)
├── LICENSE                   <- MIT or Apache 2.0
└── README.md                 <- This documentation

🧠 Hybrid Engine Logic
Content-Based: Audio DNA matching via Cosine Similarity.

Collaborative: Community pattern analysis for discovery.

Diversity Slider: A tunable weighted system allowing users to control the recommendation "sweet spot."
