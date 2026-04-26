
---

## 🧠 Hybrid Recommender System Logic

- **🎯 Content-Based Filtering**  
  Uses audio feature similarity via **Cosine Similarity**

- **👥 Collaborative Filtering**  
  Learns from user interaction patterns for discovery

- **🎚️ Diversity Slider**  
  A weighted hybrid system that allows users to tune recommendation balance

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/project-name.git](https://github.com/jay-kanakia/spotify-hybrid-recommender-system.git
cd spotify-hybrid-recommender-system

pip install -r requirements.txt

---

## 📂 Project Structure

├── .github
│ └── workflows
│ └── CI_CD.yml
│
├── .dvc
│ └── config
│
├── app.py
├── appspec.yml
├── Dockerfile
├── requirements.txt
├── dvc.yaml
├── dvc.lock
│
├── data
│ ├── raw
│ ├── cleaned
│ ├── filtered
│ └── processed
│
├── deploy
│ └── scripts
│ ├── install_dependencies.sh
│ └── start_docker.sh
│
├── src
│ ├── data
│ ├── features
│ └── models
│
├── tests
│ └── test_app.py
│
├── notebooks
│ ├── 01_EDA.ipynb
│ └── 02_Matrix_Sparsity.ipynb
│
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md


---

## 🧠 Hybrid Engine Logic

- **🎯 Content-Based Filtering**  
  Uses audio feature similarity via **Cosine Similarity**

- **👥 Collaborative Filtering**  
  Learns from user interaction patterns for discovery

- **🎚️ Diversity Slider**  
  A weighted hybrid system that allows users to tune recommendation balance

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/project-name.git
cd project-name

pip install -r requirements.txt

## ☁️ Deployment Workflow
- Push code to GitHub
- GitHub Actions triggers pipeline
- DVC pulls data from S3
- Docker image is built and pushed to ECR
- CodeDeploy updates EC2 instances

## 📈 Key Achievements
- 🚀 Reduced dataset size from 60GB → 31MB
- ⚡ Efficient large-scale processing using Dask
- 📦 Version-controlled ML pipeline with DVC
- 🔁 Fully automated CI/CD + Deployment
- 🧠 Hybrid recommendation system with tunable diversity

