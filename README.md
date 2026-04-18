# Visibility Distance Prediction (Time-Series ML System)

## Problem Statement

The objective of this project is to build a robust time-series machine learning system that predicts visibility distance using historical weather data.

The system leverages meteorological features such as:
- Temperature
- Humidity
- Wind speed
- Atmospheric pressure
- Temporal patterns (hourly, seasonal trends)

The goal is to improve decision-making in aviation, transportation, and safety-critical environments by providing accurate visibility forecasts.

---

## Project Highlights

- Built an end-to-end time-series ML pipeline
- Engineered advanced features:
  - Lag features (1–48 hours)
  - Rolling mean & standard deviation
  - Seasonal & time-based features
- Trained multiple models:
  - XGBoost
  - LightGBM
  - Random Forest
- Applied TimeSeriesSplit for proper validation
- Performed hyperparameter tuning using GridSearchCV
- Developed a stacking ensemble model:
  - Base models: XGBoost + LightGBM + Random Forest
  - Meta model: Ridge Regression
- Achieved:
  - R² Score: 0.91
  - Low MAE & RMSE on test data
- Built production-ready pipeline using Joblib
- Exposed model via FastAPI API
- Containerized using Docker

---

## Tech Stack

- Programming: Python
- ML Libraries: Scikit-learn, XGBoost, LightGBM
- Data Processing: Pandas, NumPy
- Modeling: Time-Series ML, Ensemble Learning
- Backend: FastAPI
- Database: MongoDB
- Deployment: Docker
- Serialization: Joblib

---

## Infrastructure

- AWS S3 (model storage)
- Azure (deployment support)
- GitHub Actions (CI/CD)

---

## Project Architecture

## Project Architecture - ![WhatsApp Image 2022-09-22 at 15 29 19](https://user-images.githubusercontent.com/71321529/192722336-54016f79-89ef-4c8c-9d71-a6e91ebab03f.jpeg) ## Data Collection Architecture - ![WhatsApp Image 2022-09-22 at 15 29 10](https://user-images.githubusercontent.com/71321529/192721926-de265f9b-f301-4943-ac7d-948bff7be9a0.jpeg) ## Deployment Architecture - ![deployment](https://user-images.githubusercontent.com/104005791/199660875-c8e63457-432a-44cb-8a95-800870f3da15.png)


## How to Run Locally

### Step 1: Clone Repository
git clone <your-github-repo-link>
cd <repo-name>

### Step 2: Create Environment
conda create --prefix venv python=3.8 -y
conda activate venv/

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Set Environment Variables
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

export MONGODB_URL=<MONGODB_URL>

### Step 5: Run Application
python3 app.py

### Step 6: Train Model
http://localhost:8062/train

### Step 7: Prediction Endpoint
http://localhost:8062/predict

---

## Run with Docker

### Build Image
docker build -t visibility-app .

### Run Container
docker run -d -p 8062:8062 visibility-app

---

## Models Used

### Base Models:
- XGBoost
- LightGBM
- Random Forest

### Ensemble:
- Stacking Regressor  
  - Base: XGB + LGBM + RF  
  - Meta: Ridge Regression  

---

## Evaluation Metrics

- R² Score  
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

---

## Key Features

- Time-series aware validation (no data leakage)  
- Modular pipeline architecture  
- Config-driven model tuning (model.yaml)  
- Production-ready deployment  
- Scalable API system  

---

## Future Improvements

- Add LSTM / Transformer-based models  
- Real-time streaming predictions  
- Dashboard visualization (Streamlit / React)  
- Auto-retraining pipeline  

---

## Conclusion

This project demonstrates a complete production-grade ML system for time-series forecasting, combining:
- Advanced feature engineering  
- Ensemble learning  
- Scalable deployment  

It is suitable for real-world industrial applications where accurate visibility prediction is critical.