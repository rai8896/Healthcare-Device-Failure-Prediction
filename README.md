# 🏥 Healthcare Device Failure Prediction using Machine Learning

## 📌 Project Overview

This project implements an **AI-based Predictive Maintenance System** for healthcare/medical equipment using sensor data and machine learning.

The goal is to predict potential equipment failures before they occur by analyzing sensor degradation patterns. The system combines:

- Machine Learning based failure prediction
- Statistical analysis
- Feature importance analysis
- Survival analysis using Kaplan-Meier estimator
- Interactive Streamlit dashboard

This project is inspired by real-world **medical device reliability and quality analytics workflows** used in healthcare technology.

---

# 🎯 Problem Statement

Unexpected equipment failure in healthcare environments can impact:

- Patient safety
- Operational efficiency
- Maintenance cost
- Device reliability

A predictive maintenance system can identify early degradation signals and help perform preventive maintenance before failure.

---

# 📂 Dataset

## NASA Turbofan Engine Degradation Simulation Dataset (CMAPSS)

Dataset Source:

NASA Prognostics Center of Excellence (PCoE)

The dataset contains:

- Multiple engine units
- Time-series sensor readings
- Operating conditions
- Progressive degradation patterns


Each engine has multiple cycles where sensors capture the health condition over time.

---

# 🔄 Machine Learning Pipeline
## 🔄 Project Workflow

The complete predictive maintenance pipeline followed in this project:
## 🔄 Project Workflow

```text
NASA CMAPSS Sensor Data

            ↓

Data Cleaning & Preprocessing

            ↓

Exploratory Data Analysis (EDA)

            ↓

Remaining Useful Life (RUL) Calculation

            ↓

Failure Label Generation

            ↓

Feature Engineering

            ↓

Machine Learning Model
(Random Forest Classifier)

            ↓

Failure Probability Prediction

            ↓

Survival Analysis
(Kaplan-Meier Estimator)

            ↓

Streamlit Interactive Dashboard
```





# 🧹 Data Processing

Steps performed:

- Assigned column names to raw sensor data
- Removed unnecessary constant sensors
- Generated Remaining Useful Life (RUL)
- Created binary failure classification target


### Failure Risk Label Generation

For predictive maintenance, the Remaining Useful Life (RUL) of each engine is converted into a failure risk label.

- If **RUL ≤ 30 cycles**  
  → Engine is considered at **Failure Risk**  
  → Label = `1`

- If **RUL > 30 cycles**  
  → Engine is considered **Healthy**  
  → Label = `0`

This threshold helps the machine learning model learn the difference between normal operating conditions and engines that are approaching failure.

---

# 📊 Exploratory Data Analysis

Performed:

- Class distribution analysis
- Sensor degradation analysis
- Correlation analysis
- Feature importance analysis


Example:

Important failure contributing sensors:

- sensor_11
- sensor_4
- sensor_12
- sensor_7
- sensor_15

---

# 🤖 Machine Learning Model

## Random Forest Classifier

Why Random Forest?

- Handles nonlinear sensor patterns
- Works well with tabular sensor data
- Provides feature importance
- Robust against noise


Model features:

- Engine operating settings
- Sensor measurements
- Cycle information



---

# 📈 Model Performance

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-score


Example Result:
Accuracy ≈ 97%
Failure Recall ≈ 95%

High recall is important because missing a potential failure is more costly than false alarms in predictive maintenance.

---

# 📉 Survival Analysis

To analyze equipment lifetime behavior:

Implemented:

## Kaplan-Meier Survival Analysis


It estimates:
Probability that a device continues operating without failure over time


Output:

- Survival curve
- Device reliability trend
- Failure probability over cycles


---

# 🖥️ Streamlit Dashboard

The project includes an interactive dashboard.

Features:

## 1. Failure Prediction

User inputs:

- Cycle
- Operating settings
- Sensor values


Output:

- Failure probability
- Risk level


---

## 📁 Project Structure

```
Healthcare-Device-Failure-Prediction/

│
├── app.py
│   └── Streamlit application for deploying the predictive maintenance dashboard
│
├── Predictive_Maintenance_Medical_Equipment.ipynb
│   └── Complete data analysis, preprocessing, feature engineering,
│       model training, and evaluation workflow
│
├── failure_prediction_model.pkl
│   └── Trained machine learning model used for failure risk prediction
│
├── kmf_model.pkl
│   └── Kaplan-Meier Survival Analysis model for estimating equipment reliability
│
├── feature_importance.csv
│   └── Contains important features contributing to failure prediction
│
├── requirements.txt
│   └── List of required Python libraries and dependencies
│
└── README.md
    └── Project documentation and usage instructions
```

---

# ⚙️ Installation & Running

Clone repository:

```bash
git clone https://github.com/rai8896/Healthcare-Device-Failure-Prediction.git

---

# ⚙️ Installation & Running

Follow these steps to run the project locally:

```bash
# Clone repository
git clone https://github.com/rai8896/Healthcare-Device-Failure-Prediction.git

# Go inside project directory
cd Healthcare-Device-Failure-Prediction

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

# Run Streamlit application
streamlit run app.py
```

## 📌 Author

Manish Rai


Delhi Technological University








