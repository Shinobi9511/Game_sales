# 🎮 Video Game Global Sales Prediction

End-to-End Machine Learning Pipeline with Deployment

---

## 📌 Project Overview

This project builds a complete **end-to-end Machine Learning pipeline** to predict:

> **Global Video Game Sales (in millions)**

using structured game metadata such as platform, genre, publisher, and regional sales.

The project includes:

* Data Cleaning & Preprocessing
* Feature Engineering & Encoding
* Cross-Validation
* Hyperparameter Tuning (GridSearchCV)
* Multi-Model Comparison
* Feature Importance Analysis
* EDA Dashboard
* Streamlit Deployment App
* Docker Containerization
* Production-ready `.pkl` model

---

## 📂 Dataset

Dataset used:

**vgsales.csv**

Target Variable:

```
Global_Sales
```

Selected Features:

```
Platform
Year
Genre
Publisher
NA_Sales
EU_Sales
JP_Sales
```

---

## 🧠 ML Pipeline Architecture

The pipeline uses **Scikit-Learn's ColumnTransformer + Pipeline** for clean preprocessing and training.

### 🔹 Preprocessing

* Missing value removal
* StandardScaler (Numerical features)
* OneHotEncoder (Categorical features)
* Train/Test split (80/20)

### 🔹 Algorithms Compared

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree Regressor
* Random Forest Regressor

### 🔹 Model Selection Strategy

* 5-Fold Cross-Validation
* GridSearchCV for hyperparameter tuning
* Evaluation Metrics:

  * R² Score
  * RMSE
* Best performing model saved as:

```
best_model.pkl
```

---

## 📊 Exploratory Data Analysis (EDA)

The Streamlit app includes:

* Dataset preview
* Correlation heatmap
* Sales distribution histogram
* Interactive prediction interface
* Feature importance visualization (tree-based models)

---

## 🚀 Streamlit Application

The app provides:

* 📊 EDA Dashboard
* 🎯 Real-time Sales Prediction
* 📈 Feature Importance Visualization

### Run Locally

```bash
streamlit run app.py
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t vgsales-app .
```

### Run Container

```bash
docker run -p 8501:8501 vgsales-app
```

App will be available at:

```
http://localhost:8501
```

---

## 📁 Project Structure

```
vgsales_ml_project/
│
├── train.py            # Model training pipeline
├── app.py              # Streamlit application
├── best_model.pkl      # Trained model
├── vgsales.csv         # Dataset
├── requirements.txt    # Dependencies
├── Dockerfile          # Container configuration
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone <repo-url>
cd vgsales_ml_project
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train Model

```bash
python train.py
```

### 5️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📦 requirements.txt

Core libraries:

* pandas
* numpy
* scikit-learn
* streamlit
* matplotlib
* seaborn

---

## 📈 Model Evaluation Strategy

| Step             | Method           |
| ---------------- | ---------------- |
| Data Split       | 80/20 Train-Test |
| Cross Validation | 5-Fold CV        |
| Tuning           | GridSearchCV     |
| Selection Metric | R² Score         |
| Error Metric     | RMSE             |

---

## 🔍 Feature Importance

For tree-based models (Decision Tree / Random Forest):

* Extracted via `feature_importances_`
* Displayed in Streamlit
* Useful for interpretability and business insight

---

## 💡 Why This Project Is Production-Ready

* Clean modular training script
* Encapsulated preprocessing pipeline
* No data leakage
* Hyperparameter tuning integrated
* Automatic best-model selection
* Deployment-ready architecture
* Dockerized environment
* Scalable structure

---

## 🎯 Use Cases

* Sales forecasting
* Market trend analysis
* Game release strategy modeling
* Revenue projection systems

---

## 🧑‍💻 Author

**Aanjney Kumawat**
Petroleum Engineer | Data Science & ML Enthusiast
Python | SQL | Tableau | Machine Learning | Deployment

Let me know what level you want next.
