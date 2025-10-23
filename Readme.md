
# 🧠 NVD Vulnerability Severity Classifier 

This project uses **Natural Language Processing (NLP)** and **XGBoost** to predict the **severity level** and **CVSS score** of vulnerabilities from their **text descriptions** in the **National Vulnerability Database (NVD)**.

The model does **not** require a CVE ID — it processes the vulnerability **description text** and automatically classifies its severity and estimated CVSS score.

---

## 🚀 Overview

The system consists of two machine learning models:
- 🟢 **XGBoost Classifier** → predicts severity (Low / Medium / High / Critical)  
- 🔵 **XGBoost Regressor** → predicts CVSS score (continuous 0–10 range)

Both models are trained using **TF-IDF vectorization** on textual CVE descriptions.

---


## ⚙️ Setup and Usage

### 1️⃣ Install Python 3.11
Download and install [Python 3.11](https://www.python.org/downloads/release/python-3110/).

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate
````

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train and evaluate models

```bash
python main.py
```

This step:

* Loads and preprocesses NVD data
* Vectorizes descriptions using TF-IDF
* Trains XGBoost Classifier and Regressor models
* Saves trained models in the `artifacts/model_training` directory

### 5️⃣ Run the API/web app

```bash
python app.py
```

Then open your browser at:
👉 **[http://localhost:5000/](http://localhost:5000/)**

You can input a vulnerability **description** and get:

* Predicted Severity (e.g., “High”)
* Predicted CVSS Score (e.g., “7.8”)

---

## 🧠 Example Input / Output

### 🔹 Input:

> "Buffer overflow in the XYZ parser allows remote attackers to execute arbitrary code via crafted network packets."

### 🔹 Output:

| Prediction | Value |
| ---------- | ----- |
| Severity   | High  |
| CVSS Score | 8.7   |

---

## 🧠 Model Details

* **Text Vectorization:** TF-IDF
* **Classification Model:** XGBoost Classifier
* **Regression Model:** XGBoost Regressor
* **Metrics:** Accuracy, R² Score, Mean Squared Error

---

## 🧰 Requirements

* Python 3.11
* pandas
* numpy
* scikit-learn
* xgboost
* flask

---

## 🧑‍💻 Author

**Gowtham D**
🔗 AI | NLP | Cybersecurity Enthusiast

---

## 🛡️ License

Licensed under the **MIT License** – see the LICENSE file for details.



