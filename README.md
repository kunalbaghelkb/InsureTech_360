# 🛡️ InsureTech 360: Intelligent Claims & Risk Assessment System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![ML](https://img.shields.io/badge/AI-TensorFlow%20%7C%20ScikitLearn-orange)
![GenAI](https://img.shields.io/badge/GenAI-LangChain%20%7C%20RAG-purple)

## 📌 Project Overview
**InsureTech 360** is an industry-level, full-stack AI application designed to automate and modernize the insurance claim process. Unlike traditional systems that rely on manual verification, this platform integrates multiple AI disciplines to handle "messy" real-world data.

It unifies **Computer Vision (CNN)** for damage assessment, **Deep Learning (ANN)** for fraud detection, and **Generative AI (RAG)** for policy assistance into a single dashboard.

---

## 🚀 Key Features
* **📸 Visual Inspection (CNN):** Automatically detects car damage types and severity from uploaded images using deep Convolutional Neural Networks.
* **🔍 Fraud Detection (ANN):** Analyzes tabular user data (policy details, history) to predict the probability of a claim being fraudulent using Artificial Neural Networks.
* **🤖 Smart Policy Assistant (RAG):** A GenAI-powered chatbot (built with LangChain & Vector DB) that answers user queries based on specific policy PDF documents.
* **📊 Dynamic Dashboard:** A professional UI built with HTML/CSS and Flask to visualize risk scores and analysis reports.
* **🏭 Modular Architecture:** Follows industry-standard coding practices (Pipelines, Components, Logging, Exception Handling) for scalability.

---

## 🛠️ Tech Stack
* **Languages:** Python 3.10
* **Backend Framework:** Flask
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Deep Learning:** TensorFlow/Keras (CNN, ANN)
* **Generative AI:** LangChain, FAISS (Vector DB), OpenAI/Google Gemini API
* **Frontend:** HTML5, CSS3, Bootstrap 5
* **DevTools:** Git, Dotenv

---

## 📂 Datasets & Resources Used

To run this project locally, you will need to download the following datasets and place them in the `artifacts/data/` folder as per the structure mentioned below.

### 1. Insurance Fraud Detection Dataset (CSV)
This dataset is used to train the ANN model for predicting fraudulent claims.
- **Source:** Kaggle (Vehicle Claim Fraud Detection)
- **Download Link:** [Click Here to Download CSV](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
- **Placement:** Extract and rename the file to `insurance_claims.csv` inside `artifacts/data/`.

### 2. Car Damage Detection Dataset (Images)
This dataset is used to train the CNN model to detect if a car is damaged or whole.
- **Source:** Kaggle (Car Damage Detection)
- **Download Link:** [Click Here to Download Images](https://www.kaggle.com/datasets/anujms/car-damage-detection)
- **Placement:** Extract the folders (`00-damage`, `01-whole`) inside `artifacts/data/car_images/`.
  - Ensure structure: `artifacts/data/car_images/data/training/` and `validation/`.

### 3. Policy Document for Chatbot (RAG)
This PDF is used by the Gemini-powered chatbot to answer user queries regarding insurance policies.
- **Sample File:** You can use any standard Car Insurance Policy PDF.
- **Example Link:** [Sample Policy PDF (Generic)](https://icai.newindia.co.in/NIAICAI/images/pdf/car_schedule.pdf)
- **Placement:** Save the file as `policy.pdf` inside `artifacts/data/`.

---

## 📂 Project Structure
The project follows a modular, production-ready structure:

## 📁 Project Structure

    InsureTech_360/
    ├── artifacts/                      # Stored Models, Preprocessors & Outputs (Ignored in Git)
    ├── logs/                           # Application & Pipeline Logs
    ├── notebooks/                      # Jupyter Notebooks for EDA & Experiments
    ├── src/                            # Source Code
    │   ├── components/                 # Core ML Components
    │   │   ├── data_ingestion.py       # Data Collection & Loading
    │   │   ├── data_transformation.py  # Feature Engineering & Preprocessing
    │   │   ├── model_trainer.py        # Model Training Logic
    │   │   └── model_evaluation.py     # Model Evaluation & Metrics
    │   ├── pipelines/                  # ML Pipelines
    │   │   ├── training_pipeline.py    # End-to-End Training Pipeline
    │   │   └── prediction_pipeline.py  # Inference / Prediction Pipeline
    │   ├── utils/                      # Helper & Utility Functions
    │   │   └── common.py
    │   ├── logger.py                   # Custom Logging Configuration
    │   ├── exception.py                # Custom Exception Handling
    │   └── constants.py                # Project-wide Constants & Configs
    ├── static/                         # CSS, JS, Images (Frontend Assets)
    ├── templates/                      # HTML Templates (Frontend)
    ├── .env                            # Environment Variables (Ignored in Git)
    ├── app.py                          # Application / API Entry Point
    ├── requirements.txt                # Project Dependencies
    ├── setup.py                        # Package Setup Configuration
    └── README.md                       # Project Documentation

---

## ⚙️ Installation & Setup

1. Clone the Repository
    ```bash
    git clone https://github.com/kunalbaghelkb/InsureTech_360.git && cd InsureTech_360

2. Create Virtual Environment
    ```bash
    python3.11 -m venv .venv

    # Windows
    .venv\Scripts\activate

    # Mac/Linux
    source .venv/bin/activate

3. Install Dependencies
    ```bash
    pip install -r requirements.txt

4. Set Environment Variables
Create a .env file in the root directory and add your keys
    ```bash
    GEMINI_API_KEY=your_api_key_here
    GOOGLE_LLM_MODEL_NAME=gemini-2.5-flash (configure as needed)
    GOOGLE_EMBED_MODEL_NAME=gemini-embedding-001 (configure as needed)

5. Generate Model
Execute this file to generate the model (Ensure that all required datasets and resources have been added as specified.)
    ```bash
    python src/pipelines/training_pipeline.py

6. Run the Application
    ```bash
    python app.py

---

## 👨‍💻 Author
**Kunal Baghel**

*Aspiring Data Scientist & AI Engineer*

[LinkedIn](https://linkedin.com/in/kunalbaghelz) | [GitHub](http://github.com/kunalbaghelkb)