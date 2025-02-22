
### **README.md**
```markdown
# 🚗 Car Price Prediction using MLflow  
*A Machine Learning Project for Predicting Car Selling Prices with MLflow Tracking*  

## 📌 Project Overview  
This project aims to predict car selling prices based on historical data using multiple regression models.  
It incorporates **MLflow** to track experiments and evaluate different models efficiently.

---

## 📂 Project Structure  
```
ML_A2_Assigement/
│── A2_Car_Price_Prediction.ipynb    # Jupyter Notebook for training & inference
│── best_car_price_prediction.model   # Saved best model
│── Cars.csv                           # Dataset (raw)
│── Cleaned_data.csv                    # Preprocessed dataset
│── docker-compose.yml                   # Docker configuration for MLflow
│── model/
│   ├── compare.html                     # HTML report for comparisons
│   ├── index_a2.html                     # Homepage for visualization
│   ├── index.html                        # Main page
│   ├── menu.html                         # Navigation menu
│── static/
│── mlruns/                              # MLflow tracking logs
│── venv/                                # Virtual environment (optional)
│── requirements.txt                      # Python dependencies
│── mlflow.db                             # MLflow database
│── ML_flow.png                           # Screenshot of MLflow experiments
│── ML_flow2.png                          # Screenshot of MLflow comparisons
│── README.md                             # This file
```

---

## 📊 Model Training and Experimentation  
This project uses **Linear Regression with different optimization techniques**.  
The best model is selected based on **MSE (Mean Squared Error) and R² score**.

### **💡 MLflow Tracking Setup**
- **Logged Metrics:** `train_mse`, `train_r2`, `test_mse`, `test_r2`
- **Tracked Hyperparameters:** `learning rate (lr)`, `regularization`, `momentum`, `batch method`
- **Experiment Comparison:** MLflow UI for tracking multiple runs

### **🔬 Best Model Selection Process**
The model selection process involves iterating over various hyperparameters and choosing the best configuration based on the highest **R² score**.

---

## 🚀 Installation & Setup  

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Aikezz9/ML_A2_Assigement.git
cd ML_A2_Assigement
```

### **2️⃣ Install Dependencies**
Create and activate a virtual environment (recommended):  
```bashs
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

Then install the required dependencies:
```bash
pip install -r requirements.txt
```

### **3️⃣ Run MLflow Tracking**
To track model experiments, start the MLflow server:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5050
```
Now, open **http://localhost:5050** to view the experiment logs.

---

## 🔥 Model Inference (Prediction)
After training, you can **use the best model** to make predictions:

```python
import pickle
import numpy as np

# Load the trained model
filename = "best_car_price_prediction.model"
loaded_model_data = pickle.load(open(filename, "rb"))

model = loaded_model_data["model"]
scaler = loaded_model_data["scaler"]

# Sample input data
sample_data = np.array([[2014, 103.52, 21.14]])
sample_scaled = scaler.transform(sample_data)

# Predict the price
predicted_price = model.predict(sample_scaled)
predicted_price = np.exp(predicted_price)

print(f"Predicted Car Selling Price: {predicted_price[0]:,.2f}")
```

---

## 📸 MLflow Experiment Results  
**Captured Screenshots from MLflow Experiment Tracking:**  

📌 **Experiment Runs**  
![MLflow Experiment 1](ML_flow.png)  

📌 **Best Model Comparison**  
![MLflow Experiment 2](ML_flow2.png)  

---

## 🛠️ Deployment with Docker  
This project supports **Dockerized MLflow tracking** for reproducibility.  

1️⃣ Build and run the MLflow container:  
```bash
docker-compose up -d
```
2️⃣ Open **http://localhost:5050** to view MLflow UI.

---

## 📝 Author
Developed by **[Aikezz9](https://github.com/Aikezz9)**  
Project Repository: [ML_A2_Assigement](https://github.com/Aikezz9/ML_A2_Assigement)  

