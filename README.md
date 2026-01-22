# Customer Churn Prediction using Deep Learning (ANN)

## 📌 Project Overview
Customer churn is a critical business problem where customers stop using a company’s services.  
This project uses a **Deep Learning Artificial Neural Network (ANN)** to predict whether a customer is likely to churn based on demographic and service usage data.

The model helps businesses proactively identify high-risk customers and improve retention strategies.

---

## 📊 Dataset
**Telco Customer Churn Dataset (IBM)**  
- Source: Public dataset  
- Records: Customer demographics, service details, billing information  
- Target Variable: `Churn` (Yes / No)

---

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib  

---

## ⚙️ Project Workflow
1. Data loading and inspection  
2. Data cleaning and preprocessing  
3. Encoding categorical variables  
4. Feature scaling using StandardScaler  
5. Train-test split  
6. Building ANN model  
7. Model training and validation  
8. Model evaluation using accuracy, confusion matrix, and classification report  
9. Saving the trained model  

---

## 🧠 Model Architecture
- Input Layer  
- Dense Layer (32 neurons, ReLU)  
- Dense Layer (16 neurons, ReLU)  
- Output Layer (1 neuron, Sigmoid)

---

## ▶️ How to Run the Project
### 1. Install dependencies
```bash
pip install -r requirements.txt
