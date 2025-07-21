# Salary Prediction using Loan Dataset

This repository contains a machine learning project that predicts salary based on features from a loan dataset using a **Linear Regression** model. The implementation is done in a Jupyter Notebook.

## 📁 Files

- `Salary_Prediction.ipynb` — Jupyter Notebook containing data loading, preprocessing, model training, and evaluation.
- `loan.csv` — Dataset used for training and testing the prediction model *(required for notebook to run)*.
- `requirements.txt` — List of Python dependencies to run the notebook.

## 🧠 Project Summary

The goal of this project is to predict an individual's salary based on data provided in a loan application dataset. This includes:

- Loading and exploring `loan.csv`
- Data cleaning and preprocessing
- Building a linear regression model
- Evaluating the model
- Visualizing results

## 🛠️ Technologies Used

- Python 3.x  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mincater/salary-prediction.git
cd salary-prediction
```

### 2. Install Required Libraries

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Add the Dataset

Make sure the file `loan.csv` is in the same directory as the notebook. The notebook will not run without this dataset.

### 4. Run the Notebook

```bash
jupyter notebook Salary_Prediction.ipynb
```

## 📊 Output

- Correlation analysis and plots  
- Regression model summary  
- Salary predictions based on input features from the loan data

## 📌 Note

This is a basic linear regression project intended for learning and demonstration purposes. Performance may vary depending on the dataset features and quality.

---

## 📦 requirements.txt

```txt
matplotlib
numpy
pandas
seaborn
sklearn
```
