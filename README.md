# Movie Analytics & Prediction System (TMDB 5000)

This project analyzes the TMDB 5000 Movies Dataset using both traditional Machine Learning and Deep Learning techniques. It focuses on understanding movie patterns and building predictive models for:

- Movie Revenue Prediction (Regression)
- Movie Success Prediction (Classification)

--------------------------------------------------------------------

## Files in this Repository

- ML.ipynb – Exploratory Data Analysis, feature engineering, and classical ML models  
- DL.ipynb – Deep learning model using a neural network  
- tmdb_5000_movies.csv – Dataset used for training and analysis  

--------------------------------------------------------------------

## What the Project Includes

### 1. Exploratory Data Analysis (EDA)

In ML.ipynb, the notebook performs:

- Data cleaning and handling missing values  
- Visualizations (correlation heatmaps, distributions, trends)  
- Actor-based insights:
  - Most frequent actors  
  - Actors with highest total revenue  
  - Actors with highest average ratings  

--------------------------------------------------------------------

### 2. Machine Learning Models

#### Revenue Prediction (Regression)

Models used:

- Ridge Regression  
- XGBoost Regressor  

Evaluated using:

- Mean Squared Error (MSE)  
- Adjusted R² Score  

Feature importance is visualized using XGBoost.

#### Movie Success Classification

Models used:

- Logistic Regression  
- XGBoost Classifier  

Evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  

--------------------------------------------------------------------

### 3. Deep Learning Model

In DL.ipynb, a neural network is built using TensorFlow/Keras with:

- Dense layers  
- Dropout for regularization  
- Sigmoid output for binary classification  

Training includes:

- Train-validation split  
- Loss curve visualization  

--------------------------------------------------------------------

## How to Run the Project

1. Install dependencies:

   pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow

2. Place tmdb_5000_movies.csv in the same folder as the notebooks.

3. Open Jupyter Notebook:

   jupyter notebook

4. Run in order:

   ML.ipynb  
   DL.ipynb  

--------------------------------------------------------------------

## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-Learn  
- XGBoost  
- TensorFlow / Keras  
- Matplotlib, Seaborn  

--------------------------------------------------------------------

## Author

Rohit Sangwan
