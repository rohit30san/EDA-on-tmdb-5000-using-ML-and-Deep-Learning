readme:
  title: "Movie Analytics & Prediction System (TMDB 5000)"

  description: >
    This project explores and models the TMDB 5000 Movies Dataset using both
    traditional machine learning and deep learning techniques. The goal is to
    analyze movie patterns, derive actor-based insights, and build predictive
    models for revenue prediction and movie success classification.

  goals:
    - "Revenue Prediction (Regression)"
    - "Movie Success Classification"
    - "Deep Learning-based Classification"

  project_structure:
    project_name: "Movie-ML-Project"
    files:
      ML.ipynb: "Exploratory Data Analysis + Machine Learning Models"
      DL.ipynb: "Deep Learning (Neural Network) Model"
      tmdb_5000_movies.csv: "Required dataset"
      README.md: "Project documentation"

  exploratory_data_analysis:
    steps:
      - "Data cleaning and missing value handling"
      - "Correlation heatmaps"
      - "Feature distributions"
      - "Actor-based analysis"
    actor_insights:
      - "Most frequent actors"
      - "Actors with highest total movie revenue"
      - "Actors with highest average ratings"

  feature_engineering:
    methods:
      - "Standardization using StandardScaler"
      - "Removal of irrelevant columns"
      - "Train-test split"
      - "Addition of constant term for regression"

  machine_learning_models:
    regression:
      models:
        - "Ridge Regression (RidgeCV)"
        - "XGBoost Regressor"
      metrics:
        - "Mean Squared Error (MSE)"
        - "Adjusted R² Score"
      visualization: "XGBoost feature importance plots"

    classification:
      models:
        - "Logistic Regression"
        - "XGBoost Classifier"
      metrics:
        - "Accuracy"
        - "Precision"
        - "Recall"
        - "F1-score"
        - "Confusion Matrix"

  deep_learning_model:
    type: "Multi-Layer Perceptron (MLP)"
    framework: "TensorFlow / Keras"
    architecture:
      - "Input Layer"
      - "Dense(64, ReLU)"
      - "Dropout"
      - "Dense(32, ReLU)"
      - "Dropout"
      - "Output Layer (Sigmoid)"
    training:
      - "Validation split"
      - "Loss curve visualization"
      - "Comparison with ML models"

  how_to_run:
    step_1_install_dependencies: >
      pip install numpy pandas seaborn matplotlib scikit-learn xgboost tensorflow

    step_2_dataset:
      instruction: "Place tmdb_5000_movies.csv in the same directory as notebooks"

    step_3_run_notebooks:
      command: "jupyter notebook"
      order:
        - "Run ML.ipynb first"
        - "Then run DL.ipynb"

  results_and_insights:
    - "Traditional ML models perform well with structured features"
    - "XGBoost generally outperforms Ridge Regression"
    - "Deep Learning captures nonlinear relationships"
    - "Actor-based analysis reveals meaningful industry patterns"

  technologies_used:
    - "Python"
    - "Pandas"
    - "NumPy"
    - "Scikit-Learn"
    - "XGBoost"
    - "TensorFlow / Keras"
    - "Matplotlib"
    - "Seaborn"

  author:
    name: "Rohit Sangwan"
