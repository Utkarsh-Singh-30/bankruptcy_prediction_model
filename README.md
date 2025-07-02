# bankruptcy_prediction_model

# Project Overview
  This project focuses on building a machine learning model to predict the likelihood of companies going bankrupt based on various financial attributes. The goal is to develop a robust and accurate model that can   assist in identifying at-risk companies.

# Dataset
  The dataset used in this project contains financial ratios and other relevant attributes for a large number of companies over several years. The data is in ARFF format, and the project combines data from          multiple years (1-5 years) to create a comprehensive dataset for analysis.

# Methodology
The project follows a standard machine learning pipeline, including:

  # Data Loading and Concatenation: 
    Loading ARFF files for each year and concatenating them into a single Pandas DataFrame.
  # Exploratory Data Analysis (EDA):
    Checking for missing values and visualizing their distribution.
    Performing univariate analysis (histograms, box plots, descriptive statistics) to understand individual feature distributions.
    Conducting multivariate analysis (correlation matrix, scatter plots, pair plots, grouped box plots) to explore relationships between features.
  # Data Preprocessing:
    Handling missing values using the K-Nearest Neighbors (KNN) imputer.
    Scaling numerical features using StandardScaler.
    Handling outliers using the Interquartile Range (IQR) method and Winsorizing.
  # Feature Selection:
    Employing a clustering-based approach (KMeans) to group similar features.
    Using an ensemble-based method (Gradient Boosting) within each cluster to identify important features.
    Validating the selected features using permutation importance.
  # Handling Class Imbalance:
    Comparing different oversampling and undersampling techniques (SMOTE, SMOTETomek, SMOTEENN, ADASYN) to address the imbalanced nature of the target variable (bankruptcy).
    Selecting the most effective technique based on evaluation metrics.
  # Dimensionality Reduction:
    Applying Principal Component Analysis (PCA) to reduce the number of features while retaining most of the variance.
  # Model Training:
    Training various classification models on the preprocessed and dimensionality-reduced data, including:
    Random Forest
    Support Vector Machine (SVM)
    Gradient Boosting
    XGBoost
    CatBoost
    Decision Tree
    Logistic Regression
  # Model Evaluation:
    Evaluating the performance of each model using appropriate metrics (Accuracy, Precision, Recall, F1-score, AUC).
    Visualizing the comparison of model performance.
    Analyzing the confusion matrix and learning curves for the best-performing models.
  # Model Interpretation:
    Examining feature importances to understand which features contribute most to the predictions (using techniques like SHAP).
  # Testing:
    Making predictions on new data using the trained model.
    Deployment Preparation:
    Generating a requirements.txt file to list project dependencies.
    Saving the trained model using joblib for future deployment.
# Tools and Libraries Used
  Python
  Pandas
  NumPy
  Matplotlib
  Seaborn
  SciPy
  Scikit-learn
  Imbalanced-learn
  XGBoost
  CatBoost
  SHAP
  Joblib
  Liac-arff
  Missingno
