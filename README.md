# Customer Churn Prediction

This project focuses on predicting customer churn in a telecommunications company using machine learning algorithms. The dataset used for this project is the "Telco Customer Churn" dataset, which contains information about customers and whether they have churned or not.

## Dataset

The dataset used for this project is the "Telco Customer Churn" dataset, stored in the file `WA_Fn-UseC_-Telco-Customer-Churn.csv`. It contains the following columns:

- customerID: Unique identifier for each customer
- gender: Customer's gender (Male or Female)
- SeniorCitizen: Whether the customer is a senior citizen (0 or 1)
- Partner: Whether the customer has a partner (Yes or No)
- Dependents: Whether the customer has dependents (Yes or No)
- tenure: Number of months the customer has stayed with the company
- PhoneService: Whether the customer has a phone service (Yes or No)
- MultipleLines: Whether the customer has multiple phone lines (Yes, No, or No phone service)
- InternetService: Customer's internet service provider (Fiber optic, DSL, or No)
- OnlineSecurity: Whether the customer has online security (Yes, No, or No internet service)
- OnlineBackup: Whether the customer has online backup (Yes, No, or No internet service)
- DeviceProtection: Whether the customer has device protection (Yes, No, or No internet service)
- TechSupport: Whether the customer has tech support (Yes, No, or No internet service)
- StreamingTV: Whether the customer has streaming TV (Yes, No, or No internet service)
- StreamingMovies: Whether the customer has streaming movies (Yes, No, or No internet service)
- Contract: The contract term of the customer (Month-to-month, One year, or Two year)
- PaperlessBilling: Whether the customer has opted for paperless billing (Yes or No)
- PaymentMethod: Customer's payment method (Electronic check, Mailed check, Bank transfer, or Credit card)
- MonthlyCharges: The amount charged to the customer monthly
- TotalCharges: The total amount charged to the customer
- Churn: Whether the customer has churned (Yes or No)

## Preprocessing

Before applying machine learning algorithms, the dataset undergoes preprocessing steps, including:

- Handling missing values: There are no missing values in the dataset.
- Encoding categorical variables: Categorical variables are encoded as numeric values using label encoding.
- Standardization: The feature variables are standardized using the StandardScaler from scikit-learn.

## Machine Learning Algorithms

The following machine learning algorithms are applied to predict customer churn:

1. Logistic Regression: A binary classification algorithm that models the probability of customer churn.
2. Support Vector Machine (SVM): A classification algorithm that separates churned and non-churned customers using hyperplanes.
3. Decision Tree: A classification algorithm that creates a tree-like model to make predictions based on feature values.
4. K-Nearest Neighbors (KNN): A classification algorithm that classifies a new sample based on the majority class of its k-nearest neighbors.

## Model Evaluation

The models are evaluated using the following metrics:

- Accuracy: The percentage of correct predictions.
- Confusion Matrix: A matrix showing true positive, true negative, false positive, and false negative values.
- Cross-validation: Performance of the models using cross-validation techniques.

## Dependencies

The project requires the following dependencies:

- pandas: A library for data manipulation and analysis.
- scikit-learn: A machine learning library for Python.
- matplotlib: A plotting library for creating visualizations.
- seaborn: A library for creating informative and attractive statistical graphics.
- numpy: A library for mathematical operations.

## How to Run

To run the project, follow these steps:

1. Install the required dependencies using pip or conda.
2. Download the dataset "WA_Fn-UseC_-Telco-Customer-Churn.csv" and place it in the same directory as the project code.
3. Run the code in a Python environment.

Note: The code might take some time to execute, especially during the grid search and cross-validation steps.

## Results

The project outputs the following results:

- Best parameters and accuracy score for each model after hyperparameter tuning.
- Confusion matrix plots for each model's predictions on the test set.
- Performance scores of each model on the test set.
- Cross-validation accuracy scores for each model.

Please refer to the project code for detailed implementation and analysis.
