Machine Learning Engineer (0-2 Years Experience) – Assessment Task
Objective
The purpose of this assessment is to evaluate the candidate’s ability to preprocess data, train and evaluate machine learning models, deploy a simple model as an API, and follow basic MLOps practices.

Task: Build and Deploy a Machine Learning Model for Predicting House Prices
Instructions:
You are provided with a dataset containing house attributes and their corresponding prices. Your task is to:
Preprocess the data (handle missing values, feature engineering, scaling, encoding, etc.).
Train and evaluate a regression model that predicts house prices.
Optimize the model using hyperparameter tuning.
Deploy the model as a REST API using Flask or FastAPI.
Write a short report explaining your approach, decisions, and model performance.

Dataset
You can use any publicly available dataset such as Kaggle's House Price Prediction Dataset or California Housing Dataset (from Scikit-learn).

Part 1: Data Preprocessing
Load the dataset and perform exploratory data analysis (EDA).
Handle missing values appropriately.
Perform feature engineering (scaling, encoding categorical variables, feature selection).
Visualize correlations between features and the target variable.

Part 2: Model Training & Evaluation
Split the dataset into training and testing sets.
Train a regression model (e.g., Linear Regression, Decision Tree, Random Forest, XGBoost).
Evaluate the model using RMSE, MAE, and R² scores.
Optimize the model using GridSearchCV or RandomizedSearchCV.
Save the trained model using Pickle or Joblib.

Part 3: Model Deployment
Build a simple Flask or FastAPI application to serve predictions.
Create an endpoint /predict that takes input features as JSON and returns the predicted price.
Test the API using Postman or CURL.
Containerize the application using Docker (optional for bonus points).

Part 4: Report & Documentation
Provide a brief report explaining:
Steps taken for data preprocessing and feature engineering.
Model selection and optimization approach.
Deployment strategy and API usage guide.
Include clear and well-commented code in a GitHub repository or a Jupyter Notebook.

Bonus Points (Optional)
Implement logging and error handling in the API.
Deploy the API on AWS/GCP/Azure or Render.
Use DVC or MLflow for model versioning.
Create a simple frontend UI to interact with the model.

Submission Guidelines
Submit a GitHub repository containing:
Jupyter Notebook (.ipynb) or Python scripts (.py).
A README.md with clear instructions on running the project.
Model file (.pkl or .joblib).
Flask/FastAPI script (app.py).
Dockerfile (if applicable).
Provide a link to a hosted API (if deployed).



=====================================================================================================================

House Price Prediction Model
Project Overview
This project involves the development of a machine learning model that predicts house prices based on various features such as location, size, and other house attributes. The goal is to preprocess the data, train a regression model, optimize it, deploy it as an API, and provide documentation for its usage.

Contents
Introduction
Data Preprocessing
Model Training & Evaluation
Model Deployment
API Usage Guide
File Structure
Conclusion
Introduction
The project is designed to predict house prices using a regression model. It follows a systematic approach that includes:

Data Preprocessing: Handling missing values, scaling, and feature engineering.
Model Training and Evaluation: Using a Random Forest Regressor and evaluating it with metrics like RMSE, MAE, and R².
Model Optimization: Hyperparameter tuning using GridSearchCV for better performance.
Model Deployment: Exposing the model as a REST API using Flask.
Optional Containerization: Deploying the API with Docker for easy scalability.
Data Preprocessing
Dataset
The dataset used for this project is the California Housing Dataset from Scikit-learn, which contains house attributes like average income, number of rooms, etc., and their corresponding target variable (the house price).

Exploratory Data Analysis (EDA)
We start by loading the dataset and performing an initial exploration to understand its structure and check for missing values.

python
Copy
from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = fetch_california_housing()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

# Visualize correlations with target
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
Handling Missing Values
No missing values were found in this particular dataset. However, for other datasets, missing data could be handled by imputation or removal.

Feature Engineering
We performed scaling on the numerical features using StandardScaler to standardize the data so that all features have the same scale. This is important for regression models.

python
Copy
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=['target']))
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_scaled['target'] = df['target']
Model Training & Evaluation
Splitting the Data
We split the dataset into training and testing sets to evaluate model performance properly.

python
Copy
from sklearn.model_selection import train_test_split

X = df_scaled.drop(columns='target')
y = df_scaled['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Model Selection
We chose Random Forest Regressor, a robust and flexible model that can capture non-linear relationships in the data.

python
Copy
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
Model Evaluation
We evaluated the model's performance using the following metrics:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score
python
Copy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}")
Model Optimization
We used GridSearchCV to tune the hyperparameters of the Random Forest model for optimal performance.

python
Copy
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
Model Saving
We saved the final trained model using joblib so that it can be used in the Flask application.

python
Copy
import joblib

joblib.dump(grid_search.best_estimator_, 'house_price_predictor.pkl')
Model Deployment
Flask API
We built a Flask API to serve predictions from the trained model. The API listens for POST requests at the /predict endpoint, which accepts a JSON payload containing the feature values and returns the predicted house price.

python
Copy
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('house_price_predictor.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
Testing the API
You can test the API by sending a POST request with feature values in the following format:

json
Copy
{
    "features": [1.5, 5.5, 2.0, 1.5, 0.3, 1.0, 3.5, 2.0, 1.0, 4.0]
}
Optional: Dockerization
For easier deployment, the application can be containerized using Docker. Here’s the Dockerfile for this project:

dockerfile
Copy
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
Docker Build & Run
To build and run the Docker container:

bash
Copy
docker build -t house-price-predictor .
docker run -p 5000:5000 house-price-predictor
API Usage Guide
Once the Flask API is running, you can interact with it by sending a POST request to the /predict endpoint with the house feature values.

Sample CURL Command:
bash
Copy
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [1.5, 5.5, 2.0, 1.5, 0.3, 1.0, 3.5, 2.0, 1.0, 4.0]}'
Response Format:
json
Copy
{
    "predicted_price": 230000.5
}
File Structure
text
Copy
project_directory/
├── app.py                   # Flask API for deployment
├── house_price_predictor.pkl # Trained model file
├── Dockerfile               # Docker configuration (optional)
├── requirements.txt         # List of dependencies
└── house_price_prediction.ipynb  # Jupyter notebook (optional)
Conclusion
This project demonstrates the end-to-end process of building a machine learning model to predict house prices, optimizing it, and deploying it as a REST API using Flask. The model uses the California Housing Dataset and is based on a Random Forest Regressor. The model is optimized using GridSearchCV and deployed through a Flask API, making it accessible for predictions. Additionally, optional Dockerization provides ease of deployment in scalable environments.