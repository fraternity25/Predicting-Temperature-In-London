import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Read in the data
weather = pd.read_csv("london_weather.csv")

# Data Cleaning
weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d')

# Exploratory Data Analysis
print(weather.describe())

# Visualize some of the features
plt.figure(figsize=(10, 6))
sns.heatmap(weather.corr(), annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

sns.pairplot(weather, vars=['mean_temp', 'max_temp', 'min_temp', 'sunshine', 'global_radiation'])
plt.show()

# Feature Selection
features = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'min_temp', 'precipitation', 'pressure', 'snow_depth']
X = weather[features]
y = weather['mean_temp']

# Preprocessing Data
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_preprocessed = preprocessing_pipeline.fit_transform(X)

# Handle missing values in target variable
y = y.fillna(y.median())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Function to log model experiments
def log_experiment(model, model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        mlflow.log_param("model", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, model_name)
        
        return rmse

# Train and log models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

for model_name, model in models.items():
    rmse = log_experiment(model, model_name, X_train, y_train, X_test, y_test)
    print(f"{model_name} RMSE: {rmse}")

# Search and store experiment results
experiment_results = mlflow.search_runs()
print(experiment_results)
