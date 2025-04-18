# Import necessary libraries
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools
import statsmodels.api as sm
import mlflow

# Load the dataset
# Context: This dataset contains time series data for forecasting.
data = pd.read_csv('time_series_data.csv', index_col='date', parse_dates=True)

# Split the data into training and testing sets
# Context: Training data is used to fit the model, and testing data is used for evaluation.
train, test = data[:'2023'], data['2024':]

# Best version of AIC

class ModelCombination:
    def __init__(self, param, param_seasonal, aic):
        self.param = param
        self.param_seasonal = param_seasonal
        self.aic = aic
        self.model = None

def find_best_model(y):
    best_aic = np.inf
    best_params = None
    best_model = None

    # Define ranges for hyperparameters
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    model_name = f"{EXPERIMENT_NAME}-Sarimax"

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=True,
                                                enforce_invertibility=True)
                results = mod.fit(disp=False)
                aic = results.aic

                # Check if current AIC is the best so far
                if aic < best_aic:
                    best_aic = aic
                    best_params = (param, param_seasonal)
                    best_model = ModelCombination(param, param_seasonal, aic)
                    best_model.model = results  # Storing the best model object
            except Exception as e:
                print(f"Model fitting failed for {param}, {param_seasonal}. Error: {str(e)}")
                continue

    # Log the best model and parameters using MLflow
    if best_model is not None and best_model.model is not None:
        with mlflow.start_run(run_name="Sarimax") as run:
            mlflow.statsmodels.log_model(best_model.model, model_name, registered_model_name=model_name)
            mlflow.log_params({"order": best_model.param, "seasonal_order": best_model.param_seasonal, 'enforce_stationarity': True, 'enforce_invertibility': True})
            model_uri = f"runs:/{run.info.run_id}/{model_name}"
            print("Best model saved in run %s" % run.info.run_id)
            print(f"Best model URI: {model_uri}")
            mlflow.end_run()
            return model_uri, best_model.model  
    else:
        print("No valid model found")

# I need this part
model_uri, overall_results = find_best_model(train)
model_uri = model_uri
results = overall_results
print(model_uri)
print(results.summary())
# Load the saved model
loaded_model = mlflow.statsmodels.load_model(model_uri)

# Fit an ARIMA model
# Context: ARIMA is a popular model for time series forecasting.
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))
mse = mean_squared_error(test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Plot the results
# Context: Visualizing predictions helps assess model performance.
import matplotlib.pyplot as plt
plt.plot(test, label='Actual')
plt.plot(predictions, label='Predicted', color='red')
plt.legend()
plt.show()

