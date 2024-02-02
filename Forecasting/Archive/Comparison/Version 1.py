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
model_uri, overall_results = find_best_model(Training_data)
model_uri = model_uri
results = overall_results
print(model_uri)
print(results.summary())
# Load the saved model
loaded_model = mlflow.statsmodels.load_model(model_uri)

