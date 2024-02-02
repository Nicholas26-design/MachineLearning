
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import Row
from itertools import product

# # Assuming you already have a SparkSession named 'spark'
# spark = SparkSession.builder.appName("YourAppName").getOrCreate()

class ModelCombination:
    def __init__(self, param, param_seasonal, aic):
        self.param = param
        self.param_seasonal = param_seasonal
        self.aic = aic
        self.model = None

def fit_model_partition(iterator):
    local_spark = SparkSession.builder.getOrCreate()
    for params in iterator:
        yield fit_model(params, local_spark)

def fit_model(params, data):
    try:
        param, seasonal_param = params
        mod = sm.tsa.statespace.SARIMAX(data.rdd.map(lambda row: row.y),
                                        order=param,
                                        seasonal_order=seasonal_param,
                                        enforce_stationarity=True,
                                        enforce_invertibility=True)
        results = mod.fit(disp=False)
        aic = results.aic
        return ModelCombination(param, seasonal_param, aic)
    except Exception as e:
        print(f"Model fitting failed for {param}, {seasonal_param}. Error: {str(e)}")
        return None

def find_best_model(y):
    best_aic = float('inf')
    best_model = None

    # Define ranges for hyperparameters
    p = d = q = range(0, 8)
    pdq = list(product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(product(p, d, q))]
    model_name = f"{EXPERIMENT_NAME}-Sarimax"

    # Convert 'y' to a Spark DataFrame if not already in DataFrame format
    # y_spark = y if isinstance(y, DataFrame) else spark.createDataFrame([Row(value=float(val)) for val in y])

    # Create RDDs for parallelization
    pdq_rdd = spark.sparkContext.parallelize(pdq)
    seasonal_pdq_rdd = spark.sparkContext.parallelize(seasonal_pdq)

    # Cartesian product of pdq and seasonal_pdq
    param_space = pdq_rdd.cartesian(seasonal_pdq_rdd)

    # Perform coarse grid search in parallel using mapPartitions
    results = param_space.mapPartitions(lambda iterator: fit_model_partition(iterator)).collect()


    # # Perform coarse grid search in parallel using mapPartitions
    # param_space = spark.sparkContext.parallelize(pdq + seasonal_pdq)
    # results = param_space.mapPartitions(fit_model_partition).collect()



    # Find the best model based on AIC
    for res in results:
        aic = res.aic
        if aic < best_aic:
            best_aic = aic
            best_model = res

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


# Your code to retrieve and display the best model
model_uri, overall_results = find_best_model(spark_df)
model_uri = model_uri
results = overall_results

# Rest of your code to load the saved model and display summary statistics

print(model_uri)
print(results.summary())
# Load the saved model
loaded_model = mlflow.statsmodels.load_model(model_uri)
