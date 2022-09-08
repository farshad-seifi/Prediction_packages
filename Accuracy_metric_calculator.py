import numpy as np
import pandas as pd

def Accuracy_metric_calculator(test_data, results):
    test_data = test_data.reset_index(drop=True)
    results = results.reset_index(drop=True)

    Metric_values = pd.DataFrame(np.zeros((1,5)))
    Metric_values.columns = ["MAPE", "MSE", "MAE", "RMSE", "SMAPE"]

    Metric_values["MAPE"] = 100 * np.abs((results["prediction"] - test_data["value"]) / (test_data["value"])).mean()
    Metric_values["MSE"] = ((results["prediction"] - test_data["value"]) ** 2).mean()
    Metric_values["MAE"] = (np.abs(results["prediction"] - test_data["value"])).mean()
    Metric_values["RMSE"] = np.sqrt(((results["prediction"] - test_data["value"]) ** 2).mean())
    Metric_values["SMAPE"] = 100 * (np.abs(results["prediction"] - test_data["value"]) / (
                                    np.abs(results["prediction"]) + np.abs(test_data["value"]))).mean()

    return Metric_values
