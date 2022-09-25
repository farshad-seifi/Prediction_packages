import numpy as np
import pandas as pd

Bonus_data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\Bonus_table.xlsx")
data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\test_data_month.xlsx")
data.columns = ["date", "value", "segment"]
Bonus_data.columns = ["percentage", "Bonus"]
Base_price = 10
Economy_coefficient = 0.95
Given_target = 20000
model = "SARIMA"


def Target_Calculator(data, Bonus_data, Base_price, Economy_coefficient, Given_target, model):
    targets = np.zeros((len(data["segment"].unique()), 2))
    targets = pd.DataFrame(targets)
    targets.columns = ["segment", "target"]
    forecast = np.zeros((len(data["segment"].unique()), 2))
    forecast = pd.DataFrame(forecast)
    forecast.columns = ["segment", "forecast"]

    for i in range(0, len(data["segment"].unique())):

        target_data = data[data["segment"] == data["segment"].unique()[i]]
        target_data = target_data[["date", "value"]]
        target_data["date"] = target_data["date"].apply(date_handler)


        if model == "SARIMA":
            results = SARIMA_predictor(target_data, 1, True)
        elif model == "ExponentialSmoothing":
            results = ExponentialSmoothing_predictor(target_data, 1, True)
        elif model == "RandomForest":
            results = RandomForest_Model(target_data, 1)
        elif model == "LinearRegression":
            results = LinearRegression(target_data, 1)
        elif model == "DeepTCN":
            results = DeepTCN_predictor(target_data, 1, True)
        elif model == "RNN":
            results = RNNModel_predictor(target_data, 1, "RNN")
        elif model == "LSTM":
            results = RNNModel_predictor(target_data, 1, "LSTM")
        elif model == "GRU":
            results = RNNModel_predictor(target_data, 1, "GRU")
        elif model == "NBEATS":
            results = NBEATSModel_predictor(target_data, 1)
        elif model == "fbprophet":
            results = fbprophet_predictor(target_data, 1, True)

        forecast["segment"].iloc[i] = data["segment"].unique()[i]
        forecast["forecast"].iloc[i] = results["prediction"].iloc[0]
        targets["segment"].iloc[i] = data["segment"].unique()[i]
        targets["target"].iloc[i] = results["prediction"].iloc[0]

        if (results["UCL"].iloc[0] > 0):
            targets["target"].iloc[i] = min(results["UCL"].iloc[0], (targets["target"].iloc[i]) / Economy_coefficient)
        else:
            targets["target"].iloc[i] = (targets["target"].iloc[i]) / Economy_coefficient

    if Given_target > 0:
        correction_coefficient = Given_target / targets["target"].sum()
        targets["target"] = targets["target"] * correction_coefficient

    if (data.dtypes["value"] == 'int64'):
        targets["target"] = targets["target"].round()

    actual = np.zeros((len(data["segment"].unique()), 2))
    actual = pd.DataFrame(actual)
    actual.columns = ["segment", "percentage"]
    actual["percentage"] = forecast["forecast"] / targets["target"]
    actual["segment"] = targets["segment"]

    coef = np.zeros((len(actual["percentage"]), 2))
    coef = pd.DataFrame(coef)
    coef.columns = ["segment", "step"]

    for i in range(0, len(coef["step"])):
        status = False
        for j in range(0, len(Bonus_data["percentage"]) - 1):
            if (actual["percentage"].iloc[i] * 100 <= Bonus_data["percentage"].iloc[j]):
                coef["step"].iloc[i] = Bonus_data["Bonus"].iloc[j]
                status = True
                break

        if not status:
            coef["step"].iloc[i] = Bonus_data["Bonus"].iloc[len(Bonus_data["percentage"]) - 1]

        coef["segment"].iloc[i] = forecast["segment"].iloc[i]

    Total_Budget = ((forecast["forecast"] * (1 + coef["step"]) * Base_price).sum()).round()
    Total_Target = targets["target"].sum()

    return Total_Target, Total_Budget, actual, targets


Total_Target, Total_Budget, actual, targets = Target_Calculator(data, Bonus_data, Base_price, Economy_coefficient,
                                                                Given_target, model)
