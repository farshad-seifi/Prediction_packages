import jdatetime
import numpy as np
import pandas as pd
from datetime import datetime
import pandas as pd
import jdatetime
import numpy as np
from datetime import datetime, timedelta
import collections
from dateutil import parser
from darts.utils.statistics import check_seasonality, extract_trend_and_seasonality
from datetime import datetime
import math
from pulp import *


Bonus_data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\Test data\call center\Call handling Data_final.xlsx")

data = Bonus_data[['Received Time', 'Answer Time']]
Forward_days = 10                   #Number of desire step ahead which should be planed
Start_Working = 8                   #The starting hour in each day
End_Working = 22                    #The fininshing hour in each day
Wrap_time = 20                      #Average time for closing each call
ATA = 5                             #Average time to answer each new call
Required_Service_Level = 0.80       #Desire service level
Target_Answer_Time = 20             #Desire time to answer calls in queue
Desire_Occupancy = 0.8              #Maximum ocuupancy rate of each agent
Shrinkage = 0.3                     #Shrinkage rate of agents
Shifts = ((8,16), (10,18), (15,23)) #The available shifts


def CallCenter_WFP(data, Forward_days, Start_Working, End_Working, Wrap_time, ATA,
                   Required_Service_Level, Target_Answer_Time, Desire_Occupancy, Shrinkage, Shifts):
    # Removing Null values from receiving and answering time

    data = data[(~ pd.isna(data["Received Time"]))]
    data = data[(~ pd.isna(data["Answer Time"]))]

    # Converting Received and answer time to date-time format
    data["Received Time"] = data["Received Time"].apply(time_handling)
    data["Answer Time"] = data["Answer Time"].apply(time_handling)

    # Extracting Date, hour, and weekday from data
    # The most time consuming part!!!

    data_table = np.zeros((len(data), 7))
    data_table = pd.DataFrame(data_table)
    data_table.columns = ["Receive_date", "Receive_hour", "Receive_weekday", "Finish_date", "Finish_hour",
                          "Finish_weekday", "Handle_time"]

    for i in range(0, len(data)):
        data_table["Receive_date"].iloc[i] = datetime.date(data["Received Time"].iloc[i])
        data_table["Receive_hour"].iloc[i] = data["Received Time"].iloc[i].hour
        data_table["Receive_weekday"].iloc[i] = data["Received Time"].iloc[i].day_of_week

        data_table["Finish_date"].iloc[i] = datetime.date(data["Answer Time"].iloc[i])
        data_table["Finish_hour"].iloc[i] = data["Answer Time"].iloc[i].hour
        data_table["Finish_weekday"].iloc[i] = data["Answer Time"].iloc[i].day_of_week
        data_table["Handle_time"].iloc[i] = (data["Answer Time"].iloc[i] - data["Received Time"].iloc[i]).seconds

    # Finding share of each hour of a weekday from all the calls of day

    data_table_copy = data_table.copy()
    Hourly_share = data_table_copy.groupby(["Receive_hour", "Receive_weekday"]).count()
    Hourly_share.reset_index(inplace=True)
    Hourly_share = Hourly_share[['Receive_hour', "Receive_weekday", 'Finish_date']]
    Hourly_share.columns = ['Receive_hour', "Receive_weekday", 'share']

    # Calculating AHT which stands for Average Handle Time
    AHT = data_table_copy.groupby(["Receive_hour", "Receive_weekday"]).mean()
    AHT.reset_index(inplace=True)
    AHT = AHT[['Receive_hour', "Receive_weekday", 'Handle_time']]
    AHT.columns = ['Hour', 'Week_day', 'Handle_time']

    Total_calls = Hourly_share.groupby(["Receive_weekday"]).sum()
    Total_calls.reset_index(inplace=True)
    Total_calls = Total_calls[["Receive_weekday", 'share']]

    Hourly_share = pd.merge(Hourly_share, Total_calls, how='left', on=["Receive_weekday"])
    Hourly_share["share"] = Hourly_share["share_x"] / Hourly_share["share_y"]
    Hourly_share = Hourly_share[['Receive_hour', "Receive_weekday", 'share']]

    data_table = data_table.groupby(['Receive_date', 'Receive_weekday']).count()
    data_table.reset_index(inplace=True)
    data_table = data_table[['Receive_date', 'Finish_date']]
    data_table.columns = ['date', 'value']

    # Predicting future calls

    prediction = np.zeros((Forward_days, 3))
    prediction = pd.DataFrame(prediction)
    prediction.columns = ["Date", "Forecast", "Week_day"]

    data_table['date'] = data_table['date'].apply(date_handler)

    forecasts = SARIMA_predictor(data_table, Forward_days, True)

    for i in range(0, len(prediction)):
        prediction["Date"].iloc[i] = data_table["date"].max() + timedelta(days=i)
        prediction["Forecast"].iloc[i] = forecasts["prediction"].iloc[i]
        prediction["Week_day"].iloc[i] = prediction["Date"].iloc[i].day_of_week

    Working_hours = np.zeros(((End_Working - Start_Working + 1), 1))
    Working_hours = pd.DataFrame(Working_hours)
    Working_hours.columns = ["Hour"]

    for i in range(0, len(Working_hours)):
        Working_hours["Hour"].iloc[i] = Start_Working + i

    Hourly_share.columns = ['Hour', 'Week_day', 'share']
    prediction = prediction.merge(Working_hours, how='cross')

    prediction = pd.merge(prediction, Hourly_share, how='left', on=["Week_day", "Hour"])
    prediction["Final_forecast"] = prediction["Forecast"] * prediction["share"]
    prediction = pd.merge(prediction, AHT, how='left', on=["Week_day", "Hour"])
    prediction["Handle_time"] = prediction["Handle_time"] + Wrap_time + ATA

    prediction = prediction[(~ pd.isna(prediction["share"]))]

    prediction["Erlang"] = (prediction["Final_forecast"] * prediction["Handle_time"]) / (60 * 60)
    prediction["Agents"] = 0

    # Calculating number of raws agent based on Eralng C and occupancy rate
    for i in range(0, len(prediction)):
        Agents = int(prediction["Erlang"].iloc[i])
        SL = 0
        Occupancy = 1
        while ((SL < Required_Service_Level) and (Occupancy > Desire_Occupancy)):
            Agents = Agents + 1
            nominator = ((prediction["Erlang"].iloc[i] ** Agents) / (math.factorial(Agents))) * (
                        Agents / (Agents - prediction["Erlang"].iloc[0]))
            denominator = sum(
                (prediction["Erlang"].iloc[i] ** k) / math.factorial(k) for k in range(0, Agents - 1)) + nominator
            p_w = nominator / denominator
            SL = 1 - (p_w * math.exp(
                -((Agents - prediction["Erlang"].iloc[i]) * (Target_Answer_Time / prediction["Handle_time"].iloc[i]))))
            Occupancy = prediction["Erlang"].iloc[i] / Agents

        prediction["Agents"].iloc[i] = Agents

    # Considering the effect of shrinkage on the number of agents
    prediction["Agents"] = np.ceil(prediction["Agents"] / (1 - Shrinkage))

    # Optimizing the number of agents in each shifts

    Final_Agents = np.zeros((Forward_days, len(Shifts)))
    Final_Agents = pd.DataFrame(Final_Agents)

    index = 0
    for k in prediction["Date"].unique():

        temp = prediction[prediction["Date"] == k]
        temp.reset_index(inplace=True, drop=True)

        model = LpProblem("Ex", LpMinimize)

        for i in range(1, len(Shifts) + 1):
            globals()[f"x{i}"] = LpVariable([f"x{i}"][0], lowBound=0, cat="Continuous")

        model += sum(globals()[f"x{i}"] for i in range(1, len(Shifts) + 1))

        constraints_mat = np.zeros((len(temp), len(Shifts)))
        constraints_mat = pd.DataFrame(constraints_mat)

        # counter = 0
        for i in range(0, len(temp)):
            for j in range(0, len(Shifts)):
                if (i + int(temp["Hour"].min())) in range(Shifts[j][0], Shifts[j][1]):
                    constraints_mat[j].iloc[i] = 1

        for i in range(0, len(temp)):
            model += sum(globals()[f"x{j}"] * constraints_mat[j - 1].iloc[i] for j in range(1, len(Shifts) + 1)) >= \
                     temp["Agents"].iloc[i], [f"c{i}"][0]

        model.solve()

        for i in range(0, len(Shifts)):
            Final_Agents[i].iloc[index] = globals()[f"x{i + 1}"].varValue

        index += 1

    return prediction, Final_Agents


prediction, Final_Agents = CallCenter_WFP(data, Forward_days, Start_Working, End_Working, Wrap_time, ATA,
                                                Required_Service_Level, Target_Answer_Time, Desire_Occupancy, Shrinkage, Shifts)
