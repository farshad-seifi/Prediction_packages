import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype, is_string_dtype, is_categorical_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

train_data = pd.read_csv(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\kaggle\Titanic\train.csv")
test_data = pd.read_csv(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\kaggle\Titanic\test.csv")

"""
Shenasaee sotoon label va saier sotoonhae mohem
tasmimgiri dar mored null values
tabdil dadeha be categorical va ya transform
estekhraje feature jadid
PCA
Data Augmentation
fit kardan model
eraee natayej
pishbini test
"""
data = train_data

Label = "Survived"

def classifier_model(data):

    Encoders = {}
    for i in data.columns:
        if i != Label:
            null_percentage = data[i].isna().sum() / len(data[i])
            diversification = (len(train_data[i].unique()) + train_data[i].isna().sum())/ len(data[i])

            if (is_string_dtype(data[i]) or is_categorical_dtype(data[i])) and diversification > 0.5:
                del data[i]

            elif null_percentage >= 0.5:
                del data[i]

            else:
                if (is_numeric_dtype(data[i])):
                    data[i].fillna(data[i].mean(), inplace= True)
                else:
                    data[i].fillna(data[i].value_counts().idxmax() , inplace= True)

                if(is_string_dtype(data[i]) or is_categorical_dtype(data[i])):

                    label_encoder = LabelEncoder()
                    label_encoder.fit(data[i])
                    Encoders[i] = label_encoder.fit(data[i])
                    data[i] = label_encoder.transform(data[i])

    #Data augmentation
    ratio_vec = np.zeros((len(data["Survived"].unique()) , 3))
    ratio_vec = pd.DataFrame(ratio_vec)
    ratio_vec.columns = ["label", "count", "rate"]

    for i in range(0, len(data["Survived"].unique())):

        ratio_vec["count"].iloc[i] = len(data[data["Survived"] == data["Survived"].unique()[i]])
        ratio_vec["rate"].iloc[i] = len(data[data["Survived"] == data["Survived"].unique()[i]]) / len(data)
        ratio_vec["label"].iloc[i] = data["Survived"].unique()[i]


    if (ratio_vec["count"].max() / ratio_vec["count"].min()) >= 1.1:
        for i in range(0, len(ratio_vec)):
            count = round((1/ratio_vec["rate"].iloc[i]) * ratio_vec["count"].iloc[i]) - ratio_vec["count"].iloc[i]
            resampled_data = data[data[Label] == ratio_vec["label"].iloc[i]].sample(n=int(count), replace=True)
            data = data.append(resampled_data)


    train, test = train_test_split(data, test_size= 0.3)
    model = LogisticRegression()
    model = model.fit(train.drop([Label], axis= 1), train[Label])
    predictions = model.predict(test.drop([Label], axis= 1))
    predictions = pd.DataFrame(predictions)
    accuracy = accuracy_score(test[Label], predictions)
    columns_name = data.drop([Label], axis=1).columns

    return accuracy, model, Encoders, columns_name


def classifier_predictor(model, test_data, Encoders, columns_name):

    test_data = test_data[columns_name]

    for i in test_data.columns:
        if (is_numeric_dtype(test_data[i])):
            test_data[i].fillna(test_data[i].mean(), inplace=True)
        else:
            test_data[i].fillna(test_data[i].value_counts().idxmax(), inplace=True)
        if (is_string_dtype(test_data[i]) or is_categorical_dtype(test_data[i])):

            test_data[i] = Encoders[i].transform(test_data[i])


    predictions = model.predict(test_data)

    return predictions


accuracy , model, Encoders, columns_name = classifier_model(data)
prediction = classifier_predictor(model, test_data, Encoders, columns_name)
