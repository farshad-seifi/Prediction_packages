import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype, is_string_dtype, is_categorical_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


train_data = pd.read_csv(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\kaggle\Titanic\train - Copy.csv")
test_data = pd.read_csv(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\kaggle\Titanic\test.csv")

del train_data["PassengerId"]
del test_data["PassengerId"]
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
data = train_data.copy()

Label = "Survived"
model_name = "RandomForest"


def Missing_values_Handler(data, column_name):

    if not (is_numeric_dtype(data[column_name])):
        data[column_name].fillna(data[column_name].value_counts().idxmax(), inplace=True)
    else:
        data[column_name].fillna(data[column_name].mean(), inplace=True)


def classifier_model(data, model_name):

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

                if not (is_numeric_dtype(data[i])):
                    Missing_values_Handler(data, i)

                if(is_string_dtype(data[i]) or is_categorical_dtype(data[i])):

                    label_encoder = LabelEncoder()
                    label_encoder.fit(data[i])
                    Encoders[i] = label_encoder.fit(data[i])
                    data[i] = label_encoder.transform(data[i])


                #Filling missing values
                if (is_numeric_dtype(data[i])):
                    Missing_values_Handler(data, i)
        else:
            label_encoder = LabelEncoder()
            label_encoder.fit(data[i])
            data[i] = label_encoder.transform(data[i])

    #Data augmentation
    ratio_vec = np.zeros((len(data["Survived"].unique()) , 3))
    ratio_vec = pd.DataFrame(ratio_vec)
    ratio_vec.columns = ["label", "count", "rate"]

    for i in range(0, len(data["Survived"].unique())):

        ratio_vec["count"].iloc[i] = len(data[data["Survived"] == data["Survived"].unique()[i]])
        ratio_vec["rate"].iloc[i] = len(data[data["Survived"] == data["Survived"].unique()[i]]) / len(data)
        ratio_vec["label"].iloc[i] = data["Survived"].unique()[i]


    if (ratio_vec["count"].max() / ratio_vec["count"].min()) >= 4:
        for i in range(0, len(ratio_vec)):
            count = round((1/ratio_vec["rate"].iloc[i]) * ratio_vec["count"].iloc[i]) - ratio_vec["count"].iloc[i]
            resampled_data = data[data[Label] == ratio_vec["label"].iloc[i]].sample(n=int(count), replace=True)
            data = data.append(resampled_data)


    train, test = train_test_split(data, test_size= 0.3)
    number_of_classes = len(data[Label].unique())

    if (model_name == "RandomForest") :
        model = RandomForestClassifier(n_estimators=100)
        model.fit(train.drop([Label], axis= 1), train[Label])
        predictions = model.predict(test.drop([Label], axis=1))

    elif (model_name == "LogisticRegression"):
        model = LogisticRegression()
        model = model.fit(train.drop([Label], axis= 1), train[Label])
        predictions = model.predict(test.drop([Label], axis= 1))

    elif (model_name == "Xgboost"):
        model = xgb.XGBClassifier()
        model = model.fit(train.drop([Label], axis= 1), train[Label])
        predictions = model.predict(test.drop([Label], axis= 1))

    elif (model_name == "LinearSVC"):
        model = LinearSVC()
        model = model.fit(train.drop([Label], axis= 1), train[Label])
        predictions = model.predict(test.drop([Label], axis= 1))

    elif (model_name == "MultinomialNB"):
        model = MultinomialNB()
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


accuracy , model, Encoders, columns_name = classifier_model(data, model_name)
prediction = classifier_predictor(model, test_data, Encoders, columns_name)
