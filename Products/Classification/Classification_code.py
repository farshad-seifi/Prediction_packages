import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype, is_string_dtype, is_categorical_dtype, is_integer_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




test_data = pd.read_csv(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\Test data\مودم\prov_4380.csv")
test_data.columns = ["Number", "National_ID", "Age", "Cell_type", "Register_date", "Sex", "Register_province", "Mostused_province", "Brand", "Model", "ARPU3", "ARPU12"]

test_data = test_data.sample(n = 10000)
train_data = pd.read_excel(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\Prediction_product\Test data\مودم\modem_buyers_info.xlsx")

train_data = pd.read_csv(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\kaggle\Titanic\train.csv")
test_data = pd.read_csv(r"C:\Users\fafar\OneDrive\Desktop\Desktop\PHD\kaggle\Titanic\test.csv")


data = train_data.copy()

#Defining Label and model name
Label = "Survived"

#Selecting model
model_name = "RandomForest"




def Missing_values_Handler(reference_data, target_data, column_name):

    """
    :param reference_data: The null values are fill based on the values of this data set
    :param target_data: Target dataset which should be filled
    :param column_name: The desire column which should be cleaned
    :return: This function doesn't return anything and fills missing values inplace
    """

    if not (is_numeric_dtype(data[column_name])):
        #Filling categorical with mode of them
        target_data[column_name].fillna(reference_data[column_name].value_counts().idxmax(), inplace=True)
    else:
        # Filling numeric with mean of them
        target_data[column_name].fillna(reference_data[column_name].mean(), inplace=True)



def Outlier_Handler(data, column_name):

    """
    :param data: Entire data set for using other features in cleaning outliers
    :param column_name: The desire column which should be cleaned
    :return: This function returns the cleand data with reduced rows
    """
    # This part checks the existence of outlier through the IQR method
    IQR_outliers = []
    sorted_data = sorted(data[column_name])
    q1 = np.percentile(sorted_data, 25)
    q3 = np.percentile(sorted_data, 75)
    IQR = q3 - q1
    lwr_bound = q1 - (1.5 * IQR)
    upr_bound = q3 + (1.5 * IQR)
    for i in data[column_name]:
        if ((i < lwr_bound) or (i > upr_bound)):
            IQR_outliers.append(i)

    # This part checks the existence of outlier through the Zscore method
    Zscore_outliers = []
    lcl = data[column_name].mean() - 2.64 * data[column_name].std()
    ucl = data[column_name].mean() + 2.64 * data[column_name].std()

    for i in data[column_name]:
        if ((i < lcl) or (i > ucl)):
            Zscore_outliers.append(i)

    non_outliers = []
    for i in range(0, len(data[column_name])):
        if not ((data[column_name].iloc[i] in IQR_outliers) and (data[column_name].iloc[i] in Zscore_outliers)):
            non_outliers.append(i)

    data = data.iloc[non_outliers]
    return data


def classifier_model(data, model_name):

    """
    :param data: The entire training dataset
    :param model_name: The desire model for classification
    :return: accuracy, precision, recall, f1, confusion_matrix, model, Encoders, columns_name, Scaler
    """

    Encoders = {}
    for i in data.columns:
        if i != Label:

            # Check count of null and distinct values. If a column has alot of null and distinct
            # value, it should be removed from modeling. In addition to that ID column should be removed.

            null_percentage = data[i].isna().sum() / len(data[i])
            diversification = (len(train_data[i].unique()) + train_data[i].isna().sum())/ len(data[i])

            if (is_string_dtype(data[i]) or is_categorical_dtype(data[i])) and diversification > 0.5:
                del data[i]

            elif null_percentage >= 0.5:
                del data[i]

            elif is_integer_dtype(data[i]) and diversification == 1:
                del data[i]

            else:
                # Fill missing values
                if not(is_numeric_dtype(data[i])):
                    Missing_values_Handler(data, data, i)

                # Encoding categorical values
                if(is_string_dtype(data[i]) or is_categorical_dtype(data[i])):

                    label_encoder = LabelEncoder()
                    label_encoder.fit(data[i])
                    le_dict = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
                    data[i] = data[i].apply(lambda x: le_dict.get(x, '<Unknown>'))
                    Encoders[i] = le_dict


                # Filling missing values
                if (is_numeric_dtype(data[i])):
                    Missing_values_Handler(data, data, i)
                    data = Outlier_Handler(data, i)
        else:
            label_encoder = LabelEncoder()
            label_encoder.fit(data[i])
            le_dict = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            data[i] = data[i].apply(lambda x: le_dict.get(x, '<Unknown>'))
            Encoders[i] = le_dict


    # Fitting Scalers to data
    data.reset_index(inplace = True, drop=True)
    scaled_data = data.drop([Label], axis=1)
    column_name = scaled_data.columns
    sc = StandardScaler()
    Scaler = sc.fit(scaled_data)
    scaled_data = sc.transform(scaled_data)
    scaled_data = pd.DataFrame(scaled_data)

    # pca = PCA()
    # pca = pca.fit(scaled_data)
    # scaled_data = pca.transform(scaled_data)
    # scaled_data = pd.DataFrame(scaled_data)

    scaled_data.columns = column_name
    data = pd.concat([data[Label], scaled_data] , axis=1)

    # Data augmentation
    ratio_vec = np.zeros((len(data[Label].unique()) , 3))
    ratio_vec = pd.DataFrame(ratio_vec)
    ratio_vec.columns = ["label", "count", "rate"]

    for i in range(0, len(data[Label].unique())):

        ratio_vec["count"].iloc[i] = len(data[data[Label] == data[Label].unique()[i]])
        ratio_vec["rate"].iloc[i] = len(data[data[Label] == data[Label].unique()[i]]) / len(data)
        ratio_vec["label"].iloc[i] = data[Label].unique()[i]


    if (ratio_vec["count"].max() / ratio_vec["count"].min()) >= 4:
        for i in range(0, len(ratio_vec)):
            count = round((1/ratio_vec["rate"].iloc[i]) * ratio_vec["count"].iloc[i]) - ratio_vec["count"].iloc[i]
            resampled_data = data[data[Label] == ratio_vec["label"].iloc[i]].sample(n=int(count), replace=True)
            data = data.append(resampled_data)


    # Spliting dataset to train and test
    train, test = train_test_split(data, test_size= 0.3)
    number_of_classes = len(data[Label].unique())

    # Fitting model to training data
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

    elif (model_name == "DeepLearning"):

        # define the keras model
        y = to_categorical(train[Label], num_classes=number_of_classes)
        number_of_columns = train.drop([Label], axis= 1).shape[1]

        model = Sequential()
        model.add(Dense(number_of_columns * 2, input_shape=(number_of_columns,), activation='relu'))
        model.add(Dense(int((number_of_columns * 2 + number_of_classes)/2), activation='relu'))
        model.add(Dense(number_of_classes, activation="softmax"))
        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(train.drop([Label], axis= 1), y, epochs = 50, batch_size=10)
        # evaluate the keras model
        _, accuracy = model.evaluate(train.drop([Label], axis= 1), y)

        prediction = model.predict(test.drop([Label], axis= 1))
        predictions = np.argmax(prediction, axis=1)



    # Calculating accuracy for training dataset
    accuracy = accuracy_score(test[Label], predictions)
    if number_of_classes == 2:
        precision = precision_score(test[Label], predictions)
        recall = recall_score(test[Label], predictions)
        f1 = f1_score(test[Label], predictions)
    else:
        precision = np.nan
        recall = np.nan
        f1 = np.nan

    columns_name = data.drop([Label], axis=1).columns


    test.reset_index(inplace = True, drop=True)

    confusion = confusion_matrix(test[Label], predictions)


    return accuracy, precision, recall, f1, confusion, model, Encoders, columns_name, Scaler


def classifier_predictor(model, Label, train_data, test_data, Encoders, Scaler, columns_name, model_name):

    """
    :param model: The Fitted model for classification
    :param Label: The column name which includes lables
    :param test_data: The final data for classification
    :param Encoders: Encoders dictionary
    :param Scaler: Scaler dictionary
    :param pca: PCA dictionary
    :param columns_name: The remaining columns name in training section
    :param model_name: Name of the selected model for classification
    :return: predictions_matrix
    """

    # Selecting remaining columns based on training process
    test_data = test_data[columns_name]


    for i in test_data.columns:
        if (is_numeric_dtype(test_data[i])):
            Missing_values_Handler(train_data, test_data, i)
        else:
            Missing_values_Handler(train_data, test_data, i)

        # Applying encoders to test data
        if (is_string_dtype(test_data[i]) or is_categorical_dtype(test_data[i])):

            test_data[i] = test_data[i].apply(lambda x: Encoders[i].get(x, -1))

    # Fitting Scalers and PCA
    test_data = Scaler.transform(test_data)
    test_data = pd.DataFrame(test_data)
    # test_data = pca.transform(test_data)
    # test_data = pd.DataFrame(test_data)
    test_data.columns = columns_name

    # Predicting first and second classes with their related probabilities
    if (model_name == "DeepLearning"):
        prediction = model.predict(test_data)
        predictions = np.argmax(prediction, axis=1)
        predictions = pd.DataFrame(predictions)
        predictions_matrix = predictions
        predictions_matrix.columns = ["Label"]

        inv_Label = {v: k for k, v in Encoders[Label].items()}
        predictions_matrix["Label"] = predictions_matrix["Label"].apply(lambda x: inv_Label.get(x, -1))


    elif (model_name == "LinearSVC"):
        predictions = model.predict(test_data)
        predictions = pd.DataFrame(predictions)
        predictions_matrix = predictions
        predictions_matrix.columns = ["Label"]
        inv_Label = {v: k for k, v in Encoders[Label].items()}
        predictions_matrix["Label"] = predictions_matrix["Label"].apply(lambda x: inv_Label.get(x, -1))

    else:
        predictions = model.predict(test_data)

        probabilities = model.predict_proba(test_data)

        probabilities_matrix = np.zeros((len(probabilities), 3))
        probabilities_matrix = pd.DataFrame(probabilities_matrix)
        probabilities_matrix.columns = ["first_probability", "Second_class","second_probability"]

        for i in range(0, len(probabilities)):
            probabilities_matrix["first_probability"].iloc[i] = sorted(probabilities[i], reverse=True)[0]
            probabilities_matrix["second_probability"].iloc[i] = sorted(probabilities[i], reverse=True)[1]
            probabilities_matrix["Second_class"].iloc[i] = np.argsort(probabilities[i])[-2]

        predictions_matrix = np.zeros((len(predictions), 4))
        predictions_matrix = pd.DataFrame(predictions_matrix)
        predictions_matrix.columns = ["Label", "first_probability", "Second_class", "second_probability"]
        predictions_matrix["Label"] = predictions
        predictions_matrix["first_probability"] = probabilities_matrix["first_probability"]
        predictions_matrix["second_probability"] = probabilities_matrix["second_probability"]
        predictions_matrix["Second_class"] = probabilities_matrix["Second_class"]

        inv_Label = {v: k for k, v in Encoders[Label].items()}
        predictions_matrix["Label"] = predictions_matrix["Label"].apply(lambda x: inv_Label.get(x, -1))
        predictions_matrix["Second_class"] = predictions_matrix["Second_class"].apply(lambda x: inv_Label.get(x, -1))

    return predictions_matrix


accuracy, precision, recall, f1, confusion, model, Encoders, columns_name, Scaler = classifier_model(data, model_name)
prediction = classifier_predictor(model, Label, train_data, test_data, Encoders, Scaler, columns_name, model_name)
print("Finish")
