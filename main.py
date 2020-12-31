"""
Psuedo code:

"""
import csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
TEST_SIZE = 0.2

def load_data(filename):
    data = pd.read_csv(filename)
    sale_price = data['SalePrice']
    evidence = data.drop('SalePrice', 1)
    evidence = evidence.fillna(0) #get rid of any null values

    for col in evidence:
        for i in range(len(evidence[col])):
            #for simplicity just remove any features that are not numeric
            #can add more features for better results but that requires those features to be converted to numeric
            if isinstance(evidence[col][i], str):
                evidence = evidence.drop(col, 1)
                break


    return evidence, sale_price

def train_model(evidence, labels):
    model = LinearRegression().fit(evidence, labels)
    return model

def test_model(labels, predictions):
    count = 0
    for actual in labels:
        print("Prediction: ", predictions[count], " ---- actual: ", actual)
        count += 1
        if count > 10:
            break

    print("----------------------------------------------")
    print("R squared value: ", model.score(X_test, y_test))
    print("Number of feature: ", X_train.shape[1])



file = 'train.csv'

# Load data from spreadsheet and split into train and test set

evidence, labels = load_data(file)
X_train, X_test, y_train, y_test = train_test_split(
    evidence, labels, test_size=TEST_SIZE
)

# Train model and make predictions
model = train_model(X_train, y_train)
predictions = model.predict(X_test)
test_model(y_test, predictions)


