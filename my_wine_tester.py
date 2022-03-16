"""
Team:
Bertalan/Ouanes
Authors:
Mazigh Ouanes - 1721035
Vithor Bertalan – 2135362
"""

from wine_testers import WineTester
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

class MyWineTester(WineTester):
    
    def __init__(self):        
        ## Creates model
        self.model = ExtraTreesClassifier(n_estimators = 500, min_samples_split=5, criterion="entropy", verbose=1)
        self.minmax = MinMaxScaler()
        self.ordinal = OrdinalEncoder()

    def train(self, X_train, y_train):
        """
        train the current model on train_data
        :param X_train: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :param y_train: 2D array of labels.
                each line is a different example.
                the first column is the example ID.
                the second column is the example label.
        """
        # TODO: entrainer un modèle sur X_train & y_train

        # Converts the model to pandas
        train_data = pd.DataFrame.from_records(X_train, columns=["id", "color", "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"])
        train_labels = pd.DataFrame.from_records(y_train, columns=["id", "quality"])
        
        ## Uses a MinMax transformation in the columns Density and Alcohol
        train_data[['density', 'alcohol']] = self.minmax.fit_transform(train_data[['density', 'alcohol']])
        ## Uses a Ordinal Transformation (converting labels into numbers) in the column Color
        train_data[['color']] = self.ordinal.fit_transform(train_data[['color']])
        
        ## Removes id column before training
        train_data = train_data.iloc[: , 1:]
        train_labels = train_labels.iloc[: , 1:]        
        
        ## Trains model
        self.model.fit(train_data, train_labels)

    def predict(self, X_data):
        """
        predict the labels of the test_data with the current model
        and return a list of predictions of this form:
        [
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            ...
        ]
        :param X_data: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 2D list of predictions with 2 columns: ID and prediction
        """

        ## Loads test data
        test_data = pd.DataFrame.from_records(X_data, columns=["id", "color", "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"])
        
        ## Transforms test data
        test_data[['density', 'alcohol']] = self.minmax.fit_transform(test_data[['density', 'alcohol']])
        test_data[['color']] = self.ordinal.fit_transform(test_data[['color']])
        
        ## Removes id column before predicting
        test_data = test_data.iloc[: , 1:]

        ## Predicts data and generates dataframe
        preds = self.model.predict(test_data.values)
        preds = [int(x) for x in preds]
        ids = np.arange((len(preds)))
        dict = {'ID': ids, 'prediction': preds}
        df = pd.DataFrame(dict) 
        
        ## Converts dataframe to list
        pred_list = df.values.tolist()

        return (pred_list)