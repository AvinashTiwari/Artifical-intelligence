"""
List of feature engineering steps
=================================
This script is intended as an example of how to write a custom pipeline
and does not reproduce entirely the engineering procedures we described in 
section 2.

This pipeline is not intended to be run as is.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler


class FeaturePreparer:
    # when we call the Feature preparer for the first time
    # we initialise it with the training data set, which is 
    # then stored as an attribute(raw_data)
    def __init__(self, raw_data: pd.DataFrame):
        self.raw_data = raw_data
        self.prepared_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.continuous = None
        self.categorical = None
        self.discrete = None
        self.encoding_dict = {}

    
    def separate_variable_types(self) -> None:
        # find categorical variables
        # this should be done based off training data, i.e., self.raw_data
        self.categorical = [
            var for var in self.raw_data.columns
            if self.war_data[var].dtype == 'O'
        ]
        print('There are {} categorical variables'.format(len(self.categorical)))

        # find numerical variables
        # this should be done based off training data, i.e., self.raw_data
        numerical = [var for var in self.raw_data.columns
                     if self.raw_data[var].dtype != 'O']
        print('There are {} numerical variables'.format(len(numerical)))

        # find discrete variables
        # this should be done based off training data, i.e., self.raw_data
        self.discrete = []
        for var in numerical:
            if len(self.raw_data[var].unique()) < 20:
                self.discrete.append(var)

        print('There are {} discrete variables'.format(len(self.discrete)))

        self.continuous = [
            var for var in numerical if
            var not in self.discrete and var not in ['Id', 'SalePrice']
        ]

    def split_data(self, *, training: bool = False) -> None:
        # if we are training for the first time, training = True, 
        # then divide into train and test
        if training:
            return
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.prepared_data, self.prepared_data.SalePrice, test_size=0.2, random_state=0)

        print(self.X_train.shape, self.X_test.shape)


    def handle_missing_values(self):
        # fills NA in all required variables
        for col in self.continuous:
            if self.prepared_data.loc[:, (col)].isnull().mean() > 0:
                # get the mean value from the training data, i.e., self.raw_data
                mean_val = self.raw_data.loc[:, (col)].mean()
                # replace it in the data to be passed to the model, i.e., self.prepared_data
                self.prepared_data[col].fillna(mean_val, inplace=True)

        # add label indicating 'Missing' to categorical variables
        for var in self.categorical:
            # replace NA in all categorical variables to be passed to the model
            self.prepared_data[var].fillna('Missing', inplace=True)


    def rare_imputation(self, *, variable):
        # find frequent labels / discrete numbers in training data
        temp = self.raw_data.groupby([variable])[variable].count() / np.float(len(self.raw_data))
        frequent_cat = [x for x in temp.loc[temp > 0.03].index.values]

        # replace the labels in data to be passed to model
        self.prepared_data[variable] = np.where(self.prepared_data[variable].isin(frequent_cat),
                                          self.prepared_data[variable],
                                     'Rare')


    def encode_categorical_variables(self, *, var, target, training: bool = False) -> None:
        if training:
        # make label to price dictionary
            self.encoding_dict[var] = self.prepared_data.groupby([var])[target].mean().to_dict()

        # encode variables
        self.prepared_data[var] = self.prepared_data[var].map(self.encoding_dict[var])


    def prepare_data(self, training: bool = False):
        # This is the method where we capture  all the parameters during training,
        # from the training set, 
        # then we will use this parameters to transform future data as we get it
        # from our API

        # if training = true, the function will store the transformed
        # data
        # if training = false, the function will only transform the passed
        # data.
        # This way we can use the class both for training and scoring, and
        # we will have the data stored for future use
        self.prepared_data = self.raw_data.copy(deep=True)

        self.separate_data_types() # captures the different types of vars
        self.handle_missing_values() # fills na

        for var in self.categorical + self.discrete:
            self.rare_imputation(variable=var) # replaces rare labels
            
            self.encode_categorical_variables(
                var=var,
                target='SalePrice',
                training=training) # encode labels in categorical vars
                                   # will store the encoding dict if training = True

        self.split_data(training=training) # splits data only for training
        
        if not training:
            if 'SalePrice' in self.prepared_data.columns:
                self.prepared_data.drop(['SalePrice'], axis=1)