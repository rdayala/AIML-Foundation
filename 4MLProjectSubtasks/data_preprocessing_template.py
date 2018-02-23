# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# --------------------------------------------------------------------------------------------------------

# Raghu New template

# https://www.kaggle.com/rdayala/data-preprocessing-in-python/

# Importing the libraries
# --------------------------------------
import numpy as np # library that contains mathematical tools.
import pandas as pd # library to import and manage data sets
import matplotlib.pyplot as plt # library to help us plot nice charts

%matplotlib inline

# Importing the dataset
# ---------------------------------
# this data file has some missing values for few of the attributes/columns
# missing values are shown as NaN
dataset = pd.read_csv("../input/datawithusers/Data.csv")

# matrix of features
X = dataset.iloc[:, :-1].values

# vector of dependent variable
y = dataset.iloc[:, 3].values

# Taking care of missing data (OPTIONAL SECTION)
# ---------------------------------------------------
# Imputer class - which will allow us to take care of missing data
# default, missing values have NaN
# we are going to replace the missing values with the mean of that column
# axis = 0 -> impute along columns
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# we only want the imputer to fit on those feature columns that have missing data
imputer = imputer.fit(X[:, 1:3])
# use the imputer object to transform the dataset
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data (OPTIONAL SECTION)
# ---------------------------------------------------
# LabelEncoder only encodes the values without bothering if there is ordering or not.
# OneHotEncoder class is used to create dummy variables.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# create an object of LabelEncoder class
labelencoder_X = LabelEncoder()
# we are fitting the labelencoder object to the first column of X that is Country column here
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# when encoding categorical data, it may happen that the mathematical model can think that 
# categorical data with value 1 > value 2 which doesn't make sense. 
# They are just categories with numbering. There is no relational order between them, atleast in case of Country column here.
# we have to avoid this?? How ?? Using Dummy columns
# for one column -> we will have # of dummy columns equal to # of categories
# we need to specify the index of the columns that have categorical data
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# label encoding for dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
# ---------------------------------------------------------------
# we want to import the cross-validation library
from sklearn.model_selection import train_test_split
# X_train -> training part of matrix of features
# X_test -> test part of matrix of features
# y_train -> training part of dependent variable vector
# y_test -> test part of dependent variable vector
# (X_train, y_train) are associated; similarly, (X_test, y_test) are associated
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature Scaling
# ---------------------------------------------------------
from sklearn.preprocessing import StandardScaler
# scale matrix of features
sc_X = StandardScaler()
# Here, we are scaling dummy variables as well.
X_train = sc_X.fit_transform(X_train)
# for test set, we don't need to do fit. Because, the scaler is already fit on training data.
X_test = sc_X.transform(X_test)

# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)

# Question? Do we need to fit and transform dummy variables?? It depends on the context.
# It depends on how much you want to keep interpretation in your models. If we scale dummy variables, it will be good
# because everything will be on the same scale. But, we will lose the interpretation of knowing which observation belongs to which country, etc.
# But yes, models won't break if we don't scale dummy variables. They are already scaled between 0 or 1.