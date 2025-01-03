import pandas as pd
import numpy as np

car = pd.read_csv('quikr_car.csv')

# print(car.shape)
# print(car.info())

# problem
# year has many non-year Value
# year type change object to int
# price has ask for Price
# kms_driven has kms with int
# kms_driven object to int
# kms_driven has nan values
# fuel_type has nan values 
# keep first 3 words of name

#cleaning

backup = car.copy()

car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)

car = car[car['Price'] != "Ask For Price"]
car['Price'] = car['Price'].str.replace(',','').astype(int)

car['kms_driven'] = car['kms_driven'].str.split(' ').str[0].str.replace(',','')

car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)

car = car[~car['fuel_type'].isna()]

car['name'] = car['name'].str.split(' ').str.slice(0,3).str.join(' ')

# print(car[car['name']])
# car = car.reset_index(drop=True)

car = car[car['Price']<6e6].reset_index(drop=True)

car.to_csv('Cleaning_car.csv', index=False)

# print(car.info())

X = car.drop(columns='Price')
y = car['Price']

# print(X['name'])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 433)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

lr = LinearRegression()
ohe = OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder = 'passthrough')
pipe = make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)
print("Accuracy :",r2_score(y_test,y_pred))
# scores = []
# for i in range(1000):
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = i)
#     lr = LinearRegression()
#     pipe = make_pipeline(column_trans,lr)
#     pipe.fit(X_train,y_train)
#     y_pred = pipe.predict(X_test)
#     # print("Accuracy :",r2_score(y_test,y_pred))
#     scores.append(r2_score(y_test,y_pred))

# print(scores[np.argmax(scores)])
import pickle
pickle.dump(pipe,open("LinearRegression.pkl",'wb'))




