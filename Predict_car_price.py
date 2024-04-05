import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import warnings

df = pd.read_csv('car data.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isna().sum())
df['Age'] = 2021 - df['Year']
df.drop('Year', axis=1, inplace=True)
print(df.head())
df.rename(columns = {'Selling_Price':'Selling_Price(lacs)', 'Present_Price':'Present_Price(lacs)', 'Owner':'Past_Owners'}, inplace=True)
print(df.head())
print(df.columns)

cat_cols = ['Fuel_Type', 'Seller_Type', 'Transmission', 'Past_Owners']
i = 0
while i < 4:
    fig = plt.figure(figsize=[10,4])
    
    plt.subplot(1,2,1)
    sns.countplot(x=cat_cols[i], data=df)
    i +=1
    
    plt.subplot(1,2,2)
    sns.countplot(x=cat_cols[i], data=df)
    i += 1
    
    plt.show()

print(df[df['Present_Price(lacs)']>df['Present_Price(lacs)'].quantile(0.99)])
print(df[df['Selling_Price(lacs)'] > df['Selling_Price(lacs)'].quantile(0.99)])
print(df[df['Kms_Driven'] > df['Kms_Driven'].quantile(0.99)])
sns.heatmap(df.corr(), annot = True, cmap='RdBu')
plt.show()
print(df.corr()['Selling_Price(lacs)'])
print(df.pivot_table(values='Selling_Price(lacs)', index = 'Seller_Type', columns='Fuel_Type'))
print(df.drop(labels='Car_Name', axis=1, inplace = True))
print(df.head())
df = pd.get_dummies(data=df, drop_first=True)
print(df.head())
X = df.iloc[:,1:].values
y = df.iloc[:,:1].values
print(X)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X.shape, X_train.shape, X_test.shape)
print(y.shape, y_train.shape, y_test.shape)
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1)) # O/P

model.compile(optimizer='rmsprop', loss='mse')
model.fit(X_train, y_train, epochs=100, validation_data=(X_test,y_test))
model.summary()
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
model.evaluate(X_test, y_test)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
r2_train = r2_score(y_train, train_pred)
print("R Squared value of train dataL: ",r2_train)
r2_test = r2_score(y_test, test_pred)
print("R Squared value of test data:", r2_test)
diff_r2_scores = r2_train - r2_test
print("Difference between two scores: ", diff_r2_scores.round(2))