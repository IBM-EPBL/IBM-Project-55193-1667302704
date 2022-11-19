import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('weather.data')
print(df.head())

y=df.iloc[:, 4].values
X=df.iloc[:, 0:4].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y.reshape(-1))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=10)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train,y_train)


pickle.dump(sv, open('wet.pkl', 'wb'))