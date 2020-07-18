import pandas as pd
import numpy as np
data = pd.read_csv("heart.csv")
x = data.iloc[:,:-1].values
y = data['target']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_train,y_train))

import pickle
pickle.dump(lr,open('model.pickle','wb'))