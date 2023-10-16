import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('007 Get the dataset\\Data.csv')
print(dataset,"\n")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)

