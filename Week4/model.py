# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

def read_input(dir):
    x, y = np.empty(0), np.empty(0)
    with open(dir, newline = '\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        for row in csvreader:
            x = np.append(x, float(row[0]))
            y = np.append(y, float(row[1]))
    return x, y

model = LinearRegression()

#Fitting model with trainig data
model.fit(read_input("./dataset/A"))

# Saving model to disk
with open('model.pkl','wb') as f:
    pickle.dump(model, f)

