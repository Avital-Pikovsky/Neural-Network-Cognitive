import numpy as np
import pandas as pd

#reading the csv file
data = pd.read_csv("wpbc.data", header = None)

for i in range(0,data.shape[0]):
    if data.iloc[i,34] == '?':
         data.iloc[i,34] = 0

for i in range(0,data.shape[0]):
    if data.iloc[i,34] == '?':
         data.iloc[i,34] = 0
# print(data)


#Encode the y_label
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()

#split into features and y label
data_x = data.iloc[:,3:35].values
data_x = preprocessing.scale(data_x)
data_y = data.iloc[:,1].values
data_y = le.fit_transform(data_y)
data_y = data_y.reshape(198,1)
data_x[:,31] = data_x[:,31].astype(int)

#split into test and train
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.33)

class Adaline(object):
 
    def __init__(self, eta=0.001, n_iter=200):
       
        self.eta = eta
        self.n_iter = n_iter

    def train(self, X, y):
 

        self.w_ = np.array(np.zeros( 1 + X.shape[1] ))
        self.w_ = self.w_[:, np.newaxis]    # Adds a new axis -> 2D array. Required to update the weights.
        self.cost_ = []

        for i in range(self.n_iter):

            output = self.activation( X )

            errors = (y - output)

        
            self.w_[1:] = np.add( np.asarray(self.w_[1:]), self.eta * X.T.dot( errors ) )

            self.w_[0] += self.eta * errors.sum()

            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
       
        return np.dot( X, self.w_[1:] ) + self.w_[0]

    def activation(self, X):
      

        return self.net_input(X)

    def predict(self, X):
     
        return np.where(self.activation(X) >= 0.5, 1, 0)

Adal = Adaline()
Adal.train(X_train,Y_train)

predicted = Adal.predict(X_test)
right = 0
wrong = 0
True_positive = 0
False_positive = 0
False_negative = 0
True_negative = 0
for i in range(0, predicted.shape[0]):
    if(predicted[i] == Y_test[i]):
        right += 1
        if(Y_test[i] == 1):
            True_positive += 1
        else:
            True_negative += 1

    if(predicted[i] != Y_test[i]):
        wrong +=1
        if(Y_test[i] == 1):
            False_negative += 1
        else:
            False_positive += 1

print("Accuracy: ",right/(right+wrong)*100,"%")
print("True positive: ",True_positive)
print("True negative: ",True_negative)
print("False positive: ",False_positive)
print("False negative: ",False_negative)