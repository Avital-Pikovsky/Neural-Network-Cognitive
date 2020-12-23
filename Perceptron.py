import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt

#reading the csv file
data = pd.read_csv("wpbc.data", header = None)
temp_df = []
for row in data.itertuples(index=False):
    if row[1] == 'R':
        temp_df.extend([list(row)]*3)
    else:
        temp_df.append(list(row))

data = pd.DataFrame(temp_df)
counter =0
sum=0
for i in range(0,data.shape[0]):
    if data.iloc[i,34] == '?':
         counter+=1
    else:
        val = int(data.iloc[i,34]) 
        sum += val

avg=sum/(sum-counter)

for i in range(0,data.shape[0]):
    if data.iloc[i,34] == '?':
        data.iloc[i,34] = avg

#Encode the y_label
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()

#split into features and y label
data_x = data.iloc[:,3:35].values
data_y = data.iloc[:,1].values
data_y = le.fit_transform(data_y)
data_y = data_y.reshape(data_y.shape[0],1)
data_x[:,31] = data_x[:,31].astype(int)

#split into test and train
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.33)


class Perceptron(object):

    def __init__(self, no_of_inputs, number_of_runs=100, learning_rate=0.001):
        self.threshold = number_of_runs
        self.learning_rate = learning_rate
        self.weights = np.zeros((no_of_inputs + 1),dtype = float)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        errors = []

        for _ in range(self.threshold):
            error = 0.0
            cost = 0.0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error =  (label - prediction) 
                cost = cost + error
                self.weights[1:] = self.weights[1: ]+ (self.learning_rate * (label - prediction) * inputs)
                self.weights[0] = self.weights[0] + (self.learning_rate * (label - prediction))
            errors.append(cost)
        plt.plot(errors)
        plt.title('Adaline simulation on WPBC')
        plt.xlabel('Iterations')
        plt.ylabel('Mean squared error')
       # plt.show()

Perceptron = Perceptron(32)

start = timeit.default_timer()
Perceptron.train(X_train,Y_train)
end = timeit.default_timer()

print("Time of train: ",end - start,"second")

True_positive = 0
False_positive = 0
False_negative = 0
True_negative = 0
rights,wrongs = (0,0)
for line in range(0, X_test.shape[0]):
    predict_line = Perceptron.predict(X_test[line,:])
    if predict_line == Y_test[line,:]:
        rights += 1
    else:
        wrongs += 1
    if(predict_line == 1 == (Y_test[line,:])):
        True_positive += 1
    if(predict_line == 0 == (Y_test[line,:])):
        True_negative += 1
    if(predict_line == 0 and (Y_test[line,:] == 1)):
        False_negative += 1  
    if(predict_line == 1 and (Y_test[line,:] == 0)):
        False_positive += 1 

print("True Positive - ", True_positive)
print("True Negative - ", True_negative)
print("False Positive - ", False_positive)
print("False Negative - ", False_negative)

print("The model accuary is: ",rights/X_test.shape[0]*100,"%")
