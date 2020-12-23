# Back-Propagation Neural Networks
#
import math
import random
import numpy as np
import pandas as pd
import timeit

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
data_x = preprocessing.scale(data_x)
data_y = data.iloc[:,1].values
data_y = le.fit_transform(data_y)
data_y = data_y.reshape(data_y.shape[0],1)
data_x[:,31] = data_x[:,31].astype(int)

#split into test and train
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.33)

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, number_of_inputs, number_of_hidden, number_of_outputs):
        # number of input, hidden, and output nodes
        self.number_of_inputs = number_of_inputs + 1 # +1 for bias node
        self.number_of_hidden = number_of_hidden
        self.number_of_outputs = number_of_outputs

        # activations for nodes
        self.ai = [1.0]*self.number_of_inputs
        self.ah = [1.0]*self.number_of_hidden
        self.ao = [1.0]*self.number_of_outputs

        # create weights
        self.wi = makeMatrix(self.number_of_inputs, self.number_of_hidden)
        self.wo = makeMatrix(self.number_of_hidden, self.number_of_outputs)
        # set them to random vaules
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.number_of_hidden):
            for k in range(self.number_of_outputs):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.number_of_inputs, self.number_of_hidden)
        self.co = makeMatrix(self.number_of_hidden, self.number_of_outputs)

    def update(self, inputs):
        if len(inputs) != self.number_of_inputs-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.number_of_inputs-1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.number_of_hidden):
            sum = 0.0
            for i in range(self.number_of_inputs):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.number_of_outputs):
            sum = 0.0
            for j in range(self.number_of_hidden):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.number_of_outputs:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.number_of_outputs
        for k in range(self.number_of_outputs):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.number_of_hidden
        for j in range(self.number_of_hidden):
            error = 0.0
            for k in range(self.number_of_outputs):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.number_of_hidden):
            for k in range(self.number_of_outputs):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, x_part,y_part):
        right = 0
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for p in range(0,x_part.shape[0]):
            predict = 0 
            final_weight = self.update(x_part[p])
            if(final_weight[0] >= 0.5):
                predict = 1
            else:
                predict = 0

            if(predict == y_part[p]):
                right += 1
                if(predict == 1):
                    TP += 1
                else:
                    TN += 1
            else:
                if(predict == 0):
                    FN += 1
                else:
                    FP += 1
        print("Accuarcy: ",(right/x_part.shape[0])*100,"%")
        print("True Positive: ",TP)
        print("True Negative: ",TN)
        print("False Positive: ",FP)
        print("False Negative: ",FN)


    def train(self, x_part,y_part, iterations=1000, N=0.001, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in range(0,x_part.shape[0]):
                inputs = x_part[p]
                target = y_part[p]
                self.update(inputs)
                error = error + self.backPropagate(target, N, M)
            # if i % 100 == 0:
                # print('error %-.5f' % error)



# create a network with two input, two hidden, and one output nodes
NN = NN(32, 1, 1)

start = timeit.default_timer()
NN.train(X_train,Y_train)
end = timeit.default_timer()

print("Time of train: ",end - start)

print("Result of the train data:")
NN.test(X_train,Y_train)

print("\nResult of the test data:")
NN.test(X_test,Y_test)



