# NEURAL NETWORK COGNITIVE PROJECT:
## Data preparation:
We read the data into a variable.

The original data included some missing parts, in the form of questions marks, specifically in the last column.
The way we choose to solve this problem is to replace them by the average number of the last column.
The reason for this is that otherwise the results could be damaged.

The dataset is not ‘balanced’, it has 151 cases of Non-Recur cancer and only 47 Recur cases, meaning the results could be misleading.
To solve this problem, we duplicated all the cases of women with R result, 2 times, so in the end we have 141 Recur cases. Now we have a balanced set.
The data was splitted into 2 parts, the X part is the features, we put into it only the 3-35 columns, this way we dropped the first and third columns, which are not relevant.

Into the Y part, which is the data we want to predict, we insert column number 1 of the data, meaning the info if the cancer did recur or not.
To make it easier on the algorithms, we encoded the Y label column, where it was ‘N’ it’s now 0, and where it was ‘R’ it’s now 1.
To avoid future problems, we normalized the data, meaning we reduce all the values to much smaller numbers, but keeping the relation between them.

Using the SKLEARN library, we splitted the data to 4 parts:
X_train, Y_train, X_test, Y_test.

We put 66% of the data into the training scope, and 33% into the testing scope.
We used a special feature of the library, which allows us to randomize the testing scope, meaning that every time we run the model for training and testing, it picks randomly the test and train scope. Doing so we did a Cross Validation, and the reason for that is to really test the model, to see if the results are not good or bad on accident.



## Perceptron Algorithm:

The perceptron is a simple classification algorithm used to predict a result between 2 options, for example male or female.
The algorithm mimics the neurons activity in the brain, and for that he is considered an example of an ANN - Artificial Neural Network.
When the dataset is linear departed, the algorithm works best and classifies with high accuracy.

During the training and testing of the algorithm we noticed that the results are pretty random, and not working as they should be.
After some research we found out that the reason is that the dataset is not linearly divided, and the perceptron algorithm does not work well on that kind of dataset.

1. Time it took to train the model: ∓ 0.6 Seconds.
2. Ease of parameters : 
Learning rate - a small fixed number, used to update the weights. At first we used a 0.1
the learning rate and the result wasn’t so good. We started to go down until we stopped on 0.001, which gives us better results.
Number of iteration - The higher the number the better the results, but also higher run
time, we ended up on 100 iterations, which gave us nice results and good run time.
3. The Result: First we run the test on the train part of the data

| Train  | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 17  | False Negative: 77  |
|  Actual: no | False Positive: 4  | True Negative: 97  |

After that we did our cross validation, and here our 3 different results:

| test #1  | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 14  | False Negative: 34  |
|  Actual: no | False Positive: 4   | True Negative: 45  |

| test #2 | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 35  | False Negative: 15  |
|  Actual: no | False Positive: 21  | True Negative: 26  |

| test #3 | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 45  | False Negative: 0  |
|  Actual: no | False Positive: 41  | True Negative: 11  |

4. In conclusion, it is a simple algorithm that can work really well on linear divided algorithms. In our dataset it’s not the case, so the results are pretty random, moving around the 50% mark.

## Adaline Algorithm:

Adaline is an early single-layer artificial neural network.
The difference between Adaline and the standard (McCulloch–Pitts) perceptron is that in the learning phase, the weights are adjusted according to the weighted sum of the inputs (the net). In the standard perceptron, the net is passed to the activation (transfer) function and the function's output is used for adjusting the weights.

1. Time it took to train the model : ∓ 0.3 Seconds.
2. Ease of parameters : 
Learning rate - a small fixed number, used to update the weights. At first we used a 0.5
the learning rate and the result wasn’t so good. We started to go down until we stopped at 0.001, like we did in the perceptron, which gives us better results.
Number of iteration - The higher the number the better the results, but also higher run
time, we ended up on 200 iterations, we could do more iteration here because the run time is faster.
Weights array - we did some testing and the best results were when we completely reset the weights at the beginning , rather then initialized them to 1 or 0.5.
3. The Result: First we run the test on the train part of the data:

| Train  | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 82  | False Negative: 16  |
|  Actual: no | False Positive: 24  | True Negative: 73  |

After that we did our cross validation, and here our 3 different results, including a graph that shows the results in a form of MSE, mean squared error.
The reason for the difference between the accuracy of the training part and the accuracy of the testing part is that the weights are updated to fit the training part, so when we test it on the training part the results are better.

![image](https://user-images.githubusercontent.com/57085913/103019688-e1ebec00-454f-11eb-8fbd-9bff7e62acf4.png)

| test #1  | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 35  | False Negative: 10  |
|  Actual: no | False Positive: 16   | True Negative: 36  |

| test #2 | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 36  | False Negative: 15  |
|  Actual: no | False Positive: 14  | True Negative: 32  |

| test #3 | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 35  | False Negative: 13  |
|  Actual: no | False Positive: 15  | True Negative: 34  |

4. In conclusion, The adaline algorithm performs better than the perceptron, and starts to give us positive results. 
The accuracy is not perfect yet, but results that are higher than 70% starts to give an accurate prediction.


## Backpropagation algorithm:

In machine learning, backpropagation is a widely used algorithm for training feedforward neural networks. Generalizations of backpropagation exists for other artificial neural networks (ANNs), and for functions generally.

The algorithm uses hidden layers, a group of neurons thats store weights.
The algorithm update the weights of all the layers, in our case only 1 layer,
And go back every time, including more and more cases of women.

At the end we have a fixed weight array, that is used to determine if a prediction is classified as False or True


1. Time it took to train the model : ∓ 1 minute.
2. Ease of parameters : 
Learning rate - 0.001, maybe can be change a little bit to get better result, but the change would be minimal.
Number of iteration - The higher the number the better the results, but also higher run
time, we ended up on 1000 iterations. It took us some time to run it, but the result worth it.
Weights array - we did some testing and the best results were when we randomized them between -0.2 and +0.2.
Neurons number - number of neurons in the hidden level. We found out that the number has a big impact on the algorithm. A higher number significantly gave better results, but also slowed down the algorithm. First we tried it with 100 neurons, and the computer could not handle it. After some research we fixed it on 5 neurons.
Momentum Factor - should be low number, we put it on 0.1, but other numbers were also fine.

3. The Result: First we run the test on the train part of the data:

| Train  | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 93  | False Negative: 0  |
|  Actual: no | False Positive: 6  | True Negative: 96  |

After that we did our cross validation, and here our 3 different results, including a graph that shows the results in a form of MSE, mean squared error.

![image](https://user-images.githubusercontent.com/57085913/103020488-23c96200-4551-11eb-81d9-54e5fda416cb.png)

| test #1  | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 48  | False Negative: 0  |
|  Actual: no | False Positive: 15   | True Negative: 34  |

| test #2  | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 48  | False Negative: 3  |
|  Actual: no | False Positive: 17   | True Negative: 29  |

| test #3  | Predicted: yes  | Predicted: no  |
| ------------ | ------------ | ------------ |
| Actual: yes  | True Positive: 44  | False Negative: 0  |
|  Actual: no | False Positive: 24   | True Negative: 29  |

The reason for the difference between the accuracy of the training part and the accuracy of the testing part is that the weights are updated to fit the training part, so when we test it on the training part the results are close to 100 percent, because this algorithm can be very accurate.

4. To conclude, the back propagation algorithm is clearly the best so far and its results are very accurate.
With some improvements it can be used to test real cases and help science.
We were tasked to put only 1 hidden layer, maybe in the future if we would implement algorithm with more levels the results could be even better.


