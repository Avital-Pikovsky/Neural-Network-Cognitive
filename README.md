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


