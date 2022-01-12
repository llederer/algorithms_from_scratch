# Perceptron Algorithm
# https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/

# inspired by neuron information processing
# input -> weighted and combined -> output/prediction

# utilizes a similar concept to stochastic gradient descent to update weights are each iteration

# w = w + learning_rate * (expected - predicted) * w

# Step 1: Making Predictions (weights provided)
# Classifying as 0 or 1
def predict(row, weights):
    activation = weights[0] #bias
    for i in range(len(row) - 1):
        activation += weights[i+1] * row[i]
    if activation >= 0.0:
        return 1.0
    else:
        return 0.0

# test on fake data (X1, X2, Y)
# test predictions
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

# bias, w1, w2
# activation = (0.206 * X1) + (-0.234 * X2) + -0.1
weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
for row in dataset:
	prediction = predict(row, weights)
	print("Expected=%d, Predicted=%d" % (row[-1], prediction))


# Step 2: Adding in Stochastic Gradient Descent
## Loop over each epoch
### Loop over each row in training data
#### Loop over each weight and update it for a row in an epoch

## error = expected - predicted

# w(t+1) = w(t) + learning_rate * error(t) * x(t)
# bias(t+1) = bias(t) + learning_rate * error(t)
# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    # zero out weights and start epochs
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
            # calculate error
			error = row[-1] - prediction
			sum_error += error**2 # squared error for printing purposes
            
            # update weights using erorr and learning rate
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]

		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights
 
# Calculate weights
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
l_rate = 0.1
n_epoch = 5
weights = train_weights(dataset, l_rate, n_epoch)
print(weights)