from sklearn import datasets
from random import seed
from random import randrange
from math import sqrt
import numpy

# Euclidean Distance (manual)
def euclidean_distance(row1, row2):
    distance = 0.0
    # assumes last element in row is the label (not included in distance)
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Locate the most similar neighbors to the desired test_row
def get_neighbors(train, test_row, num_neigbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neigbors):
        neighbors.append(distances[i][0])
    return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neigbors):
    neighbors = get_neighbors(train, test_row, num_neigbors)
    # get all labels (output_values) from set of neighbors
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)

# Split data into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate Algorithm with Cross Validation Split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    print(folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Iris Dataset
iris = datasets.load_iris()
# print(iris.data)
# print(iris.target)
# print(iris.target_names)
labels = iris.target

i = 0
dataset = list()
for r in iris.data:
    row = list()
    row = numpy.append(r, int(labels[i]))
    dataset.append(row)
    i = i + 1

# Evaluate Algorithm
n_folds = 5
num_neigbors = 5
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neigbors)

# Sample 
sample_dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

sample_test_row = [6.8215, 1.10456, 1]

prediction = predict_classification(sample_dataset, sample_dataset[0], 3)
print('Expected %d, Got %d.' % (sample_dataset[0][-1], prediction))

prediction = predict_classification(sample_dataset, sample_test_row, 3)
print('Got %d' % (prediction))
