## Random Forest Algorithm

"""
https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/#:~:text=Random%20forest%20is%20a%20Supervised,average%20in%20case%20of%20regression.
https://machinelearningmastery.com/implement-random-forest-scratch-python/

- Supervised ML
- Classification + Regression
    - Continuous Variables and Categorical Variables

- Bagging (Bootstrap Aggregation): models choose a random sample
    from the data set, final output is a majority vote
    from the results of all models
    - reduces high variance in decision trees
    - look out for similar split points (greedy algorithm)

- Sampling with replacement:  the same row may be chosen and added more than once to the sample dataset

- How to split the tree:
    - evaluate split points
    - Compare Gini Index (probability of a variable being wrongly classified when it is randomly chosen)
        - 0 :all elements belong to a certain class
        - 1 :elements are randomly distributed across classes
    - Information Gain - reduce the level of entropy from starting node to leaf nodes
- Trees are diverse
- Can be parallelized
- 30% of data will not be seen by the decision tree (test)
 """


## Step 1:
"""
n number of random records are taken from the data set
having k number of records 

"""

## Step 2:
"""
Individual Decision trees are constructed for each sample
"""

## Step 3:
"""
Each decision tree will generate an output
"""

## Step 4:
"""
Majority Voting or Averaging is used for the final output

"""



# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt

## Code for opening dataset and evaluating performance (k-fold)

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
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
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
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

'''

Random Forest Code Begins

'''


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini
 
# Select the best split point for a dataset
def get_split(dataset, n_features):
    # Last value in each row tells us the classification
	class_values = list(set(row[-1] for row in dataset))

	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()

    # Make a list of all the features (randomly appended)
    # For each feature index (y) + row (x) combo, test spliting and record best gini index
	while len(features) < n_features:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value (aggregate potential classifications given the subset of rows, then choose the one that appears the most)
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split (why would this happen??)
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return

	# check that max_depth hasn't been reached
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	
    # process left child list, if its greater than min, make a new split point
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	
    # process right child, if its greater than min, make a new split point
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)

# Build a decision tree
"""
Find an ideal split point for the subsample of data
Use the new root node to make the rest of the decision tree,
more split point will be needed (assuming the tree isn't extremely small)
"""
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root

# Build a subsample of the dataset (with replacement)
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset)*ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# Predict Decision Tree
"""
Move through decision tree until you hit a leaf node
if index < value: move left, else: move right
"""
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Bootstrap Aggregation (Bagging) Prediction
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    # Majority Vote of the output for each row (for classification problem)
    return max(set(predictions), key=predictions.count)

# Random Forest Algorithm (Main)
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        # Take a sample from the training dataset
        sample = subsample(train, sample_size)
        # Build decision tree with efficient split points
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    # Use boostrap aggregation to predict test data
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)

if __name__ == '__main__':
    # Test the random forest algorithm
    seed(2)
    # load and prepare data
    filename = 'sonar.all-data.csv'
    dataset = load_csv(filename)
    # convert string attributes to integers
    for i in range(0, len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # evaluate algorithm
    n_folds = 5
    max_depth = 10 # depth of decision tree
    min_size = 1 # number of rows that can be aggregated to one leaf node (determining classification)
    sample_size = 1.0
    n_features = int(sqrt(len(dataset[0])-1))
    for n_trees in [1, 5, 10, 20, 50]:
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        print('\n\n')
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))