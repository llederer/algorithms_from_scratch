import numpy as np
from matplotlib import pyplot

# Utilizes tutorial from: https://machinelearningmastery.com/gradient-descent-optimization-from-scratch/

# Implementing Gradient Descent
## A first order optimization algorithm that relies on gradient information to search for a minimum
## Only works for differentiable target functions

# Extensions to this approach
## Gradient Descent with Momentum
## Gradient Descent with Adaptive Gradients
## Stochastic Gradient Descent --> target function is an error function and the function gradient is approximate from prediction error on samples from the problem domain


# Target Function --> the function that is being optimized

# Derivative Function --> derivative of the target function for a given set of inputs

def gradient_descent(objective, derivative, bounds, num_iter, alpha):
    # Step 1: Starting Point X (defined within the bounds of the target function's input space)
    solutions, scores = list(), list()
    x = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    for i in range(num_iter):
        # Step 2: Calculate Derivative and Take a Step
        gradient = derivative(x)
        x = x - alpha * gradient

        # Step 3: Evaluate this Point
        eval = objective(x)

        # Store Solution (for graphing)
        solutions.append(x)
        scores.append(eval)

        # Report Iteration Progress
        print(f'{i+1} iteration {x} = {eval}')
    
    return [solutions, scores]

def test_objective_func(x):
    return x**2.0

def test_derivative_func(x):
    return 2.0*x

if __name__ == "__main__":

    # Given test_objective_func, lets get all the outputs of this function for a sample range
    r_min, r_max = -1.0, 1.0
    
    # Returns evenly spaced values within a given interval
    inputs = np.arange(r_min, r_max+0.1, 0.1)
    results = test_objective_func(inputs)


    # Steb 1a: Define the Objective Function, Bounds, Learning Rate, and Iterations
    bounds = np.asarray([[-1.0, 1.0]])
    alpha = 0.1 #learning rate
    num_iter = 30
    solutions, scores = gradient_descent(test_objective_func, test_derivative_func, bounds, num_iter, alpha)

    # Show Line Plot
    pyplot.plot(inputs, results)
    pyplot.plot(solutions, scores, '.-', color='red')
    pyplot.show()