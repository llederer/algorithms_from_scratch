import numpy as np
from matplotlib import pyplot
from math import sqrt
# Utilizes tutorial from: https://machinelearningmastery.com/adam-optimization-from-scratch/ 
# Based off my gradientdescent.py code 


# Implementing Adam Optimization
## Adaptive Movement Estimation
## Gradient descent with adaptive learning rates for each input variable
## Original Paper: https://arxiv.org/abs/1412.6980

# Takes the partial derivative of the target function for each input variable

# NOTE: some initialization bias because the first and second moment are initialized to 0

# Requires maintaining the following additional info:
## First moment of the gradient (exponentially decaying mean gradient)
## Second moment (variance for each input variable)


def adam(objective, derivative, bounds, num_iter, alpha, beta1, beta2, eps=1e-8):
    # Step 1: Starting Point X (defined within the bounds of the target function's input space)
    # assumes bounds --> one row for each dimension with first column defines the minimum and second column defines the maximum of the dimension
    solutions = list()

    x = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])

    # Step 1b: Initialize the first and second moments to 0
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]

    for t in range(num_iter):
        # Step 2: Calculate Derivative
        g = derivative(x[0], x[1])
        
        # Step 3: Adam update calculations
        ## Suggest using NumPy vector operations for efficiency
        for i in range(x.shape[0]):
			# m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
			# v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
			# mhat(t) = m(t) / (1 - beta1(t))
            mhat = m[i] / (1.0 - beta1**(t+1))
			# vhat(t) = v(t) / (1 - beta2(t))
            vhat = v[i] / (1.0 - beta2**(t+1))
			# x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
            x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)

        # Step 4: Evaluate this Point
        score = objective(x[0], x[1])

        solutions.append(x.copy()) #track all solutions for visualization

        # Report Iteration Progress
        print('>%d f(%s) = %.5f' % (t, x, score))
    
    return solutions

def test_objective_func(x, y):
    return x**2.0 + y**2.0

def test_derivative_func(x, y):
    return np.asarray([x * 2.0, y * 2.0])

if __name__ == "__main__":

    # Given test_objective_func, lets get all the outputs of this function for a sample range
    r_min, r_max = -1.0, 1.0
    
    # Returns evenly spaced values within a given interval
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    
    # mesh grid from the axis to visualize curvature of the response surface
    x, y = np.meshgrid(xaxis, yaxis)
    
    results = test_objective_func(x, y)
    # results = test_objective_func(np.array([1, 0, 1], np.int32), np.array([2, 3, 2], np.int32))

    print(f'{results}')
    print(f'{results.shape}')

    # # Steb 1a: Define the Objective Function, Bounds, Learning Rate, and Iterations
    # bounds = np.asarray([[-1.0, 1.0]])
    # alpha = 0.1 #learning rate
    # num_iter = 30
    # solutions, scores = gradient_descent(test_objective_func, test_derivative_func, bounds, num_iter, alpha)

    # Show Mesh Grid of Objective Function
    fig = pyplot.figure()
    axis = fig.add_subplot(111, projection='3d') #fig.gca() #projection='3d') or plt.axis()
    axis = fig.gca()
    axis.plot_surface(x, y, results, cmap='jet')
    pyplot.show()


    # Run Calc
    np.random.seed(1)
    bounds = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])
    num_iter = 60
    alpha = 0.02
    beta1 = 0.8
    beta2 = 0.999
    solutions = adam(test_objective_func, test_derivative_func, bounds, num_iter, alpha, beta1, beta2)

    # Show Contour Plot of Solutions
    pyplot.contourf(x, y, results, levels=50, cmap='jet')
    solutions = np.asarray(solutions)
    pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
    pyplot.show()
