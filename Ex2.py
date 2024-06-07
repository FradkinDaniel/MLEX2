import math
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    matrixD = []
    matrixY = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            columns = line.strip().split(',')
            last_column = columns.pop()
            last_column = last_column.split()[0]
            matrixD.append([float(col) for col in columns])
            matrixY.append(float(last_column))

    return addOnesColum(matrixD),matrixY

def addOnesColum(D1: list):
    for i in range(D1.__len__()):
        D1[i] = [1] + D1[i]
    return D1

def sigmoid(z):
    if isinstance(z, (int ,float)):
        return sigmoid_int(z)


    else:
        return sigmoid_matrix(z)

def sigmoid_int(z):
    return 1/(1+math.exp(-z))

def sigmoid_matrix(matrix):
    return [[sigmoid(x) for x in row] for row in matrix]

def predictValue(Example:list, Hypothesis:list):
    prediction = 0.0
    for i in range(Example.__len__()):
        prediction += Example[i] * Hypothesis[i]
    result = sigmoid(prediction)
    return result

def computeCost(predicted, realValue):
    if predicted == 0:
        predicted = 0.0001
    if predicted == 1:
        predicted = 0.999
    return -realValue * math.log(predicted) - (1.0 - realValue) * math.log(1.0 - predicted)

def compute_grandiant(predicted, example:list,  y):
    gradients = [0.0, 0.0, 0.0]
    for i in range(example.__len__()):
        gradients[i] = (predicted - y) * example[i]

    return gradients


def computeCostAndGradient(D: list, Y:list , Hypothesis:list):
    return computeRegularizedCostAndGradient(D,Y,Hypothesis,0)

def computeRegularizedCostAndGradient(D: list, Y:list , Hypothesis:list, lamda):
    J = 0.0
    gradients = [0.0, 0.0, 0.0]
    for i in range(D.__len__()):
        predicted = predictValue(D[i],Hypothesis)
        J += computeCost(predicted, Y[i])
        temp_grad = compute_grandiant(predicted, D[i], Y[i])
        for j in range(gradients.__len__()):
            gradients[j] += temp_grad[j]

    J /= D.__len__()
    regularizetionForCost = 0
    for i in range(gradients.__len__()):
        if i != 0:
            regularizetionForCost += pow(Hypothesis[i], 2)
            regularizetionForGrandiant = Hypothesis[i] * lamda
            gradients[i] += regularizetionForGrandiant
        gradients[i] /= D.__len__()
    regularizetionForCost *= lamda / (2 * D.__len__())
    J += regularizetionForCost
    return J, gradients

def updateHypothesis(Hypothesis:list, alpha, Gradient:list):
    for i in range(Hypothesis.__len__()):
        Hypothesis[i] -= alpha * Gradient[i]
    return Hypothesis

def gradientDescent(filename, alpha=0.001, max_iter=1000, threshold=0.0001):
    cost_J = float('inf')
    iter = 1
    Costs = []
    Data, Y = load_data(filename)

    Hypothesis = [-8, 2, -0.5]

    plot_decision_boundary(Hypothesis, Data, Y)
    temination_reason = ""

    while(True):
        cost, gradient = computeRegularizedCostAndGradient(Data, Y, Hypothesis, 900)

        Costs.append(cost)
        Hypothesis = updateHypothesis(Hypothesis, alpha, gradient)

        if (iter > max_iter):
            temination_reason = "Gradient descent terminating after %d iterations (max_iter)"% (iter + 1)
            break

        if len(Costs) > 1 and abs(Costs[-1] - Costs[-2]) < threshold:
            temination_reason = "Gradient descent terminating after %d iterations. Improvement was: %f – below threshold (%f)"% (iter + 1, abs(Costs[-1] - Costs[-2]), threshold)
            break

        iter+=1

    print(temination_reason)
    plot_decision_boundary(Hypothesis, Data, Y)
    return Costs, Hypothesis


def plot_data(X, y):
    X = np.array(X)  # Ensure X is a NumPy array
    y = np.array(y)  # Ensure y is a NumPy array
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k', label='Positive')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y', label='Negative')


def map_feature(x1, x2):
    degree = 6
    if x1.ndim == 0: x1 = np.array([x1])
    if x2.ndim == 0: x2 = np.array([x2])
    out = np.ones((x1.shape[0], 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            term = (x1 ** (i - j) * x2 ** j).reshape(-1, 1)
            out = np.hstack((out, term))
    return out


def plot_decision_boundary(theta, X, y):
    X = np.array(X)  # Ensure X is a NumPy array
    y = np.array(y)  # Ensure y is a NumPy array
    plot_data(X[:, 1:3], y)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    if np.all(theta == 0):
        theta = np.ones_like(theta) * 0.001

    if X.shape[1] <= 3:
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Debug prints
        print("theta:", theta)
        print("plot_x:", plot_x)
        print("plot_y:", plot_y)

        plt.plot(plot_x, plot_y, 'purple', label='Decision Boundary')
        plt.legend()
        plt.axis([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2, np.min(X[:, 2]) - 2, np.max(X[:, 2]) + 2])
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = np.dot(map_feature(np.array([u[i]]), np.array([v[j]])), theta)

        z = z.T

        # Debug print
        print("z shape:", z.shape)

        plt.contour(u, v, z, levels=[0], linewidths=2)

    plt.show()


if __name__ == '__main__':
    Costs, Hypothesis = gradientDescent("ex2data1.txt")
    plt.plot(Costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.show()
