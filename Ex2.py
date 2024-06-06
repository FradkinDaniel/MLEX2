import math


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
    for i in gradients:
        print(i)
    regularizetionForCost = 0
    for i in range(gradients.__len__()):
        if i != 0:
            regularizetionForCost += pow(gradients[i], 2)
            print(regularizetionForCost)
            regularizetionForGrandiant = gradients[i] * lamda
            gradients[i] += regularizetionForGrandiant
        gradients[i] /= D.__len__()
    regularizetionForCost *= lamda / (2 * D.__len__())
    J += regularizetionForCost
    for i in gradients:
        print(i)
    return J, gradients


if __name__ == '__main__':
    D, Y = load_data("ex2data1.txt")
    J,grand = computeRegularizedCostAndGradient(D, Y, [-10, 0.8, 0.08],1)
