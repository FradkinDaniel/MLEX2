import math


def load_data(filename):
    matrixD = []
    matrixY = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            columns = line.strip().split()
            last_column = columns.pop()
            matrixD.append([float(col) for col in columns])
            matrixY.append([float(last_column)])

    return addOnesColum(matrixD),matrixY
def addOnesColum(D1: list):
    for i in range(D1.__len__()):
        D1[i] = [1, D1[i]]
    return D1
def sigmoid(z):
    if z is list:
        return sigmoid_matrix(z)
    else:
        return sigmoid_int(z)
def sigmoid_int(z):
    return 1/(1+math.exp(-z))
def sigmoid_matrix(matrix):
    return [[sigmoid(x) for x in row] for row in matrix]
def predictValue(Example, Hypothesis):
    prediction = 0
    for i in range(Example.__len__()):
        prediction += Example[i] * Hypothesis[i]
    resulte =sigmoid(prediction)
    if resulte==0:
        resulte += 0.000000001
    if resulte==1:
        resulte -= 0.000000001
    return resulte
def computeCost(example:list ,prediction,Hypothesis):
    return  -prediction * math.log(predictValue(example,Hypothesis)) - (1 - prediction) * math.log(1 - predictValue(example,Hypothesis))
def compute_grandiant()

def computeCostAndGradient(D: list, Y:list , Hypothesis:list):
    error = list
    for i in range(D.):
        predictValue(row,Hypothesis)

if __name__ == '__main__':
