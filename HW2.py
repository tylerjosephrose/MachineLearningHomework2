import sys
import random
try:
    import sympy
    from sympy import pprint, init_printing, Sum, lambdify
    from sympy.abc import k
    from operator import truediv
    # The below are used for plotting
    import numpy as np
    from numpy import linspace, matrix
    import matplotlib.pyplot as mpl
except ImportError:
    sys.exit("""You need sympy, numpy, and matplotlib! Install these by running:
                pip install sympy
                pip install numpy
                pip install matplotlib""")

init_printing()

def Sum(matrix):
    sum = matrix[0, 0]
    size = matrix.shape[0]
    for row in range(1, size):
        sum = sum + matrix[row, 0]
    return sum

def meanColumn(column):
    return Sum(column)/len(column)

def stdColumn(column, mean):
    length = len(column)
    sum = 0
    for row in range(length):
        sum += (column[row] - mean)**(2)
    return ((1/length)*sum)**(1/2)

square = lambda x: x*x

# imported dataset
# lines = [line.rstrip('\n') for line in open("hm2Data.csv")]
lines = [line.rstrip('\n') for line in open("hm2Data2.csv")]
m = len(lines)
x = sympy.Matrix.zeros(m, 3)
y = sympy.Matrix.zeros(m, 1)
for i in range(m):
    splitString = lines[i].split(',')
    x[i,0] = splitString[0]
    x[i,1] = splitString[1]
    x[i,2] = splitString[2]
    y[i] = splitString[3]

# Calculate mean and std dev.
meanVector = sympy.Matrix.zeros(3, 1)
stdVector = sympy.Matrix.zeros(3, 1)
for j in range(3):
    meanVector[j] = meanColumn(x.col(j))
    stdVector[j] = stdColumn(x.col(j), meanVector[j])

# Expand the mean and std dev matricies - sympy is dumb
expandedMean = sympy.Matrix.zeros(m, 3)
expandedStd = sympy.Matrix.zeros(m, 3)
for k in range(m):
    expandedMean[k,0] = meanVector[0]
    expandedMean[k,1] = meanVector[1]
    expandedMean[k,2] = meanVector[2]
    expandedStd[k,0] = stdVector[0]
    expandedStd[k,1] = stdVector[1]
    expandedStd[k,2] = stdVector[2]

# Normalizing sympy matricies with entry by entry :(
standardizedX = sympy.Matrix(m, 3, list(map(truediv,(x-expandedMean),expandedStd)))

# m is the number of training samples
m = x.shape[0]

# Start L2 Regularization here

# extend the data set by the bias column: 
#       Each row receives a 1 in front since this is linear regression we want a 1 in front 
#       for the b part of mx + b
# we will call this ex for extended x
ex = sympy.Matrix(standardizedX)
cols = sympy.Matrix.ones(m, 1)
ex = ex.col_insert(0, cols)

# next we need to split our data in half into training and test seta
trainingX = sympy.Matrix()
trainingY = sympy.Matrix()
testingX = sympy.Matrix()
testingY = sympy.Matrix()
for i in range(97) :
    if random.randint(0, 1) == 1 and trainingX.shape[0] < 49 :
        trainingX = trainingX.row_insert(-1, ex.row(i))
        trainingY = trainingY.row_insert(-1, y.row(i))
    elif testingX.shape[0] < 48 :
        testingX = testingX.row_insert(-1, ex.row(i))
        testingY = testingY.row_insert(-1, y.row(i))
    else :
        trainingX = trainingX.row_insert(-1, ex.row(i))
        trainingY = trainingY.row_insert(-1, y.row(i))

# since we are only using half of the dataset for training,
# we will reset m to the size of the training set
m = trainingX.shape[0]
testingError = 100
babyShep = 5
count = 0
while(babyShep > 0):

    # now we need to set up the weight vector. we will use sympy for this as
    # we want symbolics so we can do a gradient later on
    w0, w1, w2, w3 = sympy.symbols('w0, w1, w2, w3')
    w = sympy.Matrix([w0, w1, w2, w3])


    # Define now the linear hypothesis
    hx = sympy.Matrix.zeros(m, 1)
    for row in range(m):
        hx[row,:] = trainingX[row,:]*w[:,:]

    # now we define the error function
    jw = Sum((hx - trainingY).applyfunc(square)) + babyShep*Sum(w.applyfunc(square))

    grad0 = sympy.Derivative(jw, w0).doit()
    grad1 = sympy.Derivative(jw, w1).doit()
    grad2 = sympy.Derivative(jw, w2).doit()
    grad3 = sympy.Derivative(jw, w3).doit()

    solution = sympy.solve([grad0, grad1, grad2, grad3], dict=True)
    w0 = solution[0][w0]
    w1 = solution[0][w1]
    w2 = solution[0][w2]
    w3 = solution[0][w3]
    wSolved = sympy.Matrix([w0, w1, w2, w3])

    htest = sympy.Matrix.zeros(48, 1)
    for row in range(48):
        htest[row,:] = testingX[row,:]*wSolved[:,:]
    temp = testingError
    testingError = Sum((htest - testingY).applyfunc(square))/48
    print("lambda: %f\n\tError: %f" % (babyShep, testingError))
    if temp < testingError :
        break
    babyShep -= .2
    count+=1

# print(sympy.N(hx))
print("w0: %f" % (w0))
print("w1: %f" % (w1))
print("w2: %f" % (w2))
print("w3: %f\n" % (w3))
minimum = min(abs(w0), abs(w1), abs(w2), abs(w3))

# Now we will remove the attribute that has the smallest effect on the outcome
# this is determined by whichever attribute has the smallest w value
reducedX = sympy.Matrix(ex)
if minimum == abs(w0) :
    print("Removing attribute 0 from data as it has the smallest impact...")
    reducedX.col_del(0)
elif minimum == abs(w1) :
    print("Removing attribute 1 from data as it has the smallest impact...")
    reducedX.col_del(1)
elif minimum == abs(w2) :
    print("Removing attribute 2 from data as it has the smallest impact...")
    reducedX.col_del(2)
elif minimum == abs(w3) :
    print("Removing attribute 3 from data as it has the smallest impact...")
    reducedX.col_del(3)

# we will make the modified quadratic set (adds squared of each value as attribute)
reducedQuadX = reducedX.col_insert(3, reducedX.col(1).applyfunc(square))
reducedQuadX = reducedQuadX.col_insert(4, reducedX.col(2).applyfunc(square))

# Cross Validation
# Split our data into 3 parts (for both linear and quadratic run)
redX1 = sympy.Matrix()
redY1 = sympy.Matrix()
redX2 = sympy.Matrix()
redY2 = sympy.Matrix()
redX3 = sympy.Matrix()
redY3 = sympy.Matrix()

redQuadX1 = sympy.Matrix()
redQuadX2 = sympy.Matrix()
redQuadX3 = sympy.Matrix()

for i in range(97) :
    rand = random.randint(0,2)
    if rand == 0 and redX1.shape[0] < 33 :
        redX1 = redX1.row_insert(-1, reducedX.row(i))
        redY1 = redY1.row_insert(-1, y.row(i))
        redQuadX1 = redQuadX1.row_insert(-1, reducedQuadX.row(i))
    elif rand == 1 and redX2.shape[0] < 32 or rand == 0 and redX1.shape[0] == 33 and redX2.shape[0] < 33:
        redX2 = redX2.row_insert(-1, reducedX.row(i))
        redY2 = redY2.row_insert(-1, y.row(i))
        redQuadX2 = redQuadX2.row_insert(-1, reducedQuadX.row(i))
    elif redX3.shape[0] < 32 :
        redX3 = redX3.row_insert(-1, reducedX.row(i))
        redY3 = redY3.row_insert(-1, y.row(i))
        redQuadX3 = redQuadX3.row_insert(-1, reducedQuadX.row(i))
    elif redX1.shape[0] < 33 :
        redX1 = redX1.row_insert(-1, reducedX.row(i))
        redY1 = redY1.row_insert(-1, y.row(i))
        redQuadX1 = redQuadX1.row_insert(-1, reducedQuadX.row(i))
    else :
        redX2 = redX2.row_insert(-1, reducedX.row(i))
        redY2 = redY2.row_insert(-1, y.row(i))
        redQuadX2 = redQuadX2.row_insert(-1, reducedQuadX.row(i))

dataX = [redX1, redX2, redX3]
dataY = [redY1, redY2, redY3]
dataQuadX = [redQuadX1, redQuadX2, redQuadX3]

# Find the testing error for linear k-fold cross validation
LinearSumError = 0
for i in range(3) :
    # now we need to set up the weight vector. we will use sympy for this as
    # we want symbolics so we can do a gradient later on
    w0, w1, w2 = sympy.symbols('w0, w1, w2')
    w = sympy.Matrix([w0, w1, w2])

    # Define now the linear hypothesis
    htrain = sympy.Matrix.zeros(dataX[i].shape[0], 1)
    for row in range(dataX[i].shape[0]):
        htrain[row,:] = dataX[i][row,:]*w[:,:]

    # now we define the error function
    jw = Sum((htrain - dataY[i]).applyfunc(square))

    grad0 = sympy.Derivative(jw, w0).doit()
    grad1 = sympy.Derivative(jw, w1).doit()
    grad2 = sympy.Derivative(jw, w2).doit()

    solution = sympy.solve([grad0, grad1, grad2, grad3], dict=True)
    w0 = solution[0][w0]
    w1 = solution[0][w1]
    w2 = solution[0][w2]
    wSolved = sympy.Matrix([w0, w1, w2])

    testlength = 97 - dataX[i].shape[0]
    testDataX = sympy.Matrix()
    testDataY = sympy.Matrix()
    if i != 0 :
        for j in range(dataX[0].shape[0]) :
            testDataX = testDataX.row_insert(-1, dataX[0].row(j))
            testDataY = testDataY.row_insert(-1, dataY[0].row(j))
    if i != 1 :
        for j in range(dataX[1].shape[0]) :
            testDataX = testDataX.row_insert(-1, dataX[1].row(j))
            testDataY = testDataY.row_insert(-1, dataY[1].row(j))
    if i != 2 :
        for j in range(dataX[2].shape[0]) :
            testDataX = testDataX.row_insert(-1, dataX[2].row(j))
            testDataY = testDataY.row_insert(-1, dataY[2].row(j))

    htest = sympy.Matrix.zeros(testlength, 1)
    for row in range(testlength):
        htest[row,:] = testDataX[row,:]*wSolved[:,:]
    testingError = Sum((htest - testDataY).applyfunc(square))/testlength
    LinearSumError = LinearSumError + testingError
    print("testing linear error is: %f" % testingError)
LinearSumError = LinearSumError/3
print("average error for linear is %f" % LinearSumError)

# find the testing error for quadratic k-fold cross validation
QuadSumError = 0
for i in range(3) :
    # now we need to set up the weight vector. we will use sympy for this as
    # we want symbolics so we can do a gradient later on
    w0, w1, w2, w3, w4 = sympy.symbols('w0, w1, w2, w3, w4')
    w = sympy.Matrix([w0, w1, w2, w3, w4])

    # Define now the linear hypothesis
    htrain = sympy.Matrix.zeros(dataQuadX[i].shape[0], 1)
    for row in range(dataQuadX[i].shape[0]):
        htrain[row,:] = dataQuadX[i][row,:]*w[:,:]

    # now we define the error function
    jw = Sum((htrain - dataY[i]).applyfunc(square))

    grad0 = sympy.Derivative(jw, w0).doit()
    grad1 = sympy.Derivative(jw, w1).doit()
    grad2 = sympy.Derivative(jw, w2).doit()
    grad3 = sympy.Derivative(jw, w3).doit()
    grad4 = sympy.Derivative(jw, w4).doit()

    solution = sympy.solve([grad0, grad1, grad2, grad3, grad4], dict=True)
    w0 = solution[0][w0]
    w1 = solution[0][w1]
    w2 = solution[0][w2]
    w3 = solution[0][w3]
    w4 = solution[0][w4]
    wSolved = sympy.Matrix([w0, w1, w2, w3, w4])

    testlength = 97 - dataQuadX[i].shape[0]
    testDataQuadX = sympy.Matrix()
    testDataY = sympy.Matrix()
    if i != 0 :
        for j in range(dataQuadX[0].shape[0]) :
            testDataQuadX = testDataQuadX.row_insert(-1, dataQuadX[0].row(j))
            testDataY = testDataY.row_insert(-1, dataY[0].row(j))
    if i != 1 :
        for j in range(dataQuadX[1].shape[0]) :
            testDataQuadX = testDataQuadX.row_insert(-1, dataQuadX[1].row(j))
            testDataY = testDataY.row_insert(-1, dataY[1].row(j))
    if i != 2 :
        for j in range(dataQuadX[2].shape[0]) :
            testDataQuadX = testDataQuadX.row_insert(-1, dataQuadX[2].row(j))
            testDataY = testDataY.row_insert(-1, dataY[2].row(j))

    htest = sympy.Matrix.zeros(testlength, 1)
    for row in range(testlength):
        htest[row,:] = testDataQuadX[row,:]*wSolved[:,:]
    testingError = Sum((htest - testDataY).applyfunc(square))/testlength
    QuadSumError = QuadSumError + testingError
    print("testing quad error is: %f" % testingError)
QuadSumError = QuadSumError/3
print("average error for quad is %f" % QuadSumError)

# keep track of which degree had less error.
d = 0
if(LinearSumError > QuadSumError):
    d = 2
else:
    d = 1

print("Degree %i had a lower average error" % d)

# final fitting
if d == 1 : # linear fit
    print("Running 100 training sets linearly")
    iterations = 0
    generalizationErrors = np.empty(100)
    modelingErrors = np.empty(100)
    while iterations < 100 :
        # first we need to split the data
        trainingX = sympy.Matrix()
        trainingY = sympy.Matrix()
        testingX = sympy.Matrix()
        testingY = sympy.Matrix()
        for i in range(97) :
            if random.randint(0, 1) == 1 and trainingX.shape[0] < 49 :
                trainingX = trainingX.row_insert(-1, reducedX.row(i))
                trainingY = trainingY.row_insert(-1, y.row(i))
            elif testingX.shape[0] < 48 :
                testingX = testingX.row_insert(-1, reducedX.row(i))
                testingY = testingY.row_insert(-1, y.row(i))
            else :
                trainingX = trainingX.row_insert(-1, reducedX.row(i))
                trainingY = trainingY.row_insert(-1, y.row(i))

        # now we need to set up the weight vector. we will use sympy for this as
        # we want symbolics so we can do a gradient later on
        w0, w1, w2 = sympy.symbols('w0, w1, w2')
        w = sympy.Matrix([w0, w1, w2])


        # Define now the linear hypothesis
        hx = sympy.Matrix.zeros(m, 1)
        for row in range(m):
            hx[row,:] = trainingX[row,:]*w[:,:]

        # now we define the error function
        jw = Sum((hx - trainingY).applyfunc(square)) + babyShep*Sum(w.applyfunc(square))

        grad0 = sympy.Derivative(jw, w0).doit()
        grad1 = sympy.Derivative(jw, w1).doit()
        grad2 = sympy.Derivative(jw, w2).doit()

        solution = sympy.solve([grad0, grad1, grad2], dict=True)
        w0 = solution[0][w0]
        w1 = solution[0][w1]
        w2 = solution[0][w2]
        wSolved = sympy.Matrix([w0, w1, w2])

        htest = sympy.Matrix.zeros(48, 1)
        for row in range(48):
            htest[row,:] = testingX[row,:]*wSolved[:,:]
            hx[row,:] = trainingX[row,:]*wSolved[:,:]
        hx[48,:] = trainingX[48,:]*wSolved[:,:]

        generalizationError = Sum((htest - testingY).applyfunc(square))/testingY.shape[0]
        modelingError = Sum((hx - trainingY).applyfunc(square))/trainingY.shape[0]

        sys.stdout.write("\r%s%%" % str(iterations+1))
        sys.stdout.flush()

        generalizationErrors[iterations] = generalizationError
        modelingErrors[iterations] = modelingError
        iterations += 1
else : # quadratic fit
    print("Running 100 training sets quadratic")
    iterations = 0
    generalizationErrors = np.empty(100)
    modelingErrors = np.empty(100)
    while iterations < 100 :
        # first we need to split the data
        trainingX = sympy.Matrix()
        trainingY = sympy.Matrix()
        testingX = sympy.Matrix()
        testingY = sympy.Matrix()
        for i in range(97) :
            if random.randint(0, 1) == 1 and trainingX.shape[0] < 49 :
                trainingX = trainingX.row_insert(-1, reducedQuadX.row(i))
                trainingY = trainingY.row_insert(-1, y.row(i))
            elif testingX.shape[0] < 48 :
                testingX = testingX.row_insert(-1, reducedQuadX.row(i))
                testingY = testingY.row_insert(-1, y.row(i))
            else :
                trainingX = trainingX.row_insert(-1, reducedQuadX.row(i))
                trainingY = trainingY.row_insert(-1, y.row(i))

        # now we need to set up the weight vector. we will use sympy for this as
        # we want symbolics so we can do a gradient later on
        w0, w1, w2, w3, w4 = sympy.symbols('w0, w1, w2, w3, w4')
        w = sympy.Matrix([w0, w1, w2, w3, w4])


        # Define now the linear hypothesis
        hx = sympy.Matrix.zeros(m, 1)
        for row in range(m):
            hx[row,:] = trainingX[row,:]*w[:,:]

        # now we define the error function
        jw = Sum((hx - trainingY).applyfunc(square)) + babyShep*Sum(w.applyfunc(square))

        grad0 = sympy.Derivative(jw, w0).doit()
        grad1 = sympy.Derivative(jw, w1).doit()
        grad2 = sympy.Derivative(jw, w2).doit()
        grad3 = sympy.Derivative(jw, w3).doit()
        grad4 = sympy.Derivative(jw, w4).doit()

        solution = sympy.solve([grad0, grad1, grad2, grad3, grad4], dict=True)
        w0 = solution[0][w0]
        w1 = solution[0][w1]
        w2 = solution[0][w2]
        w3 = solution[0][w3]
        w4 = solution[0][w4]
        wSolved = sympy.Matrix([w0, w1, w2, w3, w4])

        htest = sympy.Matrix.zeros(48, 1)
        for row in range(48):
            htest[row,:] = testingX[row,:]*wSolved[:,:]
            hx[row,:] = trainingX[row,:]*wSolved[:,:]
        hx[48,:] = trainingX[48,:]*wSolved[:,:]

        generalizationError = Sum((htest - testingY).applyfunc(square))/testingY.shape[0]
        modelingError = Sum((hx - trainingY).applyfunc(square))/trainingY.shape[0]

        sys.stdout.write("\r%s%%" % str(iterations+1))
        sys.stdout.flush()

        generalizationErrors[iterations] = generalizationError
        modelingErrors[iterations] = modelingError
        iterations += 1

print("\nGeneralization:")
print("\tMin: %f\n\tMax: %f\n\tAvg: %f" % (np.amin(generalizationErrors), np.amax(generalizationErrors), np.average(generalizationErrors)))
print("Modeling")
print("\tMin: %f\n\tMax: %f\n\tAvg: %f" % (np.amin(modelingErrors), np.amax(modelingErrors), np.average(modelingErrors)))

# Plot the error
mpl.figure(1)
x_vals = np.linspace(0, 99, 100)

mpl.plot(x_vals, generalizationErrors, 'r-',label="Generalization Error")
mpl.plot(x_vals, modelingErrors, 'g-', label="Modeling Error")
mpl.legend(loc='best')
mpl.title("Generalization vs Modeling Error")
mpl.xlabel("Iteration")
mpl.ylabel("Error")


# Plot regression line
mpl.figure(2)
t = sympy.symbols('t')
if(d == 1):
    # Linear
    regression = w0 + w1*t + w2*t
else:
    # Quad
    regression = w0 + w1*t + w2*t + w3*t**2 + w4*t**2
lam_x = sympy.lambdify(t, regression, modules=['numpy'])

x_vals = linspace(float(min(x)), float(max(x)), 100)
y_vals = lam_x(x_vals);
# mpl.plot(x_vals, y_vals)
mpl.plot(x,y,'b.')


# Plot cost contours
# mpl.figure(3)


mpl.show()
