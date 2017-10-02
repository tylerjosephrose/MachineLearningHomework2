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
    sum = 0;
    for row in range(length):
        sum += (column[row] - mean)**(2)
    return ((1/length)*sum)**(1/2)

# imported dataset
lines = [line.rstrip('\n') for line in open("hm2Data.csv")]
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
babyShep = 10
count = 0
while(count < 10):

    # now we need to set up the weight vector. we will use sympy for this as
    # we want symbolics so we can do a gradient later on
    w0, w1, w2, w3 = sympy.symbols('w0, w1, w2, w3')
    w = sympy.Matrix([w0, w1, w2, w3])


    # Define now the linear hypothesis
    hx = sympy.Matrix.zeros(m, 1)
    for row in range(m):
        hx[row,:] = trainingX[row,:]*w[:,:]

    # now we define the error function
    square = lambda x: x*x
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
    testingError = Sum((htest - testingY).applyfunc(square))/48
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

# Cross Validation
redX1 = sympy.Matrix()
redY1 = sympy.Matrix()
redX2 = sympy.Matrix()
redY2 = sympy.Matrix()
redX3 = sympy.Matrix()
redY3 = sympy.Matrix()
for i in range(97) :
    rand = random.randint(0,2)
    if rand == 0 and redX1.shape[0] < 33 :
        redX1 = redX1.row_insert(-1, reducedX.row(i))
        redY1 = redY1.row_insert(-1, y.row(i))
    elif rand == 1 and redX2.shape[0] < 32 or rand == 0 and redX1.shape[0] == 33 and redX2.shape[0] < 33:
        redX2 = redX2.row_insert(-1, reducedX.row(i))
        redY2 = redY2.row_insert(-1, y.row(i))
    elif redX3.shape[0] < 32 :
        redX3 = redX3.row_insert(-1, reducedX.row(i))
        redY3 = redY3.row_insert(-1, y.row(i))
    elif redX1.shape[0] < 33 :
        redX1 = redX1.row_insert(-1, reducedX.row(i))
        redY1 = redY1.row_insert(-1, y.row(i))
    else :
        redX2 = redX2.row_insert(-1, reducedX.row(i))
        redY2 = redY2.row_insert(-1, y.row(i))
dataX = [redX1, redX2, redX3]
dataY = [redY1, redY2, redY3]
SumError = 0;
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
    square = lambda x: x*x
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
    SumError = SumError + testingError
    print("testing error is: %f" % testingError)
SumError = SumError/3
print("average error for linear is %f" % SumError)

quit()
# Part 2: Plot
t = sympy.symbols('t')
linReg = w0 + w1*t
lam_x = sympy.lambdify(t, linReg, modules=['numpy'])
x_vals = linspace(float(min(x)), float(max(x)), 100)
y_vals = lam_x(x_vals);

'''mpl.title("Linear Regression from Symbolic Variable Solving")
mpl.xlabel("City Population (in 10,000s)")
mpl.ylabel("Estimated Company Profit (in 10,000s)")
mpl.plot(x_vals, y_vals)
mpl.plot(x, y, 'go')'''

# we will make some predictions as well
predict1 = 1*w0 + 3.5*w1
print("For the population = 35,000, we predict a profit of $%.2f" % (predict1*10000))
predict2 = 1*w0 + 7*w1
print("For the population = 70,000, we predict a profit of $%.2f\n" % (predict2*10000))

'''# contour plots
J_vals = J_vals.T
fig = mpl.figure()
ax = fig.add_subplot(211, projection='3d')
w0_vals, w1_vals = np.meshgrid(w0_vals, w1_vals)
ax.plot_surface(w0_vals, w1_vals, J_vals)
ax.set_xlabel('w_0')
ax.set_ylabel('w_1')
ax.set_title('Surface')

ax2 = fig.add_subplot(212, projection='3d')
ax2.contour(w0_vals, w1_vals, J_vals, np.logspace(-2, 3, 20))
ax2.set_title('Contour')
ax2.set_xlabel('w_0')
ax2.set_ylabel('w_1')
numpyW = np.matrix(w).astype(np.float64)
ax2.plot(numpyW[0], numpyW[1], 'rx', markersize=10, linewidth=2)'''
mpl.show()
