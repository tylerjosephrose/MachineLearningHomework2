import sys
try:
    import sympy
    from sympy import pprint, init_printing, Sum, lambdify
    from sympy.abc import k
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

# m is the number of training samples
m = x.shape[0]

# Start L2 Regularization here
# need to divide regularized data into training and test set

# extend the data set by the bias column: 
#       Each row receives a 1 in front since this is linear regression we want a 1 in front 
#       for the b part of mx + b
# we will call this ex for extended x
# here we should be using the normalized data not x
ex = sympy.Matrix(x)
cols = sympy.Matrix.ones(m, 1)
ex = ex.col_insert(0, cols)

# now we need to set up the weight vector. we will use sympy for this as
# we want symbolics so we can do a gradient later on
w0, w1, w2, w3 = sympy.symbols('w0, w1, w2, w3')
w = sympy.Matrix([w0, w1, w2, w3])


# Define now the linear hypothesis
hx = sympy.Matrix.zeros(m, 1)
for row in range(m):
    hx[row,:] = ex[row,:]*w[:,:]

# now we define the error function
square = lambda x: x*x
babyShep = 10
jw = Sum((hx - y).applyfunc(square)) + babyShep*Sum(w.applyfunc(square))

grad0 = sympy.Derivative(jw, w0).doit()
grad1 = sympy.Derivative(jw, w1).doit()
grad2 = sympy.Derivative(jw, w2).doit()
grad3 = sympy.Derivative(jw, w3).doit()

solution = sympy.solve([grad0, grad1, grad2, grad3], dict=True)
w0 = solution[0][w0]
w1 = solution[0][w1]
w2 = solution[0][w2]
w3 = solution[0][w3]

# print(sympy.N(hx))
print("w0: %f" % (w0))
print("w1: %f" % (w1))
print("w2: %f" % (w2))
print("w3: %f\n" % (w3))
minimum = min(abs(w0), abs(w1), abs(w2), abs(w3))
print("The min is %f" % minimum)
if minimum == abs(w0) :
    print("w0")
elif minimum == abs(w1) :
    print("w1")
elif minimum == abs(w2) :
    print("w2")
elif minimum == abs(w3) :
    print("w3")
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
