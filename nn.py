############################################
##### Neural Networks code
##### By Nikhil Tibrewal
############################################
##### A neural network with 1 hidden layer.
##### Adapted from my Octave implementation
##### of this neural network that I did for
##### Machine Learning course on coursera
############################################

import numpy as np
import scipy.io
from scipy.optimize import fmin_cg
from sklearn.metrics import accuracy_score
import sklearn.datasets as datasets
from sklearn import cross_validation

# Sigmoid function
def sigmoid(z):
	z = np.array(z, float)
	return 1/(1 + np.exp(-z))

# Sigmoid gradient function
def sigmoidGradient(z):
	s = sigmoid(z)
	return s * (1 - s)

# Function to randomly initialize Theta
# Each value is between -epsilon and epsilon
def randInitializeWeights(L_in, L_out, epsilon_init):
	return np.random.rand(L_out, L_in+1) * 2 * epsilon_init - epsilon_init

# Code to predict labels for given examples in X
def predict(Theta1, Theta2, X):
	m = X.shape[0]

	h1 = sigmoid(np.dot(np.hstack((np.ones((m,1), dtype=float), X)), np.transpose(Theta1)))
	h2 = sigmoid(np.dot(np.hstack((np.ones((m,1), dtype=float), h1)), np.transpose(Theta2)))
	return h2.argmax(axis=1) # for each row in h2, return index of max element (zero indexed)

# Compute the regularized cost by vectorized feed forward
def nnCostFunction(Theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaPar):
	m = X.shape[0] # m x 400

	Theta1 = np.reshape(Theta[0:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
	Theta2 = np.reshape(Theta[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

	# Variables to return
	J = 0;
	Theta1_grad = np.zeros(Theta1.shape, dtype=float); # 25 x 401
	Theta2_grad = np.zeros(Theta2.shape, dtype=float); # 10 x 26

	# Vectorized feed forward
	yi = np.eye(num_labels)[y].reshape(m, num_labels) # 5000 x 10

	a1 = np.hstack((np.ones((m, 1), dtype=float), X)) # 5000 x 401
	z2 = np.dot(a1, Theta1.T) # 5000 x 25
	a2 = np.hstack((np.ones((m, 1), dtype=float), sigmoid(z2))) # 5000 x 26
	z3 = np.dot(a2, Theta2.T) # 5000 x 10
	h_thetaX = sigmoid(z3) # 5000 x 10

	J = np.sum((-yi) * np.log(h_thetaX) - (1-yi) * np.log(1-h_thetaX))

	# Regularization value to add
	squared1sum = np.sum(np.power(Theta1, 2), axis=0) # sum along columns
	squared2sum = np.sum(np.power(Theta2, 2), axis=0)
	regVal = np.sum(squared1sum) - squared1sum[1] + np.sum(squared2sum) - squared2sum[1]

	return J/m + lambdaPar/(2.0*m) * regVal

# Compute regularized gradients
def nnGradFunction(Theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaPar):
	m = X.shape[0] # m x 400

	Theta1 = np.reshape(Theta[0:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
	Theta2 = np.reshape(Theta[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

	# Variables to return
	Theta1_grad = np.zeros(Theta1.shape, dtype=float); # 25 x 401
	Theta2_grad = np.zeros(Theta2.shape, dtype=float); # 10 x 26

	# Vectorized Feed forward
	yi = np.eye(num_labels)[y].reshape(m, num_labels) # 5000 x 10

	a1 = np.hstack((np.ones((m, 1), dtype=float), X)) # 5000 x 401
	z2 = np.dot(a1, Theta1.T) # 5000 x 25
	a2 = np.hstack((np.ones((m, 1), dtype=float), sigmoid(z2))) # 5000 x 26
	z3 = np.dot(a2, Theta2.T) # 5000 x 10
	h_thetaX = sigmoid(z3) # 5000 x 10

	# Back propagation
	for i in xrange(m):
		delta3 = h_thetaX[i].reshape(-1, 1) - yi[i].reshape(-1,1) # 10 x 1
		delta2 = np.dot(Theta2.T, delta3) # 26 x 1
		delta2 = delta2[1:] * sigmoidGradient(z2[i].reshape(-1,1))		
		
		Theta1_grad = Theta1_grad + np.dot(delta2, a1[i].reshape(1,-1))
		Theta2_grad = Theta2_grad + np.dot(delta3, a2[i].reshape(1,-1))

	# compute gradient with regularization
	ToAdd1 = lambdaPar/m * Theta1
	ToAdd1[:,1] = 0
	Theta1_grad = 1.0/m * Theta1_grad + ToAdd1
	ToAdd2 = lambdaPar/m * Theta2
	ToAdd2[:,1] = 0
	Theta2_grad = 1.0/m * Theta2_grad + ToAdd2

	return np.hstack((np.ravel(Theta1_grad), np.ravel(Theta2_grad)))

# Main code
# Load data into variables X and y

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)

input_layer_size  = X_train.shape[1]
hidden_layer_size = 25
num_labels = len(set(y))
lambdaPar = 1
epsilon_init = 0.12

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size, epsilon_init)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels, epsilon_init)
initial_Theta = np.hstack((np.ravel(initial_Theta1), np.ravel(initial_Theta2)))

options = {'maxiter': 500}
Theta = fmin_cg(f = lambda t: nnCostFunction(t, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambdaPar), 
				x0 = initial_Theta, 
				fprime = lambda t: nnGradFunction(t, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambdaPar), 
				maxiter = 500)


Theta1fitted = np.reshape(Theta[0:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
Theta2fitted = np.reshape(Theta[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

predictedTrain = predict(Theta1fitted, Theta2fitted, X_train) 
predictedTest = predict(Theta1fitted, Theta2fitted, X_test)

print "Train accuracy: ", accuracy_score(y_train, predictedTrain)
print "Test accuracy: ", accuracy_score(y_test, predictedTest)

