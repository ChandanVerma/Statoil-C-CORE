import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from dnn_app_utils_v2 import *
##### Initializing a 2 layer NN

# Exercise: Create and initialize the parameters of the 2-layer neural network.

# Instructions:

# The model's structure is: LINEAR -> RELU -> LINEAR -> SIGMOID.
# Use random initialization for the weight matrices. Use np.random.randn(shape)*0.01 with the correct shape.
# Use zero initialization for the biases. Use np.zeros(shape).


def initialize_parameters(n_x, n_h, n_y):

	#n_x : size of input layer
	#n_h : size of hidden layer
	#n_y : size of output layer

	np.random.seed(1)

	W1 = np.random.randn(n_h , n_x) * 0.01
	b1 = np.zeros(shape = (n_h , 1))
	W2 = np.random.randn(n_y , n_h) * 0.01
	b2 = np.zeros(shape = (n_y , 1))

	assert(W1.shape == (n_h, n_x))
	assert(b1.shape == (n_h, 1))
	assert(W2.shape == (n_y, n_h))
	assert(b2.shape == (n_y, 1))

	parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

	return parameters



# parameters = initialize_parameters(3,2,1)
# print("W1 = " , parameters["W1"])
# print("W2 = " , parameters["W2"])
# print("b1 = " , parameters["b1"])
# print("b2 = " , parameters["b2"])



# Exercise: Implement initialization for an L-layer Neural Network.

# Instructions:

# The model's structure is [LINEAR -> RELU]  ××  (L-1) -> LINEAR -> SIGMOID. I.e., it has  L−1L−1  layers using a ReLU activation function followed by an output layer with a sigmoid activation function.
# Use random initialization for the weight matrices. Use np.random.rand(shape) * 0.01.
# Use zeros initialization for the biases. Use np.zeros(shape).
# We will store  n[l]n[l] , the number of units in different layers, in a variable layer_dims. For example, the layer_dims for the "Planar Data classification model" from last week would have been [2,4,1]: There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit. Thus means W1's shape was (4,2), b1 was (4,1), W2 was (1,4) and b2 was (1,1). Now you will generalize this to  LL  layers!
# Here is the implementation for  L=1L=1  (one layer neural network). It should inspire you to implement the general case (L-layer neural network).
#   if L == 1:
#       parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
#       parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))

def initialize_parameters_deep(layer_dims):


	np.random.seed(3)
	parameters = {}
	L = len(layer_dims)

	for l in range(1,L):
		parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
		parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

		assert(parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
		assert(parameters["b" + str(l)].shape == (layer_dims[l], 1))

	return parameters


# parameters = initialize_parameters_deep([75,4,4,4,4,4,3,1])
# print("W1 = " , parameters["W1"])
# print("W2 = " , parameters["W2"])
# print("W3 = " , parameters["W3"])
# print("W4 = " , parameters["W4"])
# print("W5 = " , parameters["W5"])
# print("W6 = " , parameters["W6"])
# print("W7 = " , parameters["W7"])
# print("b1 = " , parameters["b1"])
# print("b2 = " , parameters["b2"])
# print("b3 = " , parameters["b3"])
# print("b4 = " , parameters["b4"])
# print("b5 = " , parameters["b5"])
# print("b6 = " , parameters["b6"])
# print("b7 = " , parameters["b7"])


# Exercise: Build the linear part of forward propagation.

# Reminder: The mathematical representation of this unit is  Z[l]=W[l]A[l−1]+b[l] . You may also find np.dot() useful. If your dimensions don't match, printing W.shape may help.

def linear_forward(A, W, b):

	Z = np.dot(W, A) + b
	assert(Z.shape == (W.shape[0], A.shape[1]) )

	cache = (A, W, b)

	return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

	if activation == 'sigmoid':
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)
	elif activation == 'relu':
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)

	assert(A.shape ==(W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)

	return A, cache


# A_prev, W, b = linear_activation_forward_test_case()

# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
# print("With sigmoid: A = " + str(A))

# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))

# Exercise: Implement the forward propagation of the above model.

# Instruction: In the code below, the variable AL will denote  A[L]=σ(Z[L])=σ(W[L]A[L−1]+b[L])A[L]=σ(Z[L])=σ(W[L]A[L−1]+b[L]) . (This is sometimes also called Yhat, i.e., this is  Ŷ Y^ .)

# Tips:

# Use the functions you had previously written
# Use a for loop to replicate [LINEAR->RELU] (L-1) times
# Don't forget to keep track of the caches in the "caches" list. To add a new value c to a list, you can use list.append(c).


def L_model_forward(X, parameter):

	caches = []
	A = X
	L = len(parameter) // 2 ## Number of layers in the neural network

	for l in range(1,L):
		A_prev = A

		A, cache = linear_activation_forward(A_prev, parameter["W" + str(l)], parameter["b" + str(l)], activation = 'relu')

		caches.append(cache)

	AL, cache = linear_activation_forward(A, parameter["W" + str(L)], parameter["b" + str(L)], activation = 'sigmoid')

	caches.append(cache)

	assert(AL.shape == (1, X.shape[1]))

	return AL, caches


# X, parameters = L_model_forward_test_case_2hidden()
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))


# Exercise: Compute the cross-entropy cost  JJ , using the following formula: −1m∑i=1m(y(i)log(a[L](i))+(1−y(i))log(1−a[L](i)))

def compute_cost(AL, Y):

	m = Y.shape[1]

	cost = - ( 1 / m ) * ( np.sum(   np.multiply(       Y,   np.log(     AL ) )
                                   + np.multiply( ( 1 - Y ), np.log( 1 - AL ) ),
                                   axis = 1
                                   )
                           )

	cost = np.squeeze(cost)
	#assert(cost.shape == ())
	return cost



# Y, AL = compute_cost_test_case()

# print("cost = " + str(compute_cost(AL, Y)))


# The three outputs (dW[l],db[l],dA[l])(dW[l],db[l],dA[l]) are computed using the input dZ[l]dZ[l].Here are the formulas you need:
# dW[l]=∂∂W[l]=1mdZ[l]A[l−1]T
# dW[l]=∂L∂W[l]=1mdZ[l]A[l−1]T
# db[l]=∂∂b[l]=1m∑i=1mdZ[l](i)
# db[l]=∂L∂b[l]=1m∑i=1mdZ[l](i)
# dA[l−1]=∂∂A[l−1]=W[l]TdZ[l]
# dA[l−1]=∂L∂A[l−1]=W[l]TdZ[l]
# Exercise: Use the 3 formulas above to implement linear_backward().

def linear_backward(dZ, cache):


	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = np.dot(dZ, cache[0].T)/m
	db = np.squeeze(np.sum(dZ,axis = 1, keepdims = True))
	dA_prev = np.dot(cache[1].T, dZ)
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	#assert (isinstance(db, float))
	return dA_prev, dW, db    
    

# dZ, linear_cache = linear_backward_test_case()

# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))


def linear_activation_backward(dA, cache, activation):

	linear_cache, activation_cache = cache

	if activation == 'relu':
		dZ = relu_backward(dA, activation_cache)
	elif activation == 'sigmoid':
		dZ = sigmoid_backward(dA, activation_cache)

	dA_prev, dW, db = linear_backward(dZ, linear_cache)

	return dA_prev, dW, db 


# AL, linear_activation_cache = linear_activation_backward_test_case()

# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")

# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    ### END CODE HERE ###
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                  current_cache,
                                                                                                  "sigmoid")
    ### END CODE HERE ###
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA{}".format(l + 2)],
                                                                    current_cache,
                                                                    "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads


# AL, Y_assess, caches = L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dA1 = "+ str(grads["dA1"]))


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W{}".format(l + 1)] - learning_rate * grads["dW{}".format(l + 1)]
        parameters["b" + str(l+1)] = parameters["b{}".format(l + 1)] - learning_rate * grads["db{}".format(l + 1)]
    ### END CODE HERE ###
        
    return parameters


# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)

# print ("W1 = "+ str(parameters["W1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["W2"]))
# print ("b2 = "+ str(parameters["b2"]))