#We use all the methods _from numpy to do matrix operation effiently
import numpy as np
from layer import Layer
from zope.interface import implementer
import matplotlib.pyplot as plt

# Object Detection Using Convolutional Neural Network
# every layers implemented from scratch using numpy

########### In Convolutional Neural Network, We have two blocks ####################
			# 1. feature learning block
			# 2. predecting block

########### 1st We will impelemnt feature learning block ###########################
########## There are 3 Layers is fearture learning block ###########################
			# 1. Convolution Layer
			# 2. Activation Layer
			# 3. pooling Layer

# Convolutional Layer
class ConvolutionalLayer():
	"this class is to do convolution to given input array with randomly selected filter bank"

	#__init__ method to store all intial data
	def __init__(self):

		self.input_array = None
		self.number_of_kernal = None
		self.diamention = None
		self.kernal_depth = None
		self.kernal_size = None
		self.kernal = None
		self.stride = None 
		self.padding = None
		self.output = None

	#this method is do convolution between input array and filter(kernal)
	def feed_forward(self, input_array, kernal_size=3, number_of_kernal=5, stride=1, padding=1,image=False):
		#storing the input array and number of filters in the filter bank.
		self.input_array = input_array
		self.number_of_kernal = number_of_kernal

		#check whether the given input array is original image or not.
		#In original 3D images, the diamentions are differnt from numpy 3D arrays.
		#Thus in 3D image array the 3rd axis is last element and In nupmy 3D arrays, the 3rd axis in first
		if not image:

			#if the input is not a image, then diamentions are straight forward
			self.diamention = self.input_array.shape
			self.input_array = self.input_array.reshape(self.input_array.shape[::-1])
		else:

			#if the input array is a image array then we have to reverse it.
			self.diamention = self.input_array.shape[::-1]

		# We have convert the image array into normal numpy array. So first element is depth of the array
		self.kernal_depth = self.diamention[0]
		self.kernal_size = kernal_size

		#we need to create the kernal according to given mesurments
 		#this class it self has a method that could return the above comment
		self.kernal = self._kernal_()

		#stride is how much blocks we need to shift the filter over the given array
		self.stride = stride

		#padding adding 0 layers on top of input array
		#padding = 1 ==> adding 1 layer 
		self.padding = padding


		# store height and width of the input array
		h_input_array, w_input_array = self.diamention[1], self.diamention[2]

		# store height and width of the kernal size which given as the input earlier
		h_kernal, w_kernal = self.kernal_size, self.kernal_size

		#check whether given parameters can do the convolution operation correctly
		if (h_input_array - h_kernal) % self.stride == 0:

			#This is 3D array
			#create the final output array, size of 
			#(input_height-filter_height)x(input_width-filter_width)x(number of filetrs in the filetr bank)
			feature_map_bank = np.zeros([self.number_of_kernal,(h_input_array - h_kernal)/self.stride+1, (w_input_array - w_kernal)/self.stride+1], dtype=int)

			# we have to do convolution between input array and every filter in the bank
			# example: if the filter bank has 10 filters then we have to do convolution with every filer and 
			#finally we have to stack to gather all out 2D array
			for k, _filter in enumerate(self.kernal):

				#This is 2D array
				#here we store output of convolution between input array and a single filter
				feature_map = np.zeros([(h_input_array - h_kernal)/self.stride+1, (w_input_array - w_kernal)/self.stride+1], dtype=int)

				#This is to get iteration through column and move the filter downward
				for i in range(0, (h_input_array - h_kernal)/self.stride+1, self.stride):

					#this array is 1D
					#store the convoled result after a row compeletd
					feature_map_row = np.zeros([(w_input_array - w_kernal)/self.stride+1], dtype=int)

					#This is to get iteration through row and move the filter right side
					for j in range(0, (w_input_array - w_kernal)/self.stride+1, self.stride):

						#find the receptive field
						# receptive field => same size of filter that choose from input array to do convolutions
						receptive_field = self.input_array[i:i+h_kernal, j:j+w_kernal]

						#convolution happen here,
						#element wise mulltiplication between
						#receptive filed of input array and filter
						convoleved = receptive_field * _filter

						#sum up the result of element wise multiplication
						sum_concoleved = np.sum(np.sum(np.sum(convoleved)))

				#mapping results 
						feature_map_row[j] = (sum_concoleved)
					feature_map[i] = (feature_map_row)
				feature_map_bank[k] = (feature_map)

			return np.array(feature_map_bank)

		else:
			#if the stride is wrong then the method return None
			print("Given Stride is not satisfied")
			return None

	#this method is to craete random filter bank for given measurments
	def _kernal_(self):

		#creating and return a random filter bank using numpy built in method
		kernal_bank = np.random.randint(low=-1, high=2, size=[self.number_of_kernal, self.kernal_size, self.kernal_size, self.kernal_depth])
		return kernal_bank

	#this method is to add given amount padding for given input array
	def _padding_(self):

		#we have to do padding in both left and right, and top and bottom.
		#so the padding amount is multiple by 2
		padding = self.padding * 2

		#create zero array which size is equal to the output array
		padded_array = np.zeros([(x+padding for x in self.input_array.shape)])

		#then place the input array inside the output array, inorder to get the padded array
		padded_array[padding/2 : -padding/2, padding/2 : -padding/2, padding/2 : -padding/2]
		return padded_array

	# this method is to do backpropagation, thus update weights for convolution layer
	#take learning rate and previous derivaties as inputs
	def backpropagation(self, upstream_derivatives, learning_rate=0.0000000007):
		# dw stand for how the weight matrix need to change
		upstream_derivatives = np.array(upstream_derivatives)
		dkernal = np.zeros(self.kernal.shape)
		dx = np.zeros(self.input_array.shape)
		h, w = self.input_array.shape[0], self.input_array.shape[1]
		dh, dw = upstream_derivatives.shape[1], upstream_derivatives.shape[2]
		for k, d in enumerate(upstream_derivatives):
			for j in range(0, (h-dh)/self.stride+1, self.stride):
				for i in range(0, (w-dw)/self.stride+1, self.stride):
					receptive_field = self.input_array[j:j+dh, i:i+dw]
					mult = receptive_field.T * d
					result = np.sum(np.sum(mult, axis=1), axis=1)
					dkernal[k, j, i] = result
					dx[k, j:j+dh, i:i+dw] += np.sum(np.sum(np.sum(self.kernal[k] * upstream_derivatives[k,j,i])))
		
		# dx stand for how the input matrix need to change
		# and this is a upstream or previous derivaties for above layer
		
		# dx = self.kernal.T.dot(upstream_derivatives)

		#update the weights or filter or kernal matrix, 
		# it also depend on learning rate and other some hyper parameters.
		# the hyper parameters came from deep reaserchers
		self.kernal = self.kernal - learning_rate*dkernal
		return (dx.T, self.kernal)
	
#Activation Layer
class ActivationLayer():
	'this class is contains activation methods, thus get an nD array and retun the same diamention of the given array after do activation'

	def __init__(self):
		self.feature_map = None

	#this is do relu operation
	#thus, set 0 to all element which are negative
	def relu(self, feature_map):

		self.feature_map = feature_map

		#create a mask that return a boolean array
		#it contains True at negative element index,
		#False at remaining
		mask = self.feature_map < 0

		#do relu operation, thus change negative to 0
		activated_matrix = np.where(mask, 0, self.feature_map)
		return np.array(activated_matrix)

	#this is sigmoid function
	def sigmoid(self):
		pass

	#this tanh operator
	def tanh(self):
		pass

# Pooling Layer : To do max pooling or average pooling to working with high score features
class MaxPoolingLayer():
	'this class is to do pooling operations, thus down sampaling the features which are detected by above layers'

	def __init__(self):
		self.activated_matrix = None
		self.diamentions = None
		self.window_size = None
		self.stride = None

	#this method is max pooling, thus getting max values inside the window
	def feed_forward(self, activated_matrix, window_size=2, stride=2):
		#store all initial values
		self.activated_matrix = activated_matrix
		self.diamentions = self.activated_matrix.shape
		#window size is amount of sampalizing
		#we get the max or average value (depend on type of pooling) in the window 
		self.window_size = window_size
		self.stride = stride

		#store depth, height, width
		d, h, w = self.diamentions
		#3D array
		#this is the final output array
		max_pooled_matrix_3D = np.zeros([d, (h-self.window_size)/self.stride+1, (w-self.window_size)/self.stride+1])

		#iterrate through depth
		for k in range(d):

			#2D array
			#result, after iterate through every row
			max_pooled_matrix_2D = np.zeros([(h-self.window_size)/self.stride+1, (w-self.window_size)/self.stride+1])

			#iterate through coloumn
			for i in range(0, (h - self.window_size)/self.stride+1, self.stride):

				#1D array
				#result, after iterate through every column in a single row
				max_pooled_row = np.zeros([(w-self.window_size)/self.stride+1])
				#iterate through a single row
				for j in range(0, (w - self.window_size)/self.stride+1, self.stride):

					#receptive filed is same size of the window that match with input array
					receptive_field = self.activated_matrix[k, i:i+self.window_size, j:j+self.window_size]
					#do max pool operation
					max_value = np.max(receptive_field)
					max_pooled_row[j] = (max_value)
				max_pooled_matrix_2D[i] = (max_pooled_row)
			max_pooled_matrix_3D[k] = max_pooled_matrix_2D
		#return final output
		return np.array(max_pooled_matrix_3D)

	def backpropagation(self, upstream_derivatives):
		output_matrix = self.activated_matrix
		d, h, w = self.diamentions
		for k in range(d):
			for i in range(0, (h - self.window_size)/self.stride+1, self.stride):
				for j in range(0, (w - self.window_size)/self.stride+1, self.stride):
					receptive_field = output_matrix[k, i:i+self.window_size, j:j+self.window_size]
					# print(receptive_field)
					max_index = np.where(np.max(receptive_field) == receptive_field)
					a = max_index[0]
					b = max_index[1]
					# print(k)
					output_matrix[k, i:i+self.window_size+a[0], j:j+self.window_size+b[0]] = upstream_derivatives[k, i, j]
		return output_matrix

########### Now We will impelemnt predecting block ############################
########## There are 5 Layers is fearture predecting block ###########################
			# 1. Dropout Layer
			# 2. Flatten Layer
			# 3. Dense Layer
			# 4. Softmax Layer
			# 5. Loss Function Layer

#Drop out Layer : To reduce more computation and memory usage, also for overfitting
class DropOutLayer():
	'This is a class for do drop out operation to reduce overfitting in training'
	def __init__(self):
		self.matrix = None
		self.dropout_value = None
		self.diamention = None

	def dropout(self, matrix, dropout_value=0.6):
		#define the input matix and the drop out value
		#drop out value is tells about, percentage of deactivating the neurons
		self.matrix = matrix
		self.dropout_value = dropout_value

		# get the input matrix diamentions
		self.diamention = self.matrix.shape

		#create a mask using drop out value
		mask = (np.random.random(self.diamention) < self.dropout_value)

		#then switched off random neurons using the mask by multiply with input matrix
		output_matrix = self.matrix * mask
		return output_matrix	

# Flatten Layer : flat a 3D array into 1D array
class FlattenLayer():
	'this class is to flat the result of feature learning block from 3D to 1D'

	def __init__(self):
		self.matrix_3D = None 

	#this method is to flat the 3D array into 1D array
	def feed_forward(self, matrix_3D):
		#store the given 3D array
		self.matrix_3D = matrix_3D

		#output is 1D array
		matrix_1D = self.matrix_3D.ravel()

		# iterate the 3D array and append all value into the 1D array
		
		return matrix_1D

	# this is back propagation method of Flatten Layer
	# thus reverse of the flatten layer
	# change 1D matrix to 3D matrix
	def backpropagation(self, upstream_derivatives):
		# previuos derivaties is 1D matrix. So we need to
		# change previous derivaties into 3D matrix
		matrix_1D = upstream_derivatives

		# this process is done by reshape method
		output_matrix_3D = matrix_1D.reshape(self.matrix_3D.shape)
		return output_matrix_3D

# Dense Layer is a fully connected layer
class DenseLayer():
	'this is a fully connected layer'
	def __init__(self):
		self.flatted_matrix = None
		self.diamentions = None
		self.number_of_neuron = None
		self.weight = None
		self.dense_output = None
		
	def feed_forward(self, flatted_matrix, number_of_neuron=0):
		# this method get an input flatted matrix
		self.flatted_matrix = flatted_matrix
		self.diamentions = self.flatted_matrix.shape
		# the number of neurons means, what is the output matrix size
		self.number_of_neuron = number_of_neuron
		# create a weight using _weight metod
		self.weight = self._weight()

		# output flatted matrix
		# size is equal to number of neurons
		dense_output = np.zeros(self.number_of_neuron)

		# do element wise multiplication and sum up
		for i in range(self.number_of_neuron):
			dense_output[i] = np.sum(self.flatted_matrix * self.weight[i].T)
		self.dense_output = dense_output
		return dense_output

	# this method define the process of backpropagation for dense layer
	def backpropagation(self, upstream_derivatives, learning_rate=0.0000000007):
		# we calculate dw and dx using previous derivaties, weight and input matrix
		self.flatted_matrix = self.flatted_matrix.reshape(1, self.flatted_matrix.shape[0])
		self.weight = self.weight.reshape(self.weight.shape)
		upstream_derivatives = np.array(upstream_derivatives)
		# print(upstream_derivatives.shape)
		if len(upstream_derivatives.shape) == 1:
			upstream_derivatives = upstream_derivatives.reshape(1, upstream_derivatives.shape[0])
		
		dw = upstream_derivatives.T.dot(self.flatted_matrix)
		
		# dx will be the previous derivaties for above layers
		dx = self.weight.T.dot(upstream_derivatives.T)
		# print(dx.shape)
		# weight updated after backpropagation
		self.weight = self.weight - learning_rate * dw
		return (dx.T, self.weight)
		
	# this method is to create weight matrix bank
	def _weight(self):
		# weight matrix size is same diamention as input matrix 
		size = list(self.diamentions)

		# add number of neuron into the size array to create a weight bank 
		# number of neuron define that how many weight would be in the weight bank
		size.insert(0, self.number_of_neuron)
		
		# weights created randomly
		# after it may change by back propagation
		weight = np.random.random(size)
		return weight


# SoftMax Layer : to calculate probabilty of each class
class SoftMaxLayer():
	'this class is to do softmax operation'
	def __init__(self):
		self.input_array = None

	def feed_forward(self, input_array):
		# ths input size is equal to size of classes which we have to predict
		# this done by setting the previos dense layer number of neurons to number of classes
		self.input_array = np.array(input_array, dtype=int)/1000000000
		# calculating probability for every element in the input matrix
		dinominator = np.sum([(np.exp(n)) for n in self.input_array])
		# out put size is same size of input matrix
		# and sum of output matriz equal one
		result = [(np.exp(n)/dinominator) for n in self.input_array]
		return result

	# this method define backpropagation for softmax layer
	def backpropagation(self, upstream_derivatives):
		# first find the local gradient by differenciating the softmax function
		# and find gradient for every input value 
		# print(upstream_derivatives)
		dinominator = np.sum([ np.exp(x) for x in self.input_array])
		local_gradient = np.array([((- np.exp(2*y) + dinominator * np.exp(y))/(dinominator**2)) for y in self.input_array])

		# finally find the derivaties w.r.t final error
		dx = local_gradient*upstream_derivatives
		return dx

#Loss Function Layer is to find the final loss
class LossFunction():
	'this layer is to find the final loss'
	def __init__(self):
		self.prediction = None
		self.label = None

	def loss(self, prediction, label):
		# this is output of the softmax layer
		# called by predicted values for classes
		self.prediction = prediction
		# print(self.prediction)
		# label is expected values for classes
		self.label = label

		# calculation the final loss or error
		# this is the previuos derivaties for softmax layer
		l = 0
		for i in range(self.label.size):
			l += - self.label[i]*np.log(self.prediction[i]) 
		# loss = np.sum([max(0, self.prediction[i] - self.prediction[np.argmax(self.label)] + 1) for i in range(self.label.size)])
		return l

	def backpropagation(self, upstream_derivatives):
		# l = []
		# for i in range(len(self.prediction)):
		# 	l.append(self.prediction[i]/self.label[i])
		return np.array(self.label)/self.prediction * upstream_derivatives

# class CostFunction():
# 	def __init__(self):
# 		self.label = None
# 		self.prediction = None
# 	def feed_forward(self, prediction, label):
# 		self.label = label
# 		self.prediction = prediction
# 		cost = self.label * np.log(self.prediction) + (1-self.label) * np.log(1-self.prediction)
# 		return cost

# 	def backpropagation(self, upstream_derivatives):
# 		d = self.label/self.prediction + (self.label-1)/(1-self.prediction)
# 		return d * upstream_derivatives




########################### END ######################################