import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Object Detection Using Convolutional Neural Network
# according to VGG-16 Arhitecture
# every layers implemented from scratch using numpy

# convolutional layer
class ConvolutionalLayer():
	def __init__(self, input_array, kernal_size, number_of_kernal, stride=1, padding=1):
		self.input_array = input_array
		self.diamention = input_array.shape
		self.number_of_kernal = number_of_kernal
		self.kernal_depth = self.diamention[-1]
		self.kernal_size = kernal_size
		self.kernal = self._kernal_()
		self.stride = stride
		self.padding = padding

	def convolve(self):
		h_input_array, w_input_array = self.diamention[0], self.diamention[1]
		h_kernal, w_kernal = self.kernal_size, self.kernal_size
		if (h_input_array - h_kernal) % self.stride == 0:
			f = []
			for k in self.kernal:
				feature_map = []
				for i in range(0, (h_input_array - h_kernal), self.stride):
					feature_map_row = []
					for j in range(0, (w_input_array - w_kernal), self.stride):
						receptive_field = self.input_array[i:i+h_kernal, j:j+w_kernal]
						convoleved = receptive_field * k
						sum_concoleved = np.sum(np.sum(np.sum(convoleved)))
						feature_map_row.append(sum_concoleved)
					feature_map.append(feature_map_row)
				f.append(feature_map)
			return np.array(f)
		else:
			print("Given Stride is not satisfied")
			return None

	def _kernal_(self):
		kernal_bank = np.random.rand(self.number_of_kernal ,  self.kernal_size, self.kernal_size, self.kernal_depth)
		return kernal_bank

	def _padding_(self):
		pass

	

class RELULayer():
	def __init__(self, feature_map):
		self.feature_map = feature_map

	def relu(self):
		mask = self.feature_map < 0
		activated_matrix = np.where(mask, 0, self.feature_map)
		return np.array(activated_matrix)

	def sigmoid(self):
		pass

	def tanh(self):
		pass

class MaxPoolingLayer():
	def __init__(self, activated_matrix, window_size=2, stride=2):
		self.activated_matrix = activated_matrix
		self.window_size = window_size
		self.stride = stride

	def max_pooling(self):
		_, h, w = self.activated_matrix.shape
		max_pooled_matrix = []
		for i in range(0, (h - self.window_size), self.stride):
			max_pooled_row = []
			for j in range(0, (w - self.window_size), self.stride):
				receptive_field = self.activated_matrix[i:i+self.window_size, j:j+self.window_size]
				max_value = np.max(receptive_field)
				max_pooled_row.append(max_value)
			max_pooled_matrix.append(max_pooled_row)
		return np.array(max_pooled_matrix)

class FullyConnectedLayer():
	pass

class DropOutLayer():
	pass

class SoftMaxLayer():
	pass

class FeedForward():
	pass

class BackproperGate():
	pass









#################################################

image = Image.open('b.jpg')
image = image.resize((100,100), Image.LANCZOS)
image = np.array(image)



conv_layer = ConvolutionalLayer(image, 3,10)
feature_map = conv_layer.convolve()
# print(feature_map[0])
# plt.imshow(feature_map[7])

relu_layer = RELULayer(feature_map)
activated_matrix = relu_layer.relu()
# print(activated_matrix.shape)
# plt.imshow(activated_matrix)

max_pooling_layer = MaxPoolingLayer(activated_matrix)
# max_pooled_matrix = max_pooling_layer.max_pooling()
# plt.imshow(max_pooled_matrix)





plt.show()

		
