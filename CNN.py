import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ConvolutionalLayer():
	def __init__(self, input_array, kernal, stride=1, padding=1):
		self.input_array = input_array
		self.kernal = kernal
		self.stride = stride
		self.padding = padding

	def convolve(self, stride=1):
		h_input_array, w_input_array, _ = self.input_array.shape
		h_kernal, w_kernal, _ = self.kernal.shape
		if (h_input_array - h_kernal) % stride == 0:
			feature_map = []
			for i in range(0, (h_input_array - h_kernal), stride):
				feature_map_row = []
				for j in range(0, (w_input_array - w_kernal), stride):
					receptive_field = self.input_array[i:i+h_kernal, j:j+w_kernal]
					convoleved = receptive_field * self.kernal
					sum_concoleved = np.sum(np.sum(np.sum(convoleved)))
					feature_map_row.append(sum_concoleved)
				feature_map.append(feature_map_row)
			return np.array(feature_map)
		else:
			print("Given Stride is not satisfied")
			return None

	def convolve1(self, stride=1):
		h_input_array, w_input_array = self.input_array.shape
		h_kernal, w_kernal = self.kernal.shape
		if (h_input_array - h_kernal) % stride == 0:
			feature_map = []
			for i in range(0, (h_input_array - h_kernal), stride):
				feature_map_row = []
				for j in range(0, (w_input_array - w_kernal), stride):
					receptive_field = self.input_array[i:i+h_kernal, j:j+w_kernal]
					convoleved = receptive_field * self.kernal
					sum_concoleved = np.sum(np.sum(convoleved))
					feature_map_row.append(sum_concoleved)
				feature_map.append(feature_map_row)
			return np.array(feature_map)
		else:
			print("Given Stride is not satisfied")
			return None

class RELULayer():
	def __init__(self, feature_map):
		self.feature_map = feature_map

	def relu(self):
		mask = self.feature_map < 0
		activated_matrix = np.where(mask, 0, self.feature_map)
		return np.array(activated_matrix)

class MaxPoolingLayer():
	def __init__(self, activated_matrix, window_size=2, stride=1):
		self.activated_matrix = activated_matrix
		self.window_size = window_size
		self.stride = stride

	def max_pooling(self):
		h, w = self.activated_matrix.shape
		max_pooled_matrix = []
		for i in range(0, (h - self.window_size), self.stride):
			max_pooled_row = []
			for j in range(0, (w - self.window_size), self.stride):
				receptive_field = self.activated_matrix[i:i+self.window_size, j:j+self.window_size]
				max_value = np.max(receptive_field)
				max_pooled_row.append(max_value)
			max_pooled_matrix.append(max_pooled_row)
		return np.array(max_pooled_matrix)

image = Image.open('b.jpg')
image = image.resize((100,100), Image.LANCZOS)
image = np.array(image)
kernal = np.array([
	[[1,0,0],[0,0,0],[-1,0,0]],
	[[1,0,0],[0,0,0],[-1,0,0]],
	[[1,0,0],[0,0,0],[-1,0,0]]
	]) 
kernal1 = np.array([
	[1,0,-1],
	[1,0,-1],
	[1,0,-1]
	]) 

conv_layer = ConvolutionalLayer(image, kernal)
feature_map = conv_layer.convolve()
# plt.imshow(feature_map)

relu_layer = RELULayer(feature_map)
activated_matrix = relu_layer.relu()
# plt.imshow(activated_matrix)

max_pooling_layer = MaxPoolingLayer(activated_matrix)
max_pooled_matrix = max_pooling_layer.max_pooling()
# plt.imshow(max_pooled_matrix)



plt.show()

		
