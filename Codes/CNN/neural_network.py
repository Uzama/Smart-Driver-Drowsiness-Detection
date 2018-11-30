import pickle
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cnn import ( ConvolutionalLayer, 
	ActivationLayer, 
	MaxPoolingLayer
	)
from cnn import (DropOutLayer, 
	FlattenLayer, 
	DenseLayer, 
	SoftMaxLayer, 
	LossFunction
	)

from load_image import load_image

class Neuron:
	def __init__(self, function, pre_neuron, first=False, *args, **kwargs):
		self.pre_neuron = pre_neuron
		self.first = first
		self.function = function
		self.kwargs = kwargs
		self.input_matrix = self.dendrid()
		self.output_matrix = self.activate()
		if type(self.output_matrix)==tuple and len(self.output_matrix) != 1:
			self.output_matrix = self.output_matrix[0]
		

	def dendrid(self):
		if not self.first:
			return self.pre_neuron.axon()
		else:
			return self.pre_neuron

	def activate(self):
		return self.function(self.input_matrix, **self.kwargs)
		

	def axon(self):
		return self.output_matrix

class NeuralNetwork:
	def __init__(self):
		self.weight = {}
		self.convolution1 = ConvolutionalLayer()
		self.convolution2 = ConvolutionalLayer()
		self.convolution3 = ConvolutionalLayer()
		self.relu = ActivationLayer()
		self.pooling1 = MaxPoolingLayer()
		self.pooling2 = MaxPoolingLayer()
		self.drop_out = DropOutLayer()
		self.flatten = FlattenLayer()
		self.dense1 = DenseLayer()
		self.dense2 = DenseLayer()
		self.dense3 = DenseLayer()
		self.dense4 = DenseLayer()
		self.softmax = SoftMaxLayer()
		self.loss = LossFunction()
		self.model = {}
		self.totalLoss = []

	def train(self, image_matrix, label):
		########## forward ###########
		output = Neuron(self.convolution1.feed_forward, image_matrix, first=True, image=True)
		output = Neuron(self.relu.relu, output)
		output = Neuron(self.convolution2.feed_forward, output, kernal_size=5, number_of_kernal=10)
		output = Neuron(self.relu.relu, output)
		output = Neuron(self.pooling1.feed_forward, output)
		output = Neuron(self.convolution3.feed_forward, output, kernal_size=4, number_of_kernal=10)
		output = Neuron(self.relu.relu, output)
		output = Neuron(self.pooling2.feed_forward, output)
		output = Neuron(self.drop_out.dropout, output)
		output = Neuron(self.flatten.feed_forward, output)
		output = Neuron(self.dense1.feed_forward, output, number_of_neuron= 100)
		output = Neuron(self.relu.relu, output)
		output = Neuron(self.dense2.feed_forward, output, number_of_neuron= 20)
		output = Neuron(self.relu.relu, output)
		output = Neuron(self.dense3.feed_forward, output, number_of_neuron= 4)
		output = Neuron(self.relu.relu, output)
		output = Neuron(self.dense4.feed_forward, output, number_of_neuron= 3)
		output = Neuron(self.relu.relu, output)
		output = Neuron(self.softmax.feed_forward, output)
		output = Neuron(self.loss.loss, output, label=label)

		############loss#############
		l = output.axon()
		self.totalLoss.append(l)
		
	def back(self, N):
		############# back ##########
		# print(self.totalLoss)
		output = np.sum(self.totalLoss) / N
		print("Cost : {}".format(output))
		output0 = Neuron(self.loss.backpropagation, output, first=True)
		output1 = Neuron(self.softmax.backpropagation, output0)
		output2 = Neuron(self.dense4.backpropagation, output1)
		output3 = Neuron(self.dense3.backpropagation, output2)
		output4 = Neuron(self.dense2.backpropagation, output3)
		output5 = Neuron(self.dense1.backpropagation, output4)
		output6 = Neuron(self.flatten.backpropagation, output5)
		output7 = Neuron(self.pooling2.backpropagation, output6)
		output8 = Neuron(self.convolution3.backpropagation, output7)
		output9 = Neuron(self.pooling1.backpropagation, output8)
		output10 = Neuron(self.convolution2.backpropagation, output9)
		output11 = Neuron(self.convolution1.backpropagation, output10)
		# print(output.axon().shape)

		# print(output4.axon()[1])
		self.model['w1'] = output11.axon()[1]
		self.model['w2'] = output10.axon()[1]
		self.model['w3'] = output8.axon()[1]
		self.model['w4'] = output5.axon()[0]
		self.model['w5'] = output4.axon()[0]
		self.model['w6'] = output3.axon()[0]
		self.model['w7'] = output2.axon()[0]

	def returnModel(self):
		return self.model

	def predict(self):

		pass


network = NeuralNetwork()
for k in range(10):
	count = 0 
	for j in ['open', 'close']:
		for i in load_image(j):
			count += 1
			if j == 'open':
				network.train(i, np.array([1, 0, 0]))
			else:
				network.train(i, np.array([0, 1, 0]))
	network.back(count)

	
file_name = 'model_pickle'
file = open(file_name, 'wb')
pickle.dump(network.returnModel(), file)
file.close()


file_r = open(file_name, 'r')
b = pickle.load(file_r)
# print(b)
# for i in b:
# 	print(b.get(i))
file_r.close()



