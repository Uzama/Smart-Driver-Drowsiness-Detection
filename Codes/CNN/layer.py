from zope.interface import Interface
from zope.interface import implementer

# This is Interface for every layers in Convolutional Neural Netwrok
# When we creating layers we must implement this Interface
# Thus implement the methods define inside the Interface
class Layer(Interface):
	'This a Interface for Layers'

	# this feed forward method
	# the function of this method depend on the layers
	# when we imlementing layers we must implement this method
	def feed_forward(self):
		"To do : implements feed forward for every layers"

	# this is back progation method
	# the function of this method depend on the layers
	# when we imlementing layers we must implement this method
	def backpropagation(self, upstream_derivatives):
		"To do : implements backpropagation for every layers"