import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# getting the current path of this file
CURRENT_PATH = os.path.curdir

# adding the path to images
IMAGE_PATH = os.path.join(CURRENT_PATH, 'images')

# setting open eye images and closed eye images
OPEN_EYE = os.path.join(IMAGE_PATH, 'open')
CLOSE_EYE = os.path.join(IMAGE_PATH, 'close')

# define the path to test images
TEST = os.path.join(IMAGE_PATH, 'test')

# store the number, thus how many images in both open and close category
NUMBER_OF_IMAGES = 20

# this is generator method
# which yields images according to
# category parameter
def load_image(category):

	# if category is open the method yields open eye images
	if category == 'open':
		# iterating through all images in side the open eye folder
		for i in range(1, NUMBER_OF_IMAGES + 1):
			# open an image
			image = Image.open(os.path.join(OPEN_EYE, '{}.png'.format(i)))
			image = image.resize((32,32), Image.LANCZOS)
			# before yield, image change to numpy array
			image = np.array(image)

			# yield the image array
			yield image

	# if category is close the method yields close eye images
	elif category == 'close':
		for i in range(1, NUMBER_OF_IMAGES + 1):
			# doing same thing for close eye
			image = Image.open(os.path.join(CLOSE_EYE, '{}.png'.format(i)))
			image = image.resize((32,32), Image.LANCZOS)
			image = np.array(image)
			yield image

	# same process for testing, similar open and close
	elif category == 'test':
		for i in range(1, NUMBER_OF_IMAGES + 1):
			image = Image.open(os.path.join(TEST, '{}.png'.format(i)))
			image = image.resize((32,32), Image.LANCZOS)
			image = np.array(image)
			yield image

