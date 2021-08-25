import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

import ipdb
import matplotlib.pyplot as plt

def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	
	# I1 = np.array(I1).astype(np.float32)/255
	# I2 = np.array(I2).astype(np.float32)/255
	

	#Convert Images to GrayScale
	# I1_gray = np.dot(I1, [0.299, 0.587, 0.114])
	# I2_gray = np.dot(I2, [0.299, 0.587, 0.114])

	I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

	# # Let's take a look at the images
	# f, axes = plt.subplots(2,1)
	# f.set_size_inches(8, 8)
	# axes[0].imshow(I1_gray, cmap=plt.get_cmap('gray'))
	# axes[0].set_title('Image 1')
	# axes[0].axis('off')

	# # axes[0, 1].imshow(I1_gray, cmap=plt.get_cmap('gray'))
	# axes[1].imshow(I2_gray, cmap=plt.get_cmap('gray'))
	# axes[1].set_title('Image 2')
	# axes[1].axis('off')
	# plt.show()

	#Detect Features in Both Images
	locs1 = corner_detection(I1_gray,sigma)
	locs2 = corner_detection(I2_gray,sigma)
	
	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1_gray,locs1)
	desc2, locs2 = computeBrief(I2_gray,locs2)

	#Match features using the descriptors
	matches = briefMatch(desc1,desc2,ratio)

	return matches, locs1, locs2
