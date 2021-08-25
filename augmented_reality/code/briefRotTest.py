import numpy as np
import cv2
from matchPics import matchPics
# import ipdb
import scipy.ndimage
import matplotlib.pyplot as plt
import opts
# from tqdm import trange

#Q2.1.6
#Read the image and convert to grayscale, if necessary
img = cv2.imread('../data/cv_cover.jpg')
opts = opts.get_opts()

# if img.ndim > 1:
#         # Must be RGB, convert to grayscale
#         img = np.dot(img, [0.299, 0.587, 0.114])

#         # img = np.tile(img[:,:,np.newaxis],(1,1,3))

# img = np.array(img).astype(np.float32)

hist_count = np.zeros(36)

for i in range(36):
	#Rotate Image
	img_rot = scipy.ndimage.rotate(img,10*i,reshape=False)

	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(img, img_rot, opts)

	num_of_matches = matches.shape[0]
	# num_of_matches = locs1.shape[0]

	#Update histogram
	# hist_count.append(num_of_matches)
	hist_count[i] = num_of_matches

#Display histogram
angles = [i for i in range(36)]

disp_hist = plt.bar(angles,hist_count)

plt.xlabel('Rotation angle / 10')
plt.ylabel('Number of matches')


plt.show()