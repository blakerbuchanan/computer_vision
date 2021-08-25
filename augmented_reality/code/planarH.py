import numpy as np
import cv2
# import ipdb
import opts

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	num_of_points = x1.shape[0]

	# Construct A matrix from x1 and x2
	A = np.empty((2*num_of_points,9))

	for i in range(num_of_points):
		# Form A
		Ai = np.array([[-x2[i,0], -x2[i,1], -1, 0, 0, 0, x1[i,0]*x2[i,0], x1[i,0]*x2[i,1], x1[i,0]], [0, 0, 0, -x2[i,0], -x2[i,1], -1, x1[i,1]*x2[i,0], x1[i,1]*x2[i,1], x1[i,1]]])
		A[2*i:2*(i+1), :] = Ai

	# Compute SVD solution and extract eigenvector corresponding to smallest eigenvalue
	svd_sol = np.linalg.svd(A)
	h = svd_sol[2][8]

	H2to1 = h.reshape((3,3))

	return H2to1

def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	add_points_x1 = np.sum(x1,axis=0)
	K1 = x1.shape[0]
	centroid_x1 = add_points_x1/K1

	add_points_x2 = np.sum(x2,axis=0)
	K2 = x2.shape[0]
	centroid_x2 = add_points_x2/K2

	#Shift the origin of the points to the centroid
	x1_shift = -x1 + centroid_x1
	x2_shift = -x2 + centroid_x2

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	norm_x1 = np.linalg.norm(x1_shift,axis=1)
	max_x1_idx = np.argmax(norm_x1)
	max_x1_vec = x1_shift[max_x1_idx,:]
	
	norm_x2 = np.linalg.norm(x2_shift,axis=1)
	max_x2_idx = np.argmax(norm_x2)
	max_x2_vec = x2_shift[max_x2_idx,:]
	

	if max_x1_vec[0] == 0.0 or max_x1_vec[1] == 0.0 or max_x2_vec[0] == 0.0 or max_x2_vec[1] == 0.0:
		H2to1 = np.array([])
	else:
		#Similarity transform 1
		T1 = np.array([[1.0/max_x1_vec[0], 0, -centroid_x1[0]/max_x1_vec[0]], [0, 1/max_x1_vec[1], -centroid_x1[1]/max_x1_vec[1]],[0,0,1]])

		#Similarity transform 2
		T2 = np.array([[1.0/max_x2_vec[0], 0, -centroid_x2[0]/max_x2_vec[0]],[0, 1/max_x2_vec[1], -centroid_x2[1]/max_x2_vec[1]],[0,0,1]])

		x1_div = np.tile(max_x1_vec,(x1_shift.shape[0],1))
		x1_temp = np.append(x1,np.ones((K1,1)),axis=1)
		x1_tilde = T1 @ x1_temp.T

		x2_div = np.tile(max_x2_vec,(x2_shift.shape[0],1))
		# x2_tilde = np.divide(x2_shift, x2_div)
		x2_temp = np.append(x2,np.ones((K2,1)),axis=1)
		x2_tilde = T2 @ x2_temp.T

		# # H2to1 = x1_tilde 
		x1_tilde = x1_tilde.T
		x1_tilde = x1_tilde[:,0:2]

		x2_tilde = x2_tilde.T
		x2_tilde = x2_tilde[:,0:2]

		#Compute homography
		H = computeH(x1_tilde,x2_tilde)

		#Denormalization
		H2to1 = np.linalg.inv(T1) @ H @ T2
	

	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	num_of_points = locs1.shape[0]
	sample_size = 4
	d = 0
	bestH2to1 = np.array([])

	for i in range(max_iters):

		# Sample a bunch of points from locs1 and locs 2
		sample = np.random.choice(num_of_points,sample_size)
		x1_sample = locs1[sample,:]
		x2_sample = locs2[sample,:]

		# computeH_norm(sampled points)
		H = computeH_norm(x1_sample,x2_sample)
		if H.size == 0:
			continue

		locs1_hom = np.append(locs1,np.ones((num_of_points,1)),axis=1)
		locs2_hom = np.append(locs2,np.ones((num_of_points,1)),axis=1)
		l_hat = H @ locs2_hom.T
		l_hat[0,:] = np.divide(l_hat[0,:], l_hat[2,:])
		l_hat[1,:] = np.divide(l_hat[1,:], l_hat[2,:])
		l_hat[2,:] = np.divide(l_hat[2,:], l_hat[2,:])

		Hvec = locs1_hom.T - l_hat

		dist = np.linalg.norm(Hvec,axis=0)

		inliers_test = dist < inlier_tol
		inliers_test = inliers_test*1

		num_inliers = np.sum(inliers_test)

		if num_inliers > d:
			# ipdb.set_trace()
			d = num_inliers
			inliers = inliers_test
			bestH2to1 = H


	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	# Create a composite image after warping the template image on top
	# of the image using the homography

	# Note that the homography we compute is from the image to the template;
	# x_template = H2to1*x_photo
	# For warping the template to the image, we need to invert it.
	hp_cover_temp = img
	cv_desk = template

	# hp_cover_temp = cv2.resize(hp_cover,(cv_cover.shape[1],cv_cover.shape[0]))
	# img = cv2.resize(img,(template.shape[1],template.shape[0]))

	# Create mask of same size as template
	mask = np.ones(shape=[hp_cover_temp.shape[0], hp_cover_temp.shape[1], hp_cover_temp.shape[2]], dtype= 'uint8')*255
	
	# Warp mask by appropriate homography
	warped_mask = cv2.warpPerspective(cv2.transpose(mask), H2to1, (cv_desk.shape[0], cv_desk.shape[1]))
	warped_mask = cv2.transpose(warped_mask)
	warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
	warped_mask = cv2.bitwise_not(warped_mask)
	
	warped_img = cv2.warpPerspective(cv2.transpose(hp_cover_temp), H2to1, (cv_desk.shape[0], cv_desk.shape[1]))
	# warped_img = cv2.warpPerspective(cv2.transpose(img), bestH2to1, (template.shape[0], template.shape[1]))

	warped_img = cv2.transpose(warped_img)
	# cv2.imwrite('perspective.png', warped_img)

	# hp_cover_mask = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
	# _, mask = cv2.threshold(hp_cover_mask,50,255,cv2.THRESH_BINARY_INV)
	masked_img = cv2.bitwise_and(cv_desk, cv_desk, mask=warped_mask)
	composite_img = masked_img + warped_img
	
	# Warp mask by appropriate homography

	# Warp template by appropriate homography

	#Use mask to combine the warped template and the image
	composite_img = masked_img + warped_img

	return composite_img
