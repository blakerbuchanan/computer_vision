import numpy as np
import cv2
import opts

#Import necessary functions
import loadVid
import planarH
import matchPics

# Write script for Q3.1
ar_source_path = '../data/ar_source.mov'
book_path = '../data/book.mov'
img_frames = loadVid.loadVid(ar_source_path)
# np.save('img_frames',img_frames)
book_frames = loadVid.loadVid(book_path)
# np.save('book_frames',book_frames)
cv_cover = cv2.imread('../data/cv_cover.jpg')

# img_frames = np.load('img_frames.npy')
crop_img_frames = img_frames[:,40:320,209:431,:]
# book_frames = np.load('book_frames.npy')


num_of_frames = img_frames.shape[0]
opts = opts.get_opts()
final_frames = []

specific_frames = [48, 49, 50]

for i in range(num_of_frames):
	
	current_source_frame = crop_img_frames[i]
	current_book_frame = book_frames[i]

	# "Harry Potterize" each frame of img_frames to each frame in book.mov
	matches, locs1, locs2 = matchPics.matchPics(current_book_frame, cv_cover, opts)
	x1 = locs1[matches[:,0],:]
	x2 = locs2[matches[:,1],:]
	bestH2to1, inliers = planarH.computeH_ransac(x1,x2,opts)

	if np.sum(inliers) < 6:
		bestH2to1 = np.copy(bestH2to1_copy)

	w = current_book_frame.shape[1]
	h = current_book_frame.shape[0]

	img = current_source_frame
	crop_img = img

	# cv2.imshow("cropped", crop_img)
	# cv2.waitKey(0)

	hp_cover_temp = cv2.resize(crop_img,(cv_cover.shape[1],cv_cover.shape[0]))
	composite_img = planarH.compositeH(bestH2to1, current_book_frame, hp_cover_temp)
	final_frames.append(composite_img)
	size = (composite_img.shape[1], composite_img.shape[0])

	bestH2to1_copy = np.copy(bestH2to1)

	# cv2.imshow('Current Frame',composite_img)
	# cv2.waitKey(0)

# img_array = []
# for filename in glob.glob('C:/New folder/Images/*.jpg'):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width,height)
#     img_array.append(img)
 
out = cv2.VideoWriter('ar.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

for i in range(len(final_frames)):
    out.write(final_frames[i])

out.release()
