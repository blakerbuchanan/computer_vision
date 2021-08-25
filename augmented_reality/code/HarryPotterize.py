import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
import matchPics
import planarH

#Write script for Q2.2.4
opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
img_frames = np.load('img_frames.npy')
book_frames = np.load('book_frames.npy')


matches, locs1, locs2 = matchPics.matchPics(cv_desk, cv_cover, opts)
x1 = locs1[matches[:,0],:]
x2 = locs2[matches[:,1],:]

bestH2to1, inliers = planarH.computeH_ransac(x1,x2,opts)

# hp_cover_temp = cv2.resize(hp_cover,(cv_cover.shape[1],cv_cover.shape[0]))
# # img = cv2.resize(img,(template.shape[1],template.shape[0]))

# warped_img = cv2.warpPerspective(cv2.transpose(hp_cover_temp), bestH2to1, (cv_desk.shape[0], cv_desk.shape[1]))
# # warped_img = cv2.warpPerspective(cv2.transpose(img), bestH2to1, (template.shape[0], template.shape[1]))

# warped_img = cv2.transpose(warped_img)
# cv2.imwrite('perspective.png', warped_img)

# hp_cover_mask = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
# _, mask = cv2.threshold(hp_cover_mask,10,255,cv2.THRESH_BINARY_INV)
# masked_img = cv2.bitwise_and(cv_desk, cv_desk, mask=mask)
# composite_img = masked_img + warped_img

hp_cover_temp = cv2.resize(hp_cover,(cv_cover.shape[1],cv_cover.shape[0]))
composite_img = planarH.compositeH(bestH2to1, cv_desk, hp_cover_temp)
print(np.sum(inliers))
# cv2.imwrite('composite_img.png', composite_img)

cv2.imshow('Final',composite_img)
cv2.waitKey(0)


# composite_img = compositeH(bestH2to1, template, img)
