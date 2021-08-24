from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts
import cv2
import ipdb
import skimage
import scipy

def main():
    opts = get_opts()

    ## Q1.1
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # img_shape = img.shape
    # height = img_shape[0]
    # width = img_shape[1]
    # depth = img_shape[2]

    # img_lab = skimage.color.rgb2lab(img)

    # filter_responses = visual_words.extract_filter_responses(opts, img_lab)

    # util.display_filter_responses(opts,filter_responses)

    ## Q1.2
    # n_cpu = util.get_num_CPU()
    # visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    ## Q1.3
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # img_path2 = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    
    # img_path3 = join(opts.data_dir, 'windmill/sun_bcyuphldelrgtuwd.jpg')

    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)

    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))

    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # filter_height = filter_responses.shape[0]
    # filter_width = filter_responses.shape[1]
    # filter_depth = filter_responses.shape[2]
    
    # filter_responses_distance = filter_responses.reshape((filter_height*filter_width,filter_depth))
    # distance_matrix = scipy.spatial.distance.cdist(filter_responses_distance,dictionary,metric='euclidean')
    # cluster_assignment = np.argmin(distance_matrix,axis=1)
    # wordmap = cluster_assignment.reshape((filter_height,filter_width))
    
    # ipdb.set_trace()
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # wordmap_test = wordmap.reshape((wordmap.shape[0]*wordmap.shape[1]))
    # util.visualize_wordmap(wordmap)

    # img2 = Image.open(img_path2)
    # img2 = np.array(img2).astype(np.float32)

    # filter_responses2 = visual_words.extract_filter_responses(opts, img2)
    # filter_height2 = filter_responses2.shape[0]
    # filter_width2 = filter_responses2.shape[1]
    # filter_depth2 = filter_responses2.shape[2]
    
    # filter_responses_distance2 = filter_responses2.reshape((filter_height2*filter_width2,filter_depth2))
    # distance_matrix2 = scipy.spatial.distance.cdist(filter_responses_distance2,dictionary,metric='euclidean')
    # cluster_assignment = np.argmin(distance_matrix2,axis=1)
    # wordmap2 = cluster_assignment.reshape((filter_height2,filter_width2))

    # img3 = Image.open(img_path3)
    # img3 = np.array(img3).astype(np.float32)

    # filter_responses3 = visual_words.extract_filter_responses(opts, img3)
    # filter_height3 = filter_responses3.shape[0]
    # filter_width3 = filter_responses3.shape[1]
    # filter_depth3 = filter_responses3.shape[2]
    
    # filter_responses_distance3 = filter_responses3.reshape((filter_height3*filter_width3,filter_depth3))
    # distance_matrix3 = scipy.spatial.distance.cdist(filter_responses_distance3,dictionary,metric='euclidean')
    # cluster_assignment = np.argmin(distance_matrix3,axis=1)
    # wordmap3 = cluster_assignment.reshape((filter_height3,filter_width3))

    # img = img/255
    # fig, axes = plt.subplots(2, 3)
    # axes[0, 0].imshow(img)
    # axes[1, 0].imshow(wordmap,cmap='gnuplot2')
    # axes[0,0].axis('off')
    # axes[1,0].axis('off')

    # img2 = img2/255
    # axes[0,1].imshow(img2)
    # axes[1,1].imshow(wordmap2,cmap='gnuplot2')
    # axes[0,1].axis('off')
    # axes[1,1].axis('off')

    # img3 = img3/255
    # axes[0,2].imshow(img3)
    # axes[1,2].imshow(wordmap3,cmap='gnuplot2')
    # axes[0,2].axis('off')
    # axes[1,2].axis('off')

    # plt.show()

    ## Q2.1-2.4
    # n_cpu = util.get_num_CPU()
    # visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.1
    # hist = visual_recog.get_feature_from_wordmap(opts,wordmap)

    # Q2.2
    # hist_all = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    
    ## Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)

    # print(conf)
    # print(accuracy)
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
