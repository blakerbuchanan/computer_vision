import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words
import ipdb
import math
from sklearn.metrics import confusion_matrix
import skimage
from tqdm import trange

def get_feature_from_wordmap(opts, wordmap):

    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    height = wordmap.shape[0]
    width = wordmap.shape[1]
    # hist_input = wordmap.reshape((height*width))

    K = opts.K
    hist_output = np.histogram(wordmap,bins = range(K+1), density = False)

    # ----- TODO -----
    return hist_output

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.
    
    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    K = opts.K
    L = opts.L
    height = int(wordmap.shape[0]/(2**L))
    width = int(wordmap.shape[1]/(2**L))

    hist_container = np.empty((2**L,2**L,K))

    num_of_hists = int((4**(L+1)-1)/3)

    hist_vec = np.empty((num_of_hists*K))

    weight = 2**(-1)
    t = 0

    # Compute the histograms for the "sub-images" and store in hist_list
    for i in range(2**L):
        for j in range(2**L):
            sub_image = wordmap[height*i:height*i + height, width*j:width*j + width]
            
            hist_container[i,j,:] = get_feature_from_wordmap(opts, sub_image)[0] #:np.histogram(sub_image,K)[0]
            hist_vec[t*K: (t+1)*K] = hist_container[i,j,:]/np.sum(hist_container[i,j,:])*float(weight)

            if np.sum(hist_container[i,j,:]) == 0:
                ipdb.set_trace()

            t = t + 1

    # Aggregation
    for l in reversed(range(L)):

        hist_list_0 = np.empty((2**l, 2**l, K))

        if l==0 or l==1:
            weight = 2**(-L)
        else:
            weight = 2**(l-L-1)

        for i in range(2**l):
            for j in range(2**l):
                hist_list_0[i,j,:] = hist_container[2*i,2*j,:] + hist_container[2*i,2*j+1,:] + hist_container[2*i+1,2*j,:] + hist_container[2*i+1,2*j+1,:]
                hist_vec[t*K:(t+1)*K] = hist_list_0[i,j,:]/np.sum(hist_list_0[i,j,:])*weight

                if np.sum(hist_list_0[i,j,:]) == 0:
                    ipdb.set_trace()

                t = t + 1

        hist_container = np.copy(hist_list_0)

    hist_vec = hist_vec/np.sum(hist_vec)

    return hist_vec
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    
    
    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    img_path = join(opts.data_dir, img_path)
    img = Image.open(img_path)

    # if img.ndim < 3:
    #     # Must be grayscale. Generate three repeated channels to make 3-channel
    #     img = np.tile[img[:,:,np.newaxis],(1,1,3)]

    # if all(img_temp <= 1.0) and all(img_temp >= 0):
    #     # Do nothing
    #     print('Satisfied')
    # else:
    #     img = np.array(img).astype(np.float32)/255

    img = np.array(img).astype(np.float32)

    # img_shape = img.shape
    # img_lab = skimage.color.rgb2lab(img)

    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)

    return feature

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    K = opts.K
    L = opts.L
    M = int(K*(4**(L+1) - 1)/3)
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    T = len(train_files) # Number of training images
    features = np.zeros((T,M))

    print('Build recognition system:')
    for i in trange(T):
        # Get histogram for image
        feature = get_image_feature(opts, train_files[i], dictionary)
        features[i,:] = feature

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)
    
    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    N = histograms.shape[0]

    word_hist_tile = np.tile(word_hist, (N,1))
    minima = np.minimum(word_hist_tile, histograms)
    sim = np.sum(minima,axis=1)

    return sim    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    trained_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    prediction = []
    accuracy = 0
    
    print('Evaluate recognition system:')
    for i in trange(len(test_files)):
        test_feature = get_image_feature(opts, test_files[i], dictionary)
        dist_test = distance_to_set(test_feature, trained_system['features'])

        max_index = np.argmax(dist_test)
        prediction.append(trained_labels[max_index])

        if trained_labels[max_index] == test_labels[i]:
            accuracy = accuracy + 1

    accuracy_final = (accuracy / len(test_files))*100.0
    confusion = confusion_matrix(np.array(test_labels), np.array(prediction))

    return confusion, accuracy_final

